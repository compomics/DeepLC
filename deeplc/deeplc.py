"""
Main code used to generate LC retention time predictions.

This provides the main interface. For the library versions see the .yml file
"""

from __future__ import annotations

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]
__credits__ = [
    "Robbin Bouwmeester",
    "Ralf Gabriels",
    "Arthur Declercq",
    "Alireza Nameni",
    "Lennart Martens",
    "Sven Degroeve",
]

import copy
import logging
import multiprocessing
import os
import random
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from psm_utils import PSM, Peptidoform, PSMList
from psm_utils.io import read_file
from torch.utils.data import DataLoader

from deeplc.calibration import Calibration, SplineTransformerCalibration
from deeplc._data import DeepLCDataset
from deeplc._finetune import DeepLCFineTuner

# If CLI/GUI/frozen: disable warnings before importing
IS_CLI_GUI = os.path.basename(sys.argv[0]) in ["deeplc", "deeplc-gui"]
IS_FROZEN = getattr(sys, "frozen", False)
if IS_CLI_GUI or IS_FROZEN:
    warnings.filterwarnings("ignore", category=UserWarning)

# Default models, will be used if no other is specified. If no best model is
# selected during calibration, the first model in the list will be used.
DEEPLC_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODELS = [
    "mods/full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.pt",
    "mods/full_hc_PXD005573_pub_8c22d89667368f2f02ad996469ba157e.pt",
    "mods/full_hc_PXD005573_pub_cb975cfdd4105f97efa0b3afffe075cc.pt",
]
DEFAULT_MODELS = [os.path.join(DEEPLC_DIR, m) for m in DEFAULT_MODELS]


logger = logging.getLogger(__name__)


def predict(
    psm_list: PSMList | None = None,
    model_files: str | list[str] | None = None,
    calibrator: Calibration | None = None,
    batch_size: int = 1024,
    single_model_mode: bool = False,
):
    """
    Make predictions for sequences, in batches if required.

    Parameters
    ----------
    psm_list
        PSMList object containing the peptidoforms to predict for.
    model_files
        Model file (or files) to use for prediction. If None, the default model is used.
    calibrator
        Calibrator object to use for calibration. If None, no calibration is performed.
    batch_size
        How many samples per batch to load (default: 1).
    single_model_mode
        Whether to use a single model instead of multiple default models. Only applies if
        model_file is None.

    Returns
    -------
    np.array
        predictions

    """
    if len(psm_list) == 0:
        return []

    # Setup dataset and dataloader
    dataset = DeepLCDataset(psm_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if model_files is not None:
        if isinstance(model_files, str):
            model_files = [model_files]
        elif isinstance(model_files, list):
            model_files = model_files
        else:
            raise ValueError("Invalid model name provided.")
    else:
        model_files = [DEFAULT_MODELS[0]] if single_model_mode else DEFAULT_MODELS

    # Get predictions; iterate over models if multiple were selected
    model_predictions = []
    for model_f in model_files:
        # Load model
        model = torch.load(model_f, weights_only=False, map_location=torch.device("cpu"))
        model.eval()

        # Predict
        ret_preds = []
        with torch.no_grad():
            for features, _ in loader:
                batch_preds = model(*features)
                ret_preds.append(batch_preds.detach().cpu().numpy())
                raise Exception()

        # Concatenate predictions
        ret_preds = np.concatenate(ret_preds, axis=0)

        # TODO: Bring outside of model loop?
        # Calibrate
        if calibrator is not None:
            ret_preds = calibrator.transform(ret_preds)

        model_predictions.append(ret_preds)

    # Average the predictions from all models
    averaged_predictions = np.mean(model_predictions, axis=0)

    return averaged_predictions


# TODO: Split-of transfer learning?
def calibrate(
    psm_list: PSMList | None = None,
    model_files: str | list[str] | None = None,
    location_retraining_models: str = "",
    sample_for_calibration_curve: int | None = None,
    return_plotly_report=False,
    n_jobs: int | None = None,
    batch_size: int = int(1e6),
    fine_tune: bool = False,
    n_epochs: int = 20,
    calibrator: Calibration | None = None,
) -> dict | None:
    """
    Find best model and calibrate.

    Parameters
    ----------
    psm_list
        PSMList object containing the peptidoforms to predict for.
    model_files
        Path to one or mode models to test and calibrat for. If a list of models is passed,
        the best performing one on the calibration data will be selected.
    location_retraining_models
        Location to save the retraining models; if None, a temporary directory is used.
    sample_for_calibration_curve
        Number of PSMs to sample for calibration curve; if None, all provided PSMs are used.
    return_plotly_report
        Whether to return a plotly report with the calibration results.
    n_jobs
        Number of jobs to use for parallel processing; if None, the number of CPU cores is used.
    batch_size
        Batch size to use for training and prediction; default is 1e6, which means all data is
        processed in one batch.
    fine_tune
        Whether to fine-tune the model on the provided PSMs. If True, the first model in
        model_files will be used for fine-tuning.
    n_epochs
        Number of epochs to use for fine-tuning the model. Default is 20.

    Returns
    -------
    dict | None
        Dictionary with plotly report information or None.

    """
    if None in psm_list["retention_time"]:
        raise ValueError("Not all PSMs have an observed retention time.")

    n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs

    if calibrator is None:
        calibrator = SplineTransformerCalibration()

    # Ensuring self.model is list of strings
    model_files = model_files or DEFAULT_MODELS
    if isinstance(model_files, str):
        model_files = [model_files]

    logger.debug("Start to calibrate predictions ...")
    logger.debug(f"Ready to find the best model out of: {model_files}")

    if fine_tune:
        logger.debug("Starting model fine-tuning...")
        dataset = DeepLCDataset(psm_list)

        base_model_path = model_files[0]
        base_model = torch.load(
            base_model_path, weights_only=False, map_location=torch.device("cpu")
        )
        base_model.eval()

        fine_tuner = DeepLCFineTuner(
            model=base_model,
            train_data=dataset,
            batch_size=batch_size,
            epochs=n_epochs,
        )
        # fine_tuner._freeze_layers()
        fine_tuned_model = fine_tuner.fine_tune()

        if location_retraining_models:
            os.makedirs(location_retraining_models, exist_ok=True)
            temp_dir_obj = TemporaryDirectory()
            t_dir_models = temp_dir_obj.name
        else:
            t_dir_models = location_retraining_models
            os.makedirs(t_dir_models, exist_ok=True)

        # Define path to save fine-tuned model
        fine_tuned_model_path = os.path.join(t_dir_models, "fine_tuned_model.pth")
        torch.save(fine_tuned_model, fine_tuned_model_path)
        model_files = [fine_tuned_model_path]

    # Limit calibration to a subset of PSMs if specified
    if sample_for_calibration_curve:
        psm_list = random.sample(list(psm_list), sample_for_calibration_curve)
        measured_tr = [psm.retention_time for psm in psm_list]

    best_perf = float("inf")
    best_calibrator = {}
    mod_calibrator = {}
    pred_dict = {}
    mod_dict = {}

    for model_name in model_files:
        logger.debug(f"Trying out the following model: {model_name}")
        predicted_tr = predict(psm_list, calibrate=False, model_name=model_name)

        model_calibrator = copy.deepcopy(calibrator)

        if isinstance(model_calibrator, SplineTransformerCalibration):
            model_calibrator.fit(predicted_tr, measured_tr, simplified=fine_tune)
        else:
            model_calibrator.fit(predicted_tr, measured_tr)

        # TODO: Use pathlib to get the model base name
        m_name = model_name.split("/")[-1]

        # Get new predictions with calibration
        preds = predict(psm_list, calibrate=True, model_files=[model_name])

        # Save the predictions and calibration parameters
        # TODO Double nested dict not required without CALLC functionality?
        m_group_name = "_".join(m_name.split("_")[:-1])
        pred_dict.setdefault(m_group_name, {})[model_name] = preds
        mod_dict.setdefault(m_group_name, {})[model_name] = model_name
        mod_calibrator.setdefault(m_group_name, {})[model_name] = model_calibrator

    # Find best-performing model, including each model's calibration
    for m_name in pred_dict:
        # TODO: Use numpy methods
        preds = [sum(a) / len(a) for a in zip(*list(pred_dict[m_name].values()), strict=True)]
        perf = sum(abs(np.array(measured_tr) - np.array(preds)))  # MAE

        logger.debug(f"For {m_name} model got a performance of: {perf / len(preds)}")

        if perf < best_perf:  # Lower is better, as MAE is used
            m_group_name = m_name

            # TODO is deepcopy really required?
            best_calibrator = copy.deepcopy(mod_calibrator[m_group_name])
            best_model = copy.deepcopy(mod_dict[m_group_name])
            best_perf = perf

    logger.debug(f"Model with the best performance got selected: {best_model}")


    return best_model, best_calibrator


# TODO: Move to psm_utils?
def _file_to_psm_list(input_file: str | Path) -> PSMList:
    """Read a file into a PSMList, optionally mapping MaxQuant modifications labels."""
    psm_list = read_file(input_file)
    if "msms" in input_file and ".txt" in input_file:
        mapper = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "unimod/map_mq_file.csv",
            ),
            index_col=0,
        )["value"].to_dict()
        psm_list.rename_modifications(mapper)

    return psm_list
