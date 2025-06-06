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
from rich.progress import track
from torch.nn import Module
from torch.utils.data import DataLoader

from deeplc._data import DeepLCDataset, get_targets
from deeplc._finetune import DeepLCFineTuner
from deeplc.calibration import Calibration, SplineTransformerCalibration

# If CLI/GUI/frozen: disable warnings before importing
IS_CLI_GUI = os.path.basename(sys.argv[0]) in ["deeplc", "deeplc-gui"]
IS_FROZEN = getattr(sys, "frozen", False)
if IS_CLI_GUI or IS_FROZEN:
    warnings.filterwarnings("ignore", category=UserWarning)

# Default models, will be used if no other is specified. If no best model is
# selected during calibration, the first model in the list will be used.
DEEPLC_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL = "mods/full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.pt"
DEFAULT_MODEL = os.path.join(DEEPLC_DIR, DEFAULT_MODEL)


logger = logging.getLogger(__name__)


def predict(
    psm_list: PSMList | None = None,
    model: str | list[str] | None = None,
    num_workers: int = 4,
    batch_size: int = 1024,
):
    """
    Make predictions for sequences, in batches if required.

    Parameters
    ----------
    psm_list
        PSMList object containing the peptidoforms to predict for.
    model_file
        Model file to use for prediction. If None, the default model is used.
    batch_size
        How many samples per batch to load (default: 1024).

    Returns
    -------
    np.array
        predictions

    """
    # Shortcut if empty PSMList is provided
    if not psm_list:
        return np.array([])

    # Avoid predicting repeated PSMs
    unique_peptidoforms, inverse_indices = _get_unique_peptidoforms(psm_list)

    # Setup dataset and dataloader
    dataset = DeepLCDataset(unique_peptidoforms, target_retention_times=None)
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    # Get model files
    model = model or DEFAULT_MODEL

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model on specified device
    model = _load_model(model=model, device=device, eval=True)

    # Predict
    predictions = []
    with torch.no_grad():
        for features, _ in track(loader):
            features = [feature_tensor.to(device) for feature_tensor in features]
            batch_preds = model(*features)
            predictions.append(batch_preds.detach().cpu().numpy())

    # Concatenate predictions and reorder to match original PSMList order
    predictions = np.concatenate(predictions, axis=0)[inverse_indices]

    return predictions


# TODO: Split-of transfer learning?
def calibrate(
    psm_list: PSMList | None = None,
    model_files: str | list[str] | None = None,
    location_retraining_models: str = "",
    sample_for_calibration_curve: int | None = None,
    n_jobs: int | None = None,
    batch_size: int = int(1e6),
    fine_tune: bool = False,
    n_epochs: int = 20,
    calibrator: Calibration | None = None,
) -> tuple[str, dict[str, Calibration]]:
    """
    Find best model and calibrate.

    Parameters
    ----------
    psm_list
        PSMList object containing the peptidoforms to predict for.
    model_files
        Path to one or mode models to test and calibrate for. If a list of models is passed,
        the best performing one on the calibration data will be selected.
    location_retraining_models
        Location to save the retraining models; if None, a temporary directory is used.
    sample_for_calibration_curve
        Number of PSMs to sample for calibration curve; if None, all provided PSMs are used.
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

    """
    if None in psm_list["retention_time"]:
        raise ValueError("Not all PSMs have an observed retention time.")

    n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs

    if calibrator is None:
        calibrator = SplineTransformerCalibration()

    # Ensuring self.model is list of strings
    model_files = model_files or DEFAULT_MODEL
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
    model_calibrators = {}
    pred_dict = {}
    mod_dict = {}

    for model_name in model_files:
        logger.debug(f"Trying out the following model: {model_name}")
        predicted_tr = predict(psm_list, calibrator=calibrator, model_name=model_name)

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
        model_calibrators.setdefault(m_group_name, {})[model_name] = model_calibrator

    # Find best-performing model, including each model's calibration
    for m_name in pred_dict:
        # TODO: Use numpy methods
        preds = [sum(a) / len(a) for a in zip(*list(pred_dict[m_name].values()), strict=True)]
        perf = sum(abs(np.array(measured_tr) - np.array(preds)))  # MAE

        logger.debug(f"For {m_name} model got a performance of: {perf / len(preds)}")

        if perf < best_perf:  # Lower is better, as MAE is used
            m_group_name = m_name

            # TODO is deepcopy really required?
            best_calibrator = copy.deepcopy(model_calibrators[m_group_name])
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


def _get_unique_peptidoforms(psm_list: PSMList) -> tuple[PSMList, np.ndarray]:
    """Get PSMs with unique peptidoforms and their inverse indices."""
    peptidoform_strings = np.array([str(psm.peptidoform) for psm in psm_list])
    unique_peptidoforms, inverse_indices = np.unique(peptidoform_strings, return_inverse=True)
    return unique_peptidoforms, inverse_indices


def _load_model(
    model: Module | Path | str | None = None,
    device: str | None = None,
    eval: bool = False,
) -> Module:
    """Load a model from a file or return the default model if none is provided."""
    # If no model is provided, use the default model
    model = model or DEFAULT_MODEL

    # If device is not specified, use the default device (GPU if available, else CPU)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from file if a path is provided
    if isinstance(model, str | Path):
        model = torch.load(model, weights_only=False, map_location=device)
    elif not isinstance(model, Module):
        raise TypeError(f"Expected a PyTorch Module or a file path, got {type(model)} instead.")

    # Ensure the model is on the specified device
    model.to(device)

    # Set the model to evaluation or training mode based on the eval flag
    if eval:
        model.eval()
    else:
        model.train()

    return model


