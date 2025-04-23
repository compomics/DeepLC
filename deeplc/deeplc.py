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

# Default models, will be used if no other is specified. If no best model is
# selected during calibration, the first model in the list will be used.
import copy
import gc
import logging
import multiprocessing
import multiprocessing.dummy
import os
import random
import sys
import warnings
from configparser import ConfigParser
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from dask import compute, delayed
from psm_utils import PSM, Peptidoform, PSMList
from psm_utils.io import read_file
from psm_utils.io.peptide_record import peprec_to_proforma
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from torch.utils.data import DataLoader, Dataset

from deeplc._exceptions import CalibrationError
from deeplc.feat_extractor import aggregate_encodings, encode_peptidoform
from deeplc.trainl3 import train_elastic_net

# If CLI/GUI/frozen: disable warnings before importing
IS_CLI_GUI = os.path.basename(sys.argv[0]) in ["deeplc", "deeplc-gui"]
IS_FROZEN = getattr(sys, "frozen", False)
if IS_CLI_GUI or IS_FROZEN:
    warnings.filterwarnings("ignore", category=UserWarning)

DEEPLC_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODELS = [
    "mods/full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.pt",
    "mods/full_hc_PXD005573_pub_8c22d89667368f2f02ad996469ba157e.pt",
    "mods/full_hc_PXD005573_pub_cb975cfdd4105f97efa0b3afffe075cc.pt",
]
DEFAULT_MODELS = [os.path.join(DEEPLC_DIR, dm) for dm in DEFAULT_MODELS]

LIBRARY = {}


# TODO: Keep for torch?
# Set to force CPU calculations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


logger = logging.getLogger(__name__)


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def divide_chunks(list_, n_chunks):
    """Yield successive n-sized chunks from list_."""
    for i in range(0, len(list_), n_chunks):
        yield list_[i : i + n_chunks]


class DeepLCDataset(Dataset):
    """
    Custom Dataset class for DeepLC used for loading features from peptide sequences.

    Parameters
    ----------
    X : ndarray
        Feature matrix for input data.
    X_sum : ndarray
        Feature matrix for sum of input data.
    X_global : ndarray
        Feature matrix for global input data.
    X_hc : ndarray
        Feature matrix for high-order context features.
    target : ndarray, optional
        The target retention times. Default is None.
    """

    def __init__(self, X, X_sum, X_global, X_hc, target=None):
        self.X = torch.from_numpy(X).float()
        self.X_sum = torch.from_numpy(X_sum).float()
        self.X_global = torch.from_numpy(X_global).float()
        self.X_hc = torch.from_numpy(X_hc).float()

        if target is not None:
            self.target = torch.from_numpy(target).float()  # Add target values if provided
        else:
            self.target = None  # If no target is provided, set it to None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.target is not None:
            # Return both features and target during training
            return (
                self.X[idx],
                self.X_sum[idx],
                self.X_global[idx],
                self.X_hc[idx],
                self.target[idx],
            )
        else:
            # Return only features during prediction
            return (self.X[idx], self.X_sum[idx], self.X_global[idx], self.X_hc[idx])


class DeepLCFineTuner:
    """
    Class for fine-tuning a DeepLC model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to fine-tune.
    train_data : torch.utils.data.Dataset
        Dataset containing the training data.
    device : str, optional, default='cpu'
        The device on which to run the model ('cpu' or 'cuda').
    learning_rate : float, optional, default=0.001
        The learning rate for the optimizer.
    epochs : int, optional, default=10
        Number of training epochs.
    batch_size : int, optional, default=256
        Batch size for training.
    validation_data : torch.utils.data.Dataset or None, optional
        If provided, used directly for validation. Otherwise, a fraction of
        `train_data` will be held out.
    validation_split : float, optional, default=0.1
        Fraction of `train_data` to reserve for validation when
        `validation_data` is None.
    patience : int, optional, default=5
        Number of epochs with no improvement on validation loss before stopping.
    """

    def __init__(
        self,
        model,
        train_data,
        device="cpu",
        learning_rate=0.001,
        epochs=10,
        batch_size=256,
        validation_data=None,
        validation_split=0.1,
        patience=5,
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.patience = patience

    def _freeze_layers(self, unfreeze_keywords="33_1"):
        """
        Freezes all layers except those that contain the unfreeze_keyword
        in their name.
        """

        for name, param in self.model.named_parameters():
            param.requires_grad = unfreeze_keywords in name

    def prepare_data(self, data, shuffle=True):
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def fine_tune(self):
        logger.debug("Starting fine-tuning...")
        if self.validation_data is None:
            # Split the training data into training and validation sets
            val_size = int(len(self.train_data) * self.validation_split)
            train_size = len(self.train_data) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.train_data, [train_size, val_size]
            )
        else:
            train_dataset = self.train_data
            val_dataset = self.validation_data
        train_loader = self.prepare_data(train_dataset)
        val_loader = self.prepare_data(val_dataset, shuffle=False)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate
        )
        loss_fn = torch.nn.L1Loss()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
            for batch in train_loader:
                batch_X, batch_X_sum, batch_X_global, batch_X_hc, target = batch

                target = target.view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_X_sum, batch_X_global, batch_X_hc)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_X, batch_X_sum, batch_X_global, batch_X_hc, target = batch
                    target = target.view(-1, 1)
                    outputs = self.model(batch_X, batch_X_sum, batch_X_global, batch_X_hc)
                    val_loss += loss_fn(outputs, target).item()
            avg_val_loss = val_loss / len(val_loader)

            logger.debug(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}"
            )
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.debug(f"Early stopping triggered {epoch + 1}")
                    break
        self.model.load_state_dict(best_model_wts)
        return self.model


class DeepLC:
    """
    DeepLC predictor.

    Methods
    -------
    calibrate_preds
        Find best model and calibrate
    make_preds
        Make predictions

    """

    library = {}

    def __init__(
        self,
        main_path: str | None = None,
        path_model: str | None = None,
        verbose: bool = True,
        bin_distance: float = 2.0,
        dict_cal_divider: int = 50,
        split_cal: int = 50,
        n_jobs: int | None = None,
        config_file: str | None = None,
        f_extractor: None = None,
        cnn_model: bool = True,
        # batch_num: int = 250000,
        batch_num: int = int(1e6),
        batch_num_tf: int = 1024,
        write_library: bool = False,
        use_library: str | None = None,
        reload_library: bool = False,
        pygam_calibration: bool = True,
        deepcallc_mod: bool = False,
        deeplc_retrain: bool = False,
        predict_ccs: bool = False,
        n_epochs: int = 20,
        single_model_mode: bool = True,
    ):
        """
        Initialize the DeepLC predictor.

        Parameters
        ----------
        main_path
            Main path of the module.
        path_model
            Path to prediction model(s); if not provided, the best default model is selected based
            on calibration peptides.
        verbose
            Turn logging on/off.
        bin_dist
            Precision factor for calibration lookup.
        dict_cal_divider
            Divider that sets the precision for fast lookup of retention times in calibration; e.g.
            10 means a precision of 0.1 between the calibration anchor points
        split_cal
            Number of splits in the chromatogram for piecewise linear calibration.
        n_jobs
            Number of CPU threads to use.
        config_file
            Path to a configuration file.
        f_extractor
            Deprecated.
        cnn_model
            Flag indicating whether to use the CNN model.
        batch_num
            Prediction batch size (in peptides); lower values reduce memory footprint.
        batch_num_tf
            Batch size for TensorFlow predictions.
        write_library
            Whether to append new predictions to a library for faster future access.
        use_library
            Library file to read from or write to for prediction caching.
        reload_library
            Whether to reload the prediction library.
        pygam_calibration
            Flag to enable calibration using PyGAM.
        deepcallc_mod
            Flag specific to deepcallc mode.
        deeplc_retrain
            Flag indicating whether to perform retraining (transfer learning) of prediction models.
        predict_ccs
            Flag to control prediction of CCS values.
        n_epochs
            Number of epochs used in retraining if deeplc_retrain is enabled.
        single_model_mode
            Flag to use a single model instead of multiple default models.

        """
        # if a config file is defined overwrite standard parameters
        if config_file:
            cparser = ConfigParser()
            cparser.read(config_file)
            dict_cal_divider = cparser.getint("DeepLC", "dict_cal_divider")
            split_cal = cparser.getint("DeepLC", "split_cal")
            n_jobs = cparser.getint("DeepLC", "n_jobs")

        self.main_path = main_path or os.path.dirname(os.path.realpath(__file__))
        self.path_model = self._get_model_paths(path_model, single_model_mode)
        self.verbose = verbose
        self.bin_distance = bin_distance
        self.dict_cal_divider = dict_cal_divider
        self.split_cal = split_cal
        self.n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs
        self.config_file = config_file
        self.cnn_model = cnn_model
        self.model_cache = {}
        self.batch_num = batch_num
        self.batch_num_tf = batch_num_tf
        self.write_library = write_library
        self.use_library = use_library
        self.reload_library = reload_library
        self.pygam_calibration = pygam_calibration
        self.deepcallc_mod = deepcallc_mod
        self.deeplc_retrain = deeplc_retrain
        self.predict_ccs = predict_ccs
        self.n_epochs = n_epochs

        # Apparently...
        self.model = self.path_model

        if f_extractor:
            warnings.DeprecationWarning("f_extractor argument is deprecated.")

        # TODO REMOVE!!!
        self.verbose = True

        # Calibration variables
        self.calibrate_dict = {}
        self.calibrate_min = float("inf")
        self.calibrate_max = 0

        if self.deepcallc_mod:
            self.write_library = False
            self.use_library = None
            self.reload_library = False

    def __str__(self):
        return """
  _____                  _      _____
 |  __ \                | |    / ____|
 | |  | | ___  ___ _ __ | |   | |
 | |  | |/ _ \/ _ \ '_ \| |   | |
 | |__| |  __/  __/ |_) | |___| |____
 |_____/ \___|\___| .__/|______\_____|
                  | |
                  |_|
              """

    @staticmethod
    def _get_model_paths(passed_model_path: str | None, single_model_mode: bool) -> list[str]:
        """Get the model paths based on the passed model path and the single model mode."""
        if passed_model_path:
            return [passed_model_path]

        if single_model_mode:
            return [DEFAULT_MODELS[0]]

        return DEFAULT_MODELS

    def _extract_features(
        self,
        peptidoforms: list[str | Peptidoform] | PSMList,
        chunk_size: int = 10000,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Extract features for all peptidoforms."""
        if isinstance(peptidoforms, PSMList):
            peptidoforms = [psm.peptidoform for psm in peptidoforms]

        logger.debug("Running feature extraction in single-threaded mode...")
        if self.n_jobs <= 1:
            encodings = [
                encode_peptidoform(pf, predict_ccs=self.predict_ccs) for pf in peptidoforms
            ]

        else:
            logger.debug("Preparing feature extraction with Dask")
            # Process peptidoforms in larger chunks to reduce task overhead.
            peptidoform_strings = [str(pep) for pep in peptidoforms]  # Faster pickling of strings

            def chunked_encode(chunk):
                return [encode_peptidoform(pf, predict_ccs=self.predict_ccs) for pf in chunk]

            tasks = [
                delayed(chunked_encode)(peptidoform_strings[i : i + chunk_size])
                for i in range(0, len(peptidoform_strings), chunk_size)
            ]

            logger.debug("Starting feature extraction with Dask")
            chunks_encodings = compute(*tasks, scheduler="processes", workers=self.n_jobs)

            # Flatten the list of lists.
            encodings = [enc for chunk in chunks_encodings for enc in chunk]

        # Aggregate the encodings into a single dictionary.
        aggregated_encodings = aggregate_encodings(encodings)

        logger.debug("Finished feature extraction")

        return aggregated_encodings

    def _apply_calibration_core(
        self,
        uncal_preds: np.ndarray,
        cal_dict: dict | list[BaseEstimator],
        cal_min: float,
        cal_max: float,
    ) -> np.ndarray:
        """Apply calibration to the predictions."""
        if len(uncal_preds) == 0:
            return np.array([])

        cal_preds = []
        if self.pygam_calibration:
            linear_model_left, spline_model, linear_model_right = cal_dict
            y_pred_spline = spline_model.predict(uncal_preds.reshape(-1, 1))
            y_pred_left = linear_model_left.predict(uncal_preds.reshape(-1, 1))
            y_pred_right = linear_model_right.predict(uncal_preds.reshape(-1, 1))

            # Use spline model within the range of X
            within_range = (uncal_preds >= cal_min) & (uncal_preds <= cal_max)
            within_range = within_range.ravel()  # Ensure this is a 1D array for proper indexing

            # Create a prediction array initialized with spline predictions
            cal_preds = np.copy(y_pred_spline)

            # Replace predictions outside the range with the linear model predictions
            cal_preds[~within_range & (uncal_preds.ravel() < cal_min)] = y_pred_left[
                ~within_range & (uncal_preds.ravel() < cal_min)
            ]
            cal_preds[~within_range & (uncal_preds.ravel() > cal_max)] = y_pred_right[
                ~within_range & (uncal_preds.ravel() > cal_max)
            ]
        else:
            for uncal_pred in uncal_preds:
                try:
                    slope, intercept = cal_dict[str(round(uncal_pred, self.bin_distance))]
                    cal_preds.append(slope * (uncal_pred) + intercept)
                except KeyError:
                    # outside of the prediction range ... use the last
                    # calibration curve
                    if uncal_pred <= cal_min:
                        slope, intercept = cal_dict[str(round(cal_min, self.bin_distance))]
                        cal_preds.append(slope * (uncal_pred) + intercept)
                    elif uncal_pred >= cal_max:
                        slope, intercept = cal_dict[str(round(cal_max, self.bin_distance))]
                        cal_preds.append(slope * (uncal_pred) + intercept)
                    else:
                        slope, intercept = cal_dict[str(round(cal_max, self.bin_distance))]
                        cal_preds.append(slope * (uncal_pred) + intercept)

        return np.array(cal_preds)

    def _make_preds_core_library(self, psm_list=None, calibrate=True, mod_name=None):
        """Get predictions stored in library and calibrate them if needed."""
        psm_list = [] if psm_list is None else psm_list
        ret_preds = []
        for psm in psm_list:
            ret_preds.append(LIBRARY[psm.peptidoform.proforma + "|" + mod_name])

        if calibrate:
            if isinstance(self.calibrate_min, dict):
                # if multiple models are used, use the model name to get the
                # calibration values (DeepCallC mode)
                calibrate_dict, calibrate_min, calibrate_max = (
                    self.calibrate_dict[mod_name],
                    self.calibrate_min[mod_name],
                    self.calibrate_max[mod_name],
                )
            else:
                # if only one model is used, use the same calibration values
                calibrate_dict, calibrate_min, calibrate_max = (
                    self.calibrate_dict,
                    self.calibrate_min,
                    self.calibrate_max,
                )

            ret_preds = self._apply_calibration_core(
                ret_preds, calibrate_dict, calibrate_min, calibrate_max
            )

        return ret_preds

    def _make_preds_core(
        self,
        X: np.ndarray | None = None,
        X_sum: np.ndarray | None = None,
        X_global: np.ndarray | None = None,
        X_hc: np.ndarray | None = None,
        calibrate=True,
        mod_name=None,
    ) -> np.ndarray:
        """Make predictions."""
        # Check calibration state
        if calibrate:
            assert self.calibrate_dict, (
                "DeepLC instance is not yet calibrated. Calibrate before making predictions, or "
                "use `calibrate=False`"
            )

        if len(X) == 0:
            return np.array([])

        dataset = DeepLCDataset(X, X_sum, X_global, X_hc)
        loader = DataLoader(dataset, batch_size=self.batch_num_tf, shuffle=False)

        ret_preds = []

        mod = torch.load(mod_name, weights_only=False, map_location=torch.device("cpu"))
        mod.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    batch_X, batch_X_sum, batch_X_global, batch_X_hc = batch
                    batch_preds = mod(batch_X, batch_X_sum, batch_X_global, batch_X_hc)
                    ret_preds.append(batch_preds.detach().cpu().numpy())

            ret_preds = np.concatenate(ret_preds, axis=0)
        except UnboundLocalError:
            logger.debug("X is empty, skipping...")
            ret_preds = []

        if calibrate:
            if isinstance(self.calibrate_min, dict):
                # if multiple models are used, use the model name to get the
                # calibration values (DeepCallC mode)
                calibrate_dict, calibrate_min, calibrate_max = (
                    self.calibrate_dict[mod_name],
                    self.calibrate_min[mod_name],
                    self.calibrate_max[mod_name],
                )
            else:
                # if only one model is used, use the same calibration values
                calibrate_dict, calibrate_min, calibrate_max = (
                    self.calibrate_dict,
                    self.calibrate_min,
                    self.calibrate_max,
                )

            ret_preds = self._apply_calibration_core(
                ret_preds, calibrate_dict, calibrate_min, calibrate_max
            )

        gc.collect()
        return ret_preds

    def make_preds(
        self,
        psm_list: PSMList | None = None,
        infile: str | Path | None = None,
        seq_df: pd.DataFrame | None = None,
        calibrate: bool = True,
        mod_name: str | None = None,
    ):
        """
        Make predictions for sequences, in batches if required.

        Parameters
        ----------
        psm_list
            PSMList object containing the peptidoforms to predict for.
        infile
            Path to a file containing the peptidoforms to predict for.
        seq_df
            DataFrame containing the sequences (column:seq), modifications
            (column:modifications) and naming (column:index).
        calibrate
            calibrate predictions or just return the predictions.
        mod_name
            specify a model to use instead of the model assigned originally to this instance of the
            object.

        Returns
        -------
        np.array
            predictions
        """
        if psm_list is None:
            if seq_df is not None:
                psm_list = _dataframe_to_psm_list(seq_df)
            elif infile is not None:
                psm_list = _file_to_psm_list(infile)
            else:
                raise ValueError("Either `psm_list` or `seq_df` or `infile` must be provided.")

        if len(psm_list) == 0:
            logger.warning("No PSMs to predict for.")
            return []

        ret_preds_batches = []
        for psm_list_t in divide_chunks(psm_list, self.batch_num):
            ret_preds = []

            # Extract features for the CNN model
            # features = self._extract_features(psm_list_t)
            # X_sum, X_global, X_hc, X = unpack_features(features)
            X, X_sum, X_global, X_hc = self._prepare_feature_matrices(psm_list_t)

            # Check if model was provided, and if not, whether multiple models are selected in
            # the DeepLC object or not.
            if mod_name:
                model_names = [mod_name]
            elif isinstance(self.model, dict):
                model_names = [m_name for m_group_name, m_name in self.model.items()]
            elif isinstance(self.model, list):
                model_names = self.model
            elif isinstance(self.model, str):
                model_names = [self.model]
            else:
                raise ValueError("Invalid model name provided.")

            # Get predictions
            if len(model_names) > 1:
                # Iterate over models if multiple were selected
                model_predictions = []
                for model_name in model_names:
                    model_predictions.append(
                        self._make_preds_core(
                            X=X,
                            X_sum=X_sum,
                            X_global=X_global,
                            X_hc=X_hc,
                            calibrate=calibrate,
                            mod_name=model_name,
                        )
                    )
                # Average the predictions from all models
                ret_preds = np.array([sum(a) / len(a) for a in zip(*ret_preds, strict=True)])
                # ret_preds = np.mean(model_predictions, axis=0)

            else:
                # Use the single model
                ret_preds = self._make_preds_core(
                    X=X,
                    X_sum=X_sum,
                    X_global=X_global,
                    X_hc=X_hc,
                    calibrate=calibrate,
                    mod_name=model_names[0],
                )

            ret_preds_batches.append(ret_preds)

        all_ret_preds = np.concatenate(ret_preds_batches, axis=0)

        return all_ret_preds

    def _calibrate_preds_pygam(
        self,
        measured_tr: np.ndarray,
        predicted_tr: np.ndarray,
    ) -> tuple[float, float, list[BaseEstimator]]:
        """Make calibration curve for predictions using PyGAM."""
        logger.debug("Getting predictions for calibration...")

        # sort two lists, predicted and observed based on measured tr
        tr_sort = [
            (mtr, ptr)
            for mtr, ptr in sorted(
                zip(measured_tr, predicted_tr, strict=True), key=lambda pair: pair[1]
            )
        ]
        measured_tr = np.array([mtr for mtr, ptr in tr_sort], dtype=np.float32)
        predicted_tr = np.array([ptr for mtr, ptr in tr_sort], dtype=np.float32)

        # Fit a SplineTransformer model
        if self.deeplc_retrain:
            spline = SplineTransformer(degree=2, n_knots=10)
            linear_model = LinearRegression()
            linear_model.fit(predicted_tr.reshape(-1, 1), measured_tr)

            linear_model_left = linear_model
            spline_model = linear_model
            linear_model_right = linear_model
        else:
            spline = SplineTransformer(degree=4, n_knots=int(len(measured_tr) / 500) + 5)
            spline_model = make_pipeline(spline, LinearRegression())
            spline_model.fit(predicted_tr.reshape(-1, 1), measured_tr)

            # Determine the top 10% of data on either end
            n_top = int(len(predicted_tr) * 0.1)

            # Fit a linear model on the bottom 10% (left-side extrapolation)
            X_left = predicted_tr[:n_top]
            y_left = measured_tr[:n_top]
            linear_model_left = LinearRegression()
            linear_model_left.fit(X_left.reshape(-1, 1), y_left)

            # Fit a linear model on the top 10% (right-side extrapolation)
            X_right = predicted_tr[-n_top:]
            y_right = measured_tr[-n_top:]
            linear_model_right = LinearRegression()
            linear_model_right.fit(X_right.reshape(-1, 1), y_right)

        calibrate_min = min(predicted_tr)
        calibrate_max = max(predicted_tr)

        return (
            calibrate_min,
            calibrate_max,
            [linear_model_left, spline_model, linear_model_right],
        )

    def _calibrate_preds_piecewise_linear(
        self,
        measured_tr: np.ndarray,
        predicted_tr: np.ndarray,
        use_median: bool = True,
    ) -> tuple[float, float, dict[str, tuple[float]]]:
        """Make calibration curve for predictions."""
        # sort two lists, predicted and observed based on measured tr
        tr_sort = [
            (mtr, ptr)
            for mtr, ptr in sorted(
                zip(measured_tr, predicted_tr, strict=False), key=lambda pair: pair[1]
            )
        ]
        measured_tr = np.array([mtr for mtr, ptr in tr_sort])
        predicted_tr = np.array([ptr for mtr, ptr in tr_sort])

        mtr_mean = []
        ptr_mean = []

        calibrate_dict = {}
        calibrate_min = float("inf")
        calibrate_max = 0

        logger.debug(
            "Selecting the data points for calibration (used to fit the linear models between)"
        )
        # smooth between observed and predicted
        split_val = predicted_tr[-1] / self.split_cal

        for range_calib_number in np.arange(0.0, predicted_tr[-1], split_val):
            ptr_index_start = np.argmax(predicted_tr >= range_calib_number)
            ptr_index_end = np.argmax(predicted_tr >= range_calib_number + split_val)

            # no points so no cigar... use previous points
            if ptr_index_start >= ptr_index_end:
                logger.debug(
                    "Skipping calibration step, due to no points in the "
                    "predicted range (are you sure about the split size?): "
                    "%s,%s",
                    range_calib_number,
                    range_calib_number + split_val,
                )
                continue

            mtr = measured_tr[ptr_index_start:ptr_index_end]
            ptr = predicted_tr[ptr_index_start:ptr_index_end]

            if use_median:
                mtr_mean.append(np.median(mtr))
                ptr_mean.append(np.median(ptr))
            else:
                mtr_mean.append(sum(mtr) / len(mtr))
                ptr_mean.append(sum(ptr) / len(ptr))

        logger.debug("Fitting the linear models between the points")

        if self.split_cal >= len(measured_tr):
            raise CalibrationError(
                f"Not enough measured tr ({len(measured_tr)}) for the chosen number of splits "
                f"({self.split_cal}). Choose a smaller split_cal parameter or provide more "
                "peptides for fitting the calibration curve."
            )
        if len(mtr_mean) == 0:
            raise CalibrationError("The measured tr list is empty, not able to calibrate")
        if len(ptr_mean) == 0:
            raise CalibrationError("The predicted tr list is empty, not able to calibrate")

        # calculate calibration curves
        for i in range(0, len(ptr_mean)):
            if i >= len(ptr_mean) - 1:
                continue
            delta_ptr = ptr_mean[i + 1] - ptr_mean[i]
            delta_mtr = mtr_mean[i + 1] - mtr_mean[i]

            slope = delta_mtr / delta_ptr
            intercept = (-1 * (ptr_mean[i] * slope)) + mtr_mean[i]

            # optimized predictions using a dict to find calibration curve very
            # fast
            for v in np.arange(
                round(ptr_mean[i], self.bin_distance),
                round(ptr_mean[i + 1], self.bin_distance),
                1 / ((self.bin_distance) * self.dict_cal_divider),
            ):
                if v < calibrate_min:
                    calibrate_min = v
                if v > calibrate_max:
                    calibrate_max = v
                calibrate_dict[str(round(v, self.bin_distance))] = (slope, intercept)

        return calibrate_min, calibrate_max, calibrate_dict

    def calibrate_preds(
        self,
        psm_list: PSMList | None = None,
        infile: str | Path | None = None,
        seq_df: pd.DataFrame | None = None,
        measured_tr: np.ndarray | None = None,
        location_retraining_models: str = "",
        sample_for_calibration_curve: int | None = None,
        use_median: bool = True,
        return_plotly_report=False,
    ) -> dict | None:
        """
        Find best model and calibrate.

        Parameters
        ----------
        psm_list
            PSMList object containing the peptidoforms to predict for.
        infile
            Path to a file containing the peptidoforms to predict for.
        seq_df
            DataFrame containing the sequences (column:seq), modifications (column:modifications),
            naming (column:index), and optionally charge (column:charge) and measured retention
            time (column:tr).
        measured_tr : list
            Measured retention time used for calibration. Should correspond to the PSMs in the
            provided PSMs. If None, the measured retention time is taken from the PSMs.
        correction_factor : float
            correction factor that needs to be applied to the supplied measured tr's
        location_retraining_models
            Location to save the retraining models; if None, a temporary directory is used.
        sample_for_calibration_curve
            Number of PSMs to sample for calibration curve; if None, all provided PSMs are used.
        use_median
            Whether to use the median value in a window to perform calibration; only applies to
            piecewise linear calibration, not to PyGAM calibration.
        return_plotly_report
            Whether to return a plotly report with the calibration results.

        Returns
        -------
        dict | None
            Dictionary with plotly report information or None.

        """
        # Getting PSMs either from psm_list, seq_df, or infile
        if psm_list is None:
            if seq_df is not None:
                psm_list = _dataframe_to_psm_list(seq_df)
            elif infile is not None:
                psm_list = _file_to_psm_list(infile)
            else:
                raise ValueError("Either `psm_list` or `seq_df` or `infile` must be provided.")

        # Getting measured retention time either from measured_tr or provided PSMs
        if not measured_tr:
            measured_tr = [psm.retention_time for psm in psm_list]
            if None in measured_tr:
                raise ValueError("Not all PSMs have an observed retention time.")

        # Ensuring self.model is list of strings
        if isinstance(self.model, str):
            self.model = [self.model]

        logger.debug("Start to calibrate predictions ...")
        logger.debug(f"Ready to find the best model out of: {self.model}")

        best_perf = float("inf")
        best_calibrate_min = 0.0
        best_calibrate_max = 0.0
        best_calibrate_dict = {}
        mod_calibrate_dict = {}
        mod_calibrate_min_dict = {}
        mod_calibrate_max_dict = {}
        pred_dict = {}
        mod_dict = {}
        temp_obs = []
        temp_pred = []

        if self.deeplc_retrain:
            logger.debug("Preparing for model fine-tuning...")

            X, X_sum, X_global, X_hc = self._prepare_feature_matrices(psm_list)
            dataset = DeepLCDataset(X, X_sum, X_global, X_hc, np.array(measured_tr))

            base_model_path = self.model[0] if isinstance(self.model, list) else self.model
            base_model = torch.load(
                base_model_path, weights_only=False, map_location=torch.device("cpu")
            )
            base_model.eval()

            fine_tuner = DeepLCFineTuner(
                model=base_model,
                train_data=dataset,
                batch_size=self.batch_num_tf,
                epochs=self.n_epochs,
            )
            # fine_tuner._freeze_layers()
            fine_tuned_model = fine_tuner.fine_tune()

            if location_retraining_models:
                os.makedirs(location_retraining_models, exist_ok=True)
                temp_dir_obj = TemporaryDirectory()
                t_dir_models = temp_dir_obj.name
                self._temp_dir_obj = temp_dir_obj
            else:
                t_dir_models = location_retraining_models
                os.makedirs(t_dir_models, exist_ok=True)

            # Define path to save fine-tuned model
            fine_tuned_model_path = os.path.join(t_dir_models, "fine_tuned_model.pth")
            torch.save(fine_tuned_model, fine_tuned_model_path)
            self.model = [fine_tuned_model_path]

        # Limit calibration to a subset of PSMs if specified
        if sample_for_calibration_curve:
            psm_list = random.sample(list(psm_list), sample_for_calibration_curve)
            measured_tr = [psm.retention_time for psm in psm_list]

        for model_name in self.model:
            logger.debug(f"Trying out the following model: {model_name}")
            predicted_tr = self.make_preds(psm_list, calibrate=False, mod_name=model_name)

            if self.pygam_calibration:
                calibrate_output = self._calibrate_preds_pygam(measured_tr, predicted_tr)
            else:
                calibrate_output = self._calibrate_preds_piecewise_linear(
                    measured_tr, predicted_tr, use_median=use_median
                )
            self.calibrate_min, self.calibrate_max, self.calibrate_dict = calibrate_output
            # TODO: Currently, calibration dict can be both a dict (linear) or a list of models
            # (PyGAM)... This should be handled better in the future.

            # Skip this model if calibrate_dict is empty
            # TODO: Should this do something when using PyGAM and calibrate_dict is a list?
            if isinstance(self.calibrate_dict, dict) and len(self.calibrate_dict.keys()) == 0:
                continue

            m_name = model_name.split("/")[-1]

            # Get new predictions with calibration
            preds = self.make_preds(psm_list, calibrate=True, seq_df=seq_df, mod_name=model_name)

            m_group_name = "deepcallc" if self.deepcallc_mod else "_".join(m_name.split("_")[:-1])
            m = model_name
            try:
                pred_dict[m_group_name][m] = preds
                mod_dict[m_group_name][m] = m
                mod_calibrate_dict[m_group_name][m] = self.calibrate_dict
                mod_calibrate_min_dict[m_group_name][m] = self.calibrate_min
                mod_calibrate_max_dict[m_group_name][m] = self.calibrate_max
            except KeyError:
                pred_dict[m_group_name] = {}
                mod_dict[m_group_name] = {}
                mod_calibrate_dict[m_group_name] = {}
                mod_calibrate_min_dict[m_group_name] = {}
                mod_calibrate_max_dict[m_group_name] = {}

                pred_dict[m_group_name][m] = preds
                mod_dict[m_group_name][m] = m
                mod_calibrate_dict[m_group_name][m] = self.calibrate_dict
                mod_calibrate_min_dict[m_group_name][m] = self.calibrate_min
                mod_calibrate_max_dict[m_group_name][m] = self.calibrate_max

        for m_name in pred_dict:
            preds = [sum(a) / len(a) for a in zip(*list(pred_dict[m_name].values()), strict=True)]
            if len(measured_tr) == 0:
                perf = sum(abs(seq_df["tr"] - preds))
            else:
                perf = sum(abs(np.array(measured_tr) - np.array(preds)))

            logger.debug(f"For {m_name} model got a performance of: {perf / len(preds)}")

            if perf < best_perf:
                m_group_name = "deepcallc" if self.deepcallc_mod else m_name

                # TODO is deepcopy really required?
                best_calibrate_dict = copy.deepcopy(mod_calibrate_dict[m_group_name])
                best_calibrate_min = copy.deepcopy(mod_calibrate_min_dict[m_group_name])
                best_calibrate_max = copy.deepcopy(mod_calibrate_max_dict[m_group_name])

                best_model = copy.deepcopy(mod_dict[m_group_name])
                best_perf = perf

                temp_obs = np.array(measured_tr)
                temp_pred = np.array(preds)

        self.calibrate_dict = best_calibrate_dict
        self.calibrate_min = best_calibrate_min
        self.calibrate_max = best_calibrate_max
        self.model = best_model

        if self.deepcallc_mod:
            self.deepcallc_model = train_elastic_net(
                pd.DataFrame(pred_dict["deepcallc"]), seq_df["tr"]
            )

        logger.debug(f"Model with the best performance got selected: {best_model}")

        if return_plotly_report:
            import deeplc.plot

            plotly_return_dict = {}
            plotly_df = pd.DataFrame(
                list(zip(temp_obs, temp_pred, strict=True)),
                columns=[
                    "Observed retention time",
                    "Predicted retention time",
                ],
            )
            plotly_return_dict["scatter"] = deeplc.plot.scatter(plotly_df)
            plotly_return_dict["baseline_dist"] = deeplc.plot.distribution_baseline(plotly_df)
            return plotly_return_dict

        return None


def _get_pool(n_jobs: int) -> multiprocessing.Pool | multiprocessing.dummy.Pool:  # type: ignore
    """Get a Pool object for parallel processing."""
    if multiprocessing.current_process().daemon:
        logger.warning(
            "DeepLC is running in a daemon process. Disabling multiprocessing as daemonic "
            "processes can't have children."
        )
        return multiprocessing.dummy.Pool(1)
    elif n_jobs == 1:
        return multiprocessing.dummy.Pool(1)
    else:
        max_n_jobs = multiprocessing.cpu_count()
        if n_jobs > max_n_jobs:
            logger.warning(
                f"Number of jobs ({n_jobs}) exceeds the number of CPUs ({max_n_jobs}). "
                f"Setting number of jobs to {max_n_jobs}."
            )
            return multiprocessing.Pool(max_n_jobs)
        else:
            return multiprocessing.Pool(n_jobs)


def _lists_to_psm_list(
    sequences: list[str],
    modifications: list[str | None],
    identifiers: list[str],
    charges: list[int] | None,
    retention_times: list[float] | None = None,
    n_jobs: int = 1,
) -> PSMList:
    """Convert lists into a PSMList using Dask for parallel processing."""
    if not charges:
        charges = [None] * len(sequences)

    if not retention_times:
        retention_times = [None] * len(sequences)

    def create_psm(args):
        sequence, modifications, identifier, charge, retention_time = args
        return PSM(
            peptidoform=peprec_to_proforma(sequence, modifications, charge=charge),
            spectrum_id=identifier,
            retention_time=retention_time,
        )

    args_list = list(
        zip(sequences, modifications, identifiers, charges, retention_times, strict=True)
    )
    tasks = [delayed(create_psm)(args) for args in args_list]
    list_of_psms = list(compute(*tasks, scheduler="processes"))
    return PSMList(psm_list=list_of_psms)


# TODO: I'm not sure what the expected behavior was for charges; they were parsed
# from the dataframe, while the passed list was used to check whether it they should get
# parsed. I'll allow both with a priority for the passed charges.
def _dataframe_to_psm_list(
    dataframe: pd.DataFrame,
    charges: list[int] | None,
    n_jobs: int = 1,
) -> PSMList:
    """Convert a DataFrame with sequences, modifications, and identifiers into a PSMList."""
    sequences = dataframe["seq"]
    modifications = dataframe["modifications"]
    identifiers = dataframe.index
    retention_times = dataframe["tr"] if "tr" in dataframe.columns else None

    if not charges and "charge" in dataframe.columns:
        charges = dataframe["charge"]

    return _lists_to_psm_list(
        sequences, modifications, identifiers, charges, retention_times, n_jobs=n_jobs
    )


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
