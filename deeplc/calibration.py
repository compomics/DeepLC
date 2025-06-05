"""Retention time calibration."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from deeplc._exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


class Calibration(ABC):
    """Abstract base class for retention time calibration."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(measured_tr: np.ndarray, predicted_tr: np.ndarray) -> None: ...

    @abstractmethod
    def transform(tr: np.ndarray) -> np.ndarray: ...
    

class IdentityCalibration(Calibration):
    """No calibration, just returns the predicted retention times."""

    def fit(self, measured_tr: np.ndarray, predicted_tr: np.ndarray) -> None:
        """No fitting required for NoCalibration."""
        pass

    def transform(self, tr: np.ndarray) -> np.ndarray:
        """
        Transform the predicted retention times without any calibration.

        Parameters
        ----------
        tr
            Retention times to be transformed.

        Returns
        -------
        np.ndarray
            Transformed retention times (same as input).
        """
        return tr


class PiecewiseLinearCalibration(Calibration):
    def __init__(
        self,
        split_cal: int = 50,
        bin_distance: float = 2.0,
        dict_cal_divider: int = 50,
        use_median: bool = True,
    ):
        """
        Piece-wise linear calibration for retention time.

        Parameters
        ----------
        split_cal
            Number of splits.
        bin_distance
            Distance between bins.
        dict_cal_divider
            # TODO: Make more descriptive
            Divider for the dictionary used in the piece-wise linear model.
        use_median
            # TODO: Make more descriptive
            If True, use median instead of mean for calibration.

        """
        super().__init__()
        self.split_cal = split_cal
        self.bin_distance = bin_distance
        self.dict_cal_divider = dict_cal_divider
        self.use_median = use_median

        self._calibrate_min = None
        self._calibrate_max = None
        self._calibrate_dict = None
        self._fit = False

    def fit(self, measured_tr: np.ndarray, predicted_tr: np.ndarray) -> None:
        """
        Fit a piece-wise linear model to the measured and predicted retention times.

        Parameters
        ----------
        measured_tr
            Measured retention times.
        predicted_tr
            Predicted retention times.

        """
        measured_tr, predicted_tr = _process_arrays(measured_tr, predicted_tr)

        mtr_mean = []
        ptr_mean = []

        calibrate_dict = {}
        calibrate_min = float("inf")
        calibrate_max = 0

        LOGGER.debug(
            "Selecting the data points for calibration (used to fit the linear models between)"
        )
        # smooth between observed and predicted
        split_val = predicted_tr[-1] / self.split_cal

        for range_calib_number in np.arange(0.0, predicted_tr[-1], split_val):
            ptr_index_start = np.argmax(predicted_tr >= range_calib_number)
            ptr_index_end = np.argmax(predicted_tr >= range_calib_number + split_val)

            # no points so no cigar... use previous points
            if ptr_index_start >= ptr_index_end:
                LOGGER.debug(
                    "Skipping calibration step, due to no points in the "
                    "predicted range (are you sure about the split size?): "
                    "%s,%s",
                    range_calib_number,
                    range_calib_number + split_val,
                )
                continue

            mtr = measured_tr[ptr_index_start:ptr_index_end]
            ptr = predicted_tr[ptr_index_start:ptr_index_end]

            if self.use_median:
                mtr_mean.append(np.median(mtr))
                ptr_mean.append(np.median(ptr))
            else:
                mtr_mean.append(sum(mtr) / len(mtr))
                ptr_mean.append(sum(ptr) / len(ptr))

        LOGGER.debug("Fitting the linear models between the points")

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

            # optimized predictions using a dict to find calibration curve very fast
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

        self._calibrate_min = calibrate_min
        self._calibrate_max = calibrate_max
        self._calibrate_dict = calibrate_dict

        self._fit = True

    def transform(self, tr: np.ndarray) -> np.ndarray:
        """
        Transform the predicted retention times using the fitted piece-wise linear model.

        Parameters
        ----------
        tr
            Retention times to be transformed.

        Returns
        -------
        np.ndarray
            Transformed retention times.
        """
        if not self._fit:
            raise CalibrationError(
                "The model has not been fitted yet. Please call fit() before transform()."
            )

        if tr.shape[0] == 0:
            return np.array([])

        # TODO: Can this be vectorized?
        cal_preds = []
        for uncal_pred in tr:
            try:
                slope, intercept = self.cal_dict[str(round(uncal_pred, self.bin_distance))]
                cal_preds.append(slope * (uncal_pred) + intercept)
            except KeyError:
                # outside of the prediction range ... use the last
                # calibration curve
                if uncal_pred <= self.cal_min:
                    slope, intercept = self.cal_dict[str(round(self.cal_min, self.bin_distance))]
                    cal_preds.append(slope * (uncal_pred) + intercept)
                elif uncal_pred >= self.cal_max:
                    slope, intercept = self.cal_dict[str(round(self.cal_max, self.bin_distance))]
                    cal_preds.append(slope * (uncal_pred) + intercept)
                else:
                    slope, intercept = self.cal_dict[str(round(self.cal_max, self.bin_distance))]
                    cal_preds.append(slope * (uncal_pred) + intercept)

        return np.array(cal_preds)


class SplineTransformerCalibration(Calibration):
    def __init__(self):
        """SplineTransformer calibration for retention time."""
        super().__init__()
        self._calibrate_min = None
        self._calibrate_max = None
        self._linear_model_left = None
        self._spline_model = None
        self._linear_model_right = None

        self._fit = False

    def fit(
        self,
        measured_tr: np.ndarray,
        predicted_tr: np.ndarray,
        simplified: bool = False,  # TODO: Move to __init__?
    ) -> None:
        """
        Fit the SplineTransformer model to the measured and predicted retention times.

        Parameters
        ----------
        measured_tr
            Measured retention times.
        predicted_tr
            Predicted retention times.
        simplified
            If True, use a simplified model with fewer knots and a linear model.
            If False, use a more complex model with more knots and a spline model.

        """
        measured_tr, predicted_tr = _process_arrays(measured_tr, predicted_tr)

        # Fit a SplineTransformer model
        if simplified:
            spline = SplineTransformer(degree=2, n_knots=10)
            linear_model = LinearRegression()
            linear_model.fit(predicted_tr.reshape(-1, 1), measured_tr)

            linear_model_left = linear_model
            # TODO @RobbinBouwmeester: Should this be the spline model?
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

        self._calibrate_min = min(predicted_tr)
        self._calibrate_max = max(predicted_tr)
        self._linear_model_left = linear_model_left
        self._spline_model = spline_model
        self._linear_model_right = linear_model_right

        self._fit = True

    def transform(self, tr: np.ndarray) -> np.ndarray:
        """
        Transform the predicted retention times using the fitted SplineTransformer model.

        Parameters
        ----------
        tr
            Retention times to be transformed.

        Returns
        -------
        np.ndarray
            Transformed retention times.
        """
        if not self._fit:
            raise CalibrationError(
                "The model has not been fitted yet. Please call fit() before transform()."
            )

        if tr.shape[0] == 0:
            return np.array([])

        y_pred_spline = self._spline_model.predict(tr.reshape(-1, 1))
        y_pred_left = self._linear_model_left.predict(tr.reshape(-1, 1))
        y_pred_right = self._linear_model_right.predict(tr.reshape(-1, 1))

        # Use spline model within the range of X
        within_range = (tr >= self.cal_min) & (tr <= self.cal_max)
        within_range = within_range.ravel()  # Ensure this is a 1D array for proper indexing

        # Create a prediction array initialized with spline predictions
        cal_preds = np.copy(y_pred_spline)

        # Replace predictions outside the range with the linear model predictions
        cal_preds[~within_range & (tr.ravel() < self.cal_min)] = y_pred_left[
            ~within_range & (tr.ravel() < self.cal_min)
        ]
        cal_preds[~within_range & (tr.ravel() > self.cal_max)] = y_pred_right[
            ~within_range & (tr.ravel() > self.cal_max)
        ]

        return np.array(cal_preds)


def _process_arrays(
    measured_tr: np.ndarray,
    predicted_tr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Process the measured and predicted retention times."""
    # Check array lengths
    if len(measured_tr) != len(predicted_tr):
        raise ValueError(
            f"Measured and predicted retention times must have the same length. "
            f"Got {len(measured_tr)} and {len(predicted_tr)}."
        )

    # Ensure both arrays are 1D
    if len(measured_tr.shape) > 1:
        measured_tr = measured_tr.flatten()
    if len(predicted_tr.shape) > 1:
        predicted_tr = predicted_tr.flatten()

    # Sort arrays by predicted_tr
    indices = np.argsort(predicted_tr)
    measured_tr = np.array(measured_tr, dtype=np.float32)[indices]
    predicted_tr = np.array(predicted_tr, dtype=np.float32)[indices]

    return measured_tr, predicted_tr
