"Unit and integration tests for DeepLC."

import logging
import subprocess

import numpy as np
import pandas as pd
import pytest

import deeplc


def _r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
    denominator = ((y_true - np.average(y_true)) ** 2).sum(dtype=np.float64)
    if denominator == 0.0:
        return 1.0
    return 1 - numerator / denominator


def test_cli_basic():
    """Test command line interface help message."""
    assert (
        subprocess.getstatusoutput("deeplc -h")[0] == 0
    ), "`deeplc -h` returned non-zero status code"


def test_cli_full():
    """ " Test command line interface with input files."""
    file_path_pred = "examples/datasets/test_train.csv"
    file_path_cal = "examples/datasets/test_train.csv"
    file_path_out = "pytest_cli_out.csv"

    command = [
        "deeplc",
        "--file_pred",
        file_path_pred,
        "--file_cal",
        file_path_cal,
        "--file_pred_out",
        file_path_out,
    ]
    subprocess.run(command, check=True)

    preds_df = pd.read_csv(file_path_out)
    train_df = pd.read_csv(file_path_pred)
    model_r2 = _r2_score(train_df["tr"], preds_df["predicted retention time"])
    logging.info("DeepLC R2 score on %s: %f", file_path_pred, model_r2)
    assert model_r2 > 0.90, f"DeepLC R2 score on {file_path_pred} below 0.9 \
(was {model_r2})"


if __name__ == "__main__":
    pytest.main()
