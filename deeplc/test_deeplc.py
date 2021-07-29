"Unit and integration tests for DeepLC."

# Standard library
import logging
import pytest
import subprocess

# Third party
import pandas as pd
from sklearn.metrics import r2_score

# DeepLC
import deeplc


def test_cli_basic():
    """ Test command line interface help message. """
    assert subprocess.getstatusoutput('deeplc -h')[0] == 0, "`deeplc -h` \
returned non-zero status code"


def test_cli_full():
    """" Test command line interface with input files."""
    file_path_pred = "examples/datasets/test_train.csv"
    file_path_cal = "examples/datasets/test_train.csv"
    file_path_out = "pytest_cli_out.csv"

    command = [
        "deeplc", "--file_pred", file_path_pred, "--file_cal", file_path_cal,
        "--file_pred_out", file_path_out
    ]
    subprocess.run(command, check=True)

    preds_df = pd.read_csv(file_path_out)
    model_r2 = r2_score(preds_df['tr'], preds_df['predicted_tr'])
    logging.info("DeepLC R2 score on %s: %f", file_path_pred, model_r2)
    assert model_r2 > 0.95, f"DeepLC R2 score on {file_path_pred} below 0.95 \
(was {model_r2})"


if __name__ == "__main__":
    pytest.main()
