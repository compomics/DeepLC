"""
Code used to run the retention time predictor
"""
__license__ = "Apache License, Version 2.0"

# Standard library
from collections import Counter
import argparse
import itertools
import logging
import multiprocessing
import os
import pickle
import pkg_resources
import random
import re
import sys

# Third party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# DeepLC
from deeplc import DeepLC
from deeplc import FeatExtractor

__version__ = pkg_resources.require("deeplc")[0].version


def parse_arguments():
    """Read arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_pred",
        type=str,
        dest="file_pred",
        default="",
        help="Path to peptide file for which to make predictions (required)")

    parser.add_argument(
        "--file_cal",
        type=str,
        dest="file_cal",
        default="",
        help="Path to peptide file with retention times to use for calibration\
            (optional)")

    parser.add_argument(
        "--file_pred_out",
        type=str,
        dest="file_pred_out",
        default="",
        help="Path to output file with predictions (optional)")

    parser.add_argument(
        "--file_model",
        help="Path to prediction model(s). Seperate with spaces. Leave empty \
            to select the best of the default models (optional)",
        nargs="+",
        default=[
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "mods/full_hc_LUNA_HILIC_fixed_mods.hdf5"
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "mods/full_hc_LUNA_SILICA_fixed_mods.hdf5"
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "mods/full_hc_dia_fixed_mods.hdf5"
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "mods/full_hc_PXD000954_fixed_mods.hdf5"
            ),
        ])

    parser.add_argument(
        "--split_cal",
        type=int,
        dest="split_cal",
        default=50,
        # TODO add help
        )

    parser.add_argument(
        "--dict_divider",
        type=int,
        dest="dict_divider",
        default=50,
        # TODO add help
        )

    parser.add_argument(
        "--batch_num",
        type=int,
        dest="batch_num",
        default=250000,
        help="Batch size (in peptides) for predicting the retention time. Set\
            lower to decrease memory footprint (optional, default=250000)")

    parser.add_argument(
        "--n_threads",
        type=int,
        dest="n_threads",
        default=16,
        help="Number of threads to use (optional, default=maximum available)")

    parser.add_argument(
        "--verbose",
        action='store_true',
        dest="verbose",
        default=False,
        help="Verbose logging"
    )

    parser.add_argument("--version", action="version", version=__version__)

    results = parser.parse_args()

    if not results.file_pred:
        parser.print_help()
        exit(0)

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results


def main():
    """Main function for the CLI."""
    argu = parse_arguments()

    logging.basicConfig(
		stream=sys.stdout,
		format='%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG if argu.verbose else logging.ERROR
	)

    if not argu.verbose:
        os.environ['KMP_WARNINGS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    max_threads = multiprocessing.cpu_count()
    if not argu.n_threads:
        argu.n_threads = max_threads
    elif argu.n_threads > max_threads:
        argu.n_threads = max_threads

    run(file_pred=argu.file_pred,
        file_cal=argu.file_cal,
        file_pred_out=argu.file_pred_out,
        file_model=argu.file_model,
        n_threads=argu.n_threads,
        verbose=argu.verbose,
        split_cal=argu.split_cal,
        dict_divider=argu.dict_divider,
        batch_num=argu.batch_num)


def run(file_pred="",
        file_cal="",
        file_pred_out="",
        file_model="",
        n_threads=None,
        verbose=False,
        split_cal=50,
        dict_divider=50,
        batch_num=50000):
    """
    Main function to run the DeepLC code

    Parameters
    ----------
    file_pred : str
        the file in peprec format that we need to make predictions for
        this file is not required to contain a tr column
    file_cal : str
        the file in peprec format that we use for calibrating the prediction
        model. This file is required to contain a tr column
    file_pred_out : str
        outfile for predictions, the file is in peprec format and predictions
        are added in the column !!!
    file_model : str | list
        the model(s) to try for retention time prediction can be a single
        location or several locations for multiple models to try
    n_threads : int
        number of threads to run mainly the feature extraction on
    split_cal : int
        number of splits or divisions to use for the calibration
    dict_divider : int
        !!!

    Returns
    -------
    None
    """

    # Read input files
    df_pred = pd.read_csv(file_pred)
    df_pred = df_pred.fillna("")

    if len(file_cal) > 1:
        df_cal = pd.read_csv(file_cal)
        df_cal = df_cal.fillna("")

    # Make a feature extraction object; you can skip this if you do not want to
    # use the default settings for DeepLC. Here we want to use a model that does
    # not use RDKit features so we skip the chemical descriptor making
    # procedure.
    f_extractor = FeatExtractor(add_sum_feat=False,
                                ptm_add_feat=False,
                                ptm_subtract_feat=False,
                                standard_feat=False,
                                chem_descr_feat=False,
                                add_comp_feat=False,
                                cnn_feats=True,
                                verbose=verbose)

    # Make the DeepLC object that will handle making predictions and
    # calibration
    dlc = DeepLC(path_model=file_model,
                 f_extractor=f_extractor,
                 cnn_model=True,
                 n_jobs=n_threads,
                 verbose=verbose,
                 batch_num=batch_num)

    # Calibrate the original model based on the new retention times
    if len(file_cal) > 1:
        dlc.calibrate_preds(seq_df=df_cal)

    # Make predictions; calibrated or uncalibrated
    if len(file_cal) > 1:
        preds = dlc.make_preds(seq_df=df_pred)
    else:
        preds = dlc.make_preds(seq_df=df_pred, calibrate=False)

    df_pred["predicted_tr"] = preds
    df_pred.to_csv(file_pred_out)

    if len(file_cal) > 1 and "tr" in df_pred.columns:
        plt.figure(figsize=(11.5, 9))
        plt.scatter(df_pred["tr"], df_pred["predicted_tr"], s=3)
        plt.title("DeepLC predictions")
        plt.xlabel("Observed retention times")
        plt.ylabel("Predicted retention times")
        plt.savefig(os.path.splitext(file_pred_out)[0] + '.png', dpi=300)


if __name__ == "__main__":
    main()
