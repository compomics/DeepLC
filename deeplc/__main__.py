"""
Code used to run the retention time predictor
"""

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__credits__ = ["Robbin Bouwmeester", "Ralf Gabriels", "Prof. Lennart Martens", "Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]

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
        default=None
    )

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
        "--plot_predictions",
        dest='plot_predictions',
		action='store_true',
        default=False,
        help='Save scatter plot of predictions vs observations (default=False)'
    )

    parser.add_argument(
        "--n_threads",
        type=int,
        dest="n_threads",
        default=16,
        help="Number of threads to use (optional, default=maximum available)")

    parser.add_argument(
        "--log_level",
        type=str,
        dest="log_level",
        default='info',
        help="Logging level (debug, info, warning, error, or critical; default=info)"
    )

    parser.add_argument("--version", action="version", version=__version__)

    results = parser.parse_args()

    if not results.file_pred:
        parser.print_help()
        exit(0)

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results


def setup_logging(passed_level):
	log_mapping = {
		'critical': logging.CRITICAL,
		'error': logging.ERROR,
		'warning': logging.WARNING,
		'info': logging.INFO,
		'debug': logging.DEBUG,
	}

	if passed_level.lower() not in log_mapping:
		print(
			"Invalid log level. Should be one of the following: ",
			', '.join(log_mapping.keys())
		)
		exit(1)

	logging.basicConfig(
		stream=sys.stdout,
		format='%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		level=log_mapping[passed_level.lower()]
	)


def main():
    """Main function for the CLI."""
    argu = parse_arguments()

    setup_logging(argu.log_level)
    if argu.log_level.lower() != "debug":
        logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
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
        verbose=True,
        split_cal=argu.split_cal,
        dict_divider=argu.dict_divider,
        batch_num=argu.batch_num,
        plot_predictions=argu.plot_predictions)


def run(file_pred="",
        file_cal="",
        file_pred_out="",
        file_model=None,
        n_threads=None,
        verbose=False,
        split_cal=50,
        dict_divider=50,
        batch_num=50000,
        plot_predictions=False):
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
        are added in the column TODO
    file_model : str | list | None 
        the model(s) to try for retention time prediction can be a single
        location or several locations for multiple models to try
    n_threads : int
        number of threads to run mainly the feature extraction on
    split_cal : int
        number of splits or divisions to use for the calibration
    dict_divider : int
        TODO
    batch_num : int
        TODO
    plot_predictions : bool
        Save scatter plot of predictions vs observations

    Returns
    -------
    None
    """

    logging.info("Using DeepLC version %s", __version__)

    if len(file_cal) == 0 and file_model != None:
        fm_dict = {}
        sel_group = ""
        for fm in file_model:
            if len(sel_group) == 0:
                sel_group = "_".join(fm.split("_")[:-1])
                fm_dict[sel_group]= fm
                continue
            m_group = "_".join(fm.split("_")[:-1])
            if m_group == sel_group:
                fm_dict[m_group] = fm
        file_model = fm_dict

    # Read input files
    df_pred = pd.read_csv(file_pred)
    if len(df_pred.columns) < 2:
        df_pred = pd.read_csv(file_pred,sep=" ")
    df_pred = df_pred.fillna("")

    if len(file_cal) > 1:
        df_cal = pd.read_csv(file_cal)
        if len(df_cal.columns) < 2:
            df_cal = pd.read_csv(df_cal,sep=" ")
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
        logging.info("Selecting best model and calibrating predictions...")
        dlc.calibrate_preds(seq_df=df_cal)

    # Make predictions; calibrated or uncalibrated
    logging.info("Making predictions using model: %s", dlc.model)
    if len(file_cal) > 1:
        preds = dlc.make_preds(seq_df=df_pred)
    else:
        preds = dlc.make_preds(seq_df=df_pred, calibrate=False)

    df_pred["predicted_tr"] = preds
    logging.debug("Writing predictions to file: %s", file_pred_out)
    df_pred.to_csv(file_pred_out)

    if plot_predictions:
        if len(file_cal) > 1 and "tr" in df_pred.columns:
            file_pred_figure = os.path.splitext(file_pred_out)[0] + '.png'
            logging.debug("Saving scatterplot of predictions to file: %s", file_pred_figure)
            plt.figure(figsize=(11.5, 9))
            plt.scatter(df_pred["tr"], df_pred["predicted_tr"], s=3)
            plt.title("DeepLC predictions")
            plt.xlabel("Observed retention times")
            plt.ylabel("Predicted retention times")
            plt.savefig(file_pred_figure, dpi=300)
        else:
            logging.warning('No observed retention time in input data. Cannot \
plot predictions')

    logging.info("DeepLC finished!")


if __name__ == "__main__":
    main()
