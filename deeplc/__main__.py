"""Main command line interface to DeepLC."""

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__credits__ = ["Robbin Bouwmeester", "Ralf Gabriels", "Prof. Lennart Martens", "Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]

import logging
import os
import sys
import warnings

import pandas as pd
from matplotlib import pyplot as plt

from deeplc import __version__, DeepLC, FeatExtractor
from deeplc._argument_parser import parse_arguments
from deeplc._exceptions import DeepLCError


logger = logging.getLogger(__name__)


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

def main(gui=False):
    """Main function for the CLI."""
    argu = parse_arguments(gui=gui)

    setup_logging(argu.log_level)

    # Reset logging levels if DEBUG (see deeplc.py)
    if argu.log_level.lower() == "debug":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        logging.getLogger('tensorflow').setLevel(logging.DEBUG)
        warnings.filterwarnings('default', category=DeprecationWarning)
        warnings.filterwarnings('default', category=FutureWarning)
        warnings.filterwarnings('default', category=UserWarning)
    else:
        os.environ['KMP_WARNINGS'] = '0'

    try:
        run(**vars(argu))
    except DeepLCError as e:
        logger.exception(e)
        sys.exit(1)


def run(
    file_pred,
    file_cal=None,
    file_pred_out=None,
    plot_predictions=False,
    file_model=None,
    pygam_calibration=False,
    split_cal=50,
    dict_divider=50,
    use_library=None,
    write_library=False,
    batch_num=50000,
    n_threads=None,
    log_level="info",
    verbose=True,
):
    """Run DeepLC."""

    logger.info("Using DeepLC version %s", __version__)
    logger.debug("Using %i CPU threads", n_threads)

    if not file_cal and file_model != None:
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

    if file_cal:
        df_cal = pd.read_csv(file_cal)
        if len(df_cal.columns) < 2:
            df_cal = pd.read_csv(df_cal,sep=" ")
        df_cal = df_cal.fillna("")

    # Make a feature extraction object; you can skip this if you do not want to
    # use the default settings for DeepLC. Here we want to use a model that does
    # not use RDKit features so we skip the chemical descriptor making
    # procedure.
    f_extractor = FeatExtractor(
        add_sum_feat=False,
        ptm_add_feat=False,
        ptm_subtract_feat=False,
        standard_feat=False,
        chem_descr_feat=False,
        add_comp_feat=False,
        cnn_feats=True,
        verbose=verbose
    )

    # Make the DeepLC object that will handle making predictions and calibration
    dlc = DeepLC(
        path_model=file_model,
        f_extractor=f_extractor,
        cnn_model=True,
        pygam_calibration=pygam_calibration,
        split_cal=split_cal,
        dict_cal_divider=dict_divider,
        write_library=write_library,
        use_library=use_library,
        batch_num=batch_num,
        n_jobs=n_threads,
        verbose=verbose,
    )

    # Calibrate the original model based on the new retention times
    if file_cal:
        logger.info("Selecting best model and calibrating predictions...")
        dlc.calibrate_preds(seq_df=df_cal)

    # Make predictions; calibrated or uncalibrated
    logger.info("Making predictions using model: %s", dlc.model)
    if file_cal:
        preds = dlc.make_preds(seq_df=df_pred)
    else:
        preds = dlc.make_preds(seq_df=df_pred, calibrate=False)

    df_pred["predicted_tr"] = preds
    logger.info("Writing predictions to file: %s", file_pred_out)
    df_pred.to_csv(file_pred_out)

    if plot_predictions:
        if file_cal and "tr" in df_pred.columns:
            file_pred_figure = os.path.splitext(file_pred_out)[0] + '.png'
            logger.info(
                "Saving scatterplot of predictions to file: %s",
                file_pred_figure
            )
            plt.figure(figsize=(11.5, 9))
            plt.scatter(df_pred["tr"], df_pred["predicted_tr"], s=3)
            plt.title("DeepLC predictions")
            plt.xlabel("Observed retention times")
            plt.ylabel("Predicted retention times")
            plt.savefig(file_pred_figure, dpi=300)
        else:
            logger.warning(
                "No observed retention time in input data. Cannot plot "
                "predictions."
            )

    logger.info("DeepLC finished!")


if __name__ == "__main__":
    main()
