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
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from psm_utils.io import read_file

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
    file_model=None,
    pygam_calibration=True,
    split_cal=50,
    dict_divider=50,
    use_library=None,
    write_library=False,
    batch_num=50000,
    n_threads=None,
    transfer_learning=False,
    log_level="info",
    verbose=True,
):
    """Run DeepLC."""
    logger.info("Using DeepLC version %s", __version__)
    logger.debug("Using %i CPU threads", n_threads)

    df_pred = False
    df_cal = False
    first_line_pred = ""
    first_line_cal = ""

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
    
    with open(file_pred) as f:
        first_line_pred = f.readline().strip()
    if file_cal:
        with open(file_cal) as f:
            first_line_cal = f.readline().strip()

    if "modifications" in first_line_pred.split(",") and "seq" in first_line_pred.split(","):
        # Read input files
        df_pred = pd.read_csv(file_pred)
        if len(df_pred.columns) < 2:
            df_pred = pd.read_csv(file_pred,sep=" ")
        df_pred = df_pred.fillna("")
        file_pred = ""

        list_of_psms = []
        for seq,mod,ident in zip(df_pred["seq"],df_pred["modifications"],df_pred.index):
            list_of_psms.append(PSM(peptidoform=peprec_to_proforma(seq,mod),spectrum_id=ident))
        psm_list_pred = PSMList(psm_list=list_of_psms)
        df_pred = None
    else:
        psm_list_pred = read_file(file_pred)
        if "msms" in file_pred and ".txt" in file_pred:
            mapper = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "unimod/map_mq_file.csv"),index_col=0)["value"].to_dict()
            psm_list_pred.rename_modifications(mapper)

    # Allow for calibration file to be empty (undefined), fill in if/elif if present
    psm_list_cal = []
    if "modifications" in first_line_cal.split(",") and "seq" in first_line_cal.split(",") and file_cal:
        df_cal = pd.read_csv(file_cal)
        if len(df_cal.columns) < 2:
            df_cal = pd.read_csv(df_cal,sep=" ")
        df_cal = df_cal.fillna("")
        file_cal = ""

        list_of_psms = []
        for seq,mod,ident,tr in zip(df_cal["seq"],df_cal["modifications"],df_cal.index,df_cal["tr"]):
            list_of_psms.append(PSM(peptidoform=peprec_to_proforma(seq,mod),spectrum_id=ident,retention_time=tr))
        psm_list_cal = PSMList(psm_list=list_of_psms)
        df_cal = None
    elif file_cal:
        psm_list_cal = read_file(file_cal)
        if "msms" in file_cal and ".txt" in file_cal:
            mapper = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "unimod/map_mq_file.csv"),index_col=0)["value"].to_dict()
            psm_list_cal.rename_modifications(mapper)
    # Make a feature extraction object; you can skip this if you do not want to
    # use the default settings for DeepLC. Here we want to use a model that does
    # not use RDKit features so we skip the chemical descriptor making
    # procedure.
    f_extractor = FeatExtractor(
        cnn_feats=True,
        verbose=verbose
    )
    
    # Make the DeepLC object that will handle making predictions and calibration
    dlc = DeepLC(
        path_model=file_model,
        f_extractor=f_extractor,
        cnn_model=True,
        split_cal=split_cal,
        dict_cal_divider=dict_divider,
        write_library=write_library,
        use_library=use_library,
        batch_num=batch_num,
        n_jobs=n_threads,
        verbose=verbose,
        deeplc_retrain=transfer_learning
    )
    
    # Calibrate the original model based on the new retention times
    if len(psm_list_cal) > 0:
        logger.info("Selecting best model and calibrating predictions...")
        logger.info("Initiating transfer learning?")
        dlc.calibrate_preds(psm_list=psm_list_cal)

    # Make predictions; calibrated or uncalibrated
    logger.info("Making predictions using model: %s", dlc.model)
    if len(psm_list_cal) > 0:
        preds = dlc.make_preds(seq_df=df_pred, infile=file_pred, psm_list=psm_list_pred)
    else:
        preds = dlc.make_preds(seq_df=df_pred, infile=file_pred, psm_list=psm_list_pred, calibrate=False)
    
    #df_pred["predicted_tr"] = preds
    logger.info("Writing predictions to file: %s", file_pred_out)
    
    file_pred_out = open(file_pred_out,"w")
    file_pred_out.write("Sequence proforma,predicted retention time\n")
    for psm,tr in zip(psm_list_pred,preds):
        file_pred_out.write(f"{psm.peptidoform.proforma},{tr}\n")
    file_pred_out.close()

    logger.info("DeepLC finished!")


if __name__ == "__main__":
    main()
