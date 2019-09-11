from lc_pep import LCPep
from feat_extractor import FeatExtractor

# Native imports
import pickle
import sys
import os
import random
import itertools
import argparse
from collections import Counter
import re

# Pandas
import pandas as pd

# Matplotlib
from matplotlib import pyplot as plt

# Numpy
import numpy as np

def parse_arguments():
    """
    Read arguments from the command line
    Parameters
    ----------
        
    Returns
    -------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--file_pred", type=str, dest="file_pred",default="",
                        help="Specify the file and path to make predictions for")
    
    parser.add_argument("--file_cal", type=str, dest="file_cal", default="",
                        help="Specify the file and path for calibrating the predictions (leave empty for no calibration)")
    
    parser.add_argument("--file_pred_out", type=str, dest="file_pred_out", default="",
                        help="Specify the outputfile for the (calibrated) predictions")

    parser.add_argument("--file_model", type=str, dest="file_model", default="",
                        help="Specify the model to use to make the (calibrated) predictions")

    parser.add_argument("--n_threads", type=int, dest="n_threads", default=32,
                        help="Number of peaks to extract and consider for combinations in a spectrum")

    parser.add_argument("--split_cal", type=int, dest="split_cal", default=50,
                        help="Number of peaks to extract and consider for combinations in a spectrum")

    parser.add_argument("--dict_divider", type=int, dest="dict_divider", default=50,
                        help="Number of peaks to extract and consider for combinations in a spectrum")

    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    results = parser.parse_args()

    return results

def main():
    argu = parse_argument()

    run(file_pred=argu.file_pred,
        file_cal=argu.file_cal,
        file_pred_out=argu.file_pred_out,
        file_model=argu.file_model,
        n_threads=argu.n_threads,
        split_cal=argu.split_cal,
        dict_divider=argu.dict_divider)

def run(file_pred="",
        file_cal="",
        file_pred_out="",
        file_model="",
        n_threads=32,
        split_cal=50,
        dict_divider=50):

    df_pred = pd.read_csv(file_pred)
    df_pred = df.fillna("")

    if len(file_cal) > 1:
        df_cal = pd.read_csv(file_cal)
        df_cal = df.fillna("")

    # Make a feature extraction object; you can skip this if you do not want to use the default settings
    # for pep_lc. Here we want to use a model that does not use RDKit features so we skip the chemical
    # descriptor making procedure.
    f_extractor = FeatExtractor(add_sum_feat=False,
                                ptm_add_feat=False,
                                ptm_subtract_feat=False,
                                standard_feat = False,
                                chem_descr_feat = False,
                                add_comp_feat = False,
                                cnn_feats = True,
                                verbose = True)
    
    # Make the pep_lc object that will handle making predictions and calibration
    pepper = LCPep(path_model=file_model,
                   f_extractor=f_extractor,
                   cnn_model=True,
                   verbose=False)

    # Calibrate the original model based on the new retention times
    if len(file_cal) > 1:
        pepper.calibrate_preds(seq_df=df_cal)
    
    # Make predictions; calibrated and uncalibrated
    if len(file_cal) > 1:
        preds = pepper.make_preds(seq_df=df_cal)
    else:
        preds = pepper.make_preds(seq_df=df_cal,calibrate=False)

    df_pred["Predicted tR"] = preds

    if len(file_cal) > 1:
        plt.figure(figsize=(11.5,9))
        plt.scatter(df_cal["tr"],preds,s=3)
        plt.title("Predicted retention times")
        plt.xlabel("Observed tr")
        plt.ylabel("Predicted tr")
        plt.savefig("preds_%s.png" % (file_pred), dpi=150)

if __name__ == "__main__":
    main()