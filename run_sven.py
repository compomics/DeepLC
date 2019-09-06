import pandas as pd
import numpy as np
from collections import Counter
import re

from lc_pep import LCPep
from feat_extractor import FeatExtractor

# Native imports
import pickle
import sys
import os
import random
import itertools

# Pandas
import pandas as pd

# Matplotlib
from matplotlib import pyplot as plt

# Numpy
import numpy as np

def main():
    df = pd.read_csv("Sven/Velos005137.mgf.ionbot.csv")
    
    # The way we treat modifictions is a bit different, make function...
    df = df.fillna("")
    mod_replace = lambda s : "|".join([su.capitalize() for su in re.sub("\[.*\]","",s).split("|")])
    df["modifications"] = df["modifications"].apply(mod_replace)
    df.rename(columns={'matched_peptide':'seq',
                       'rt':'tr'}, 
                       inplace=True)
    #df = df[df["q_value"] < 0.01]
    #df = df[df["DB"] != "D"]

    train_ids = [k for k,v in Counter(df["scan_id"]).items() if v == 1]
    df_cal = df[df['scan_id'].isin(train_ids)]

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
    pepper = LCPep(config_file = "config.ini",
                path_model=os.path.join(os.getcwd(),"mods/full_dia.hdf5"), #"mods/dia_no_mod.pickle"
                f_extractor=f_extractor,
                cnn_model=True,
                verbose=False)

    # Calibrate the original model based on the new retention times
    pepper.calibrate_preds(seq_df=df_cal)
    
    # Make predictions; calibrated and uncalibrated
    calib_train_preds = pepper.make_preds(seq_df=df_cal)
    uncalib_train_preds = pepper.make_preds(seq_df=df_cal,calibrate=False)
    print("Predictions (calibrated): ",calib_train_preds[0:5])
    print("Predictions (uncalibrated): ",uncalib_train_preds)

    plt.figure(figsize=(11.5,9))
    # compare calibrated and uncalibrated predictions
    plt.scatter(df_cal["tr"],calib_train_preds,label="Calibrated",s=1)
    plt.scatter(df_cal["tr"],uncalib_train_preds,label="Uncalibrated",s=1)
    plt.title("Train data (no rank contender)")
    plt.xlabel("Observed tr")
    plt.ylabel("Predicted tr")
    plt.legend()
    plt.savefig("train_cal.png", dpi=150)
    #plt.show()

    test_ids = [k for k,v in Counter(df["scan_id"]).items() if v != 1]
        
    df_test = df[df['scan_id'].isin(test_ids)]

    calib_test_preds = pepper.make_preds(seq_df=df_test)
    uncalib_test_preds = pepper.make_preds(seq_df=df_test,calibrate=False)
    
    plt.figure(figsize=(11.5,9))
    # compare calibrated and uncalibrated predictions
    plt.scatter(df_test["tr"],calib_test_preds,label="Calibrated",s=1)
    plt.scatter(df_test["tr"],uncalib_test_preds,label="Uncalibrated",s=1)
    plt.title("Test data (rank contenders)")
    plt.xlabel("Observed tr")
    plt.ylabel("Predicted tr")
    plt.legend()
    plt.savefig("test_cal.png", dpi=150)
    #plt.show()

    plt.figure(figsize=(11.5,9))
    plt.scatter(abs(df_test["tr"]-calib_test_preds),df_test["ionbot_score"],s=1)
    plt.title("Test data (rank contenders)")
    plt.xlabel("Error abs(tr_pred - tr_obs)")
    plt.ylabel("ionbot_score")
    plt.savefig("test_error_abs.png", dpi=150)
    #plt.show()
    
    plt.figure(figsize=(11.5,9))
    plt.scatter(df_test["tr"]-calib_test_preds,df_test["ionbot_score"],s=1)
    plt.title("Test data (rank contenders)")
    plt.xlabel("Error tr_pred - tr_obs")
    plt.ylabel("ionbot_score")
    plt.savefig("test_error.png", dpi=150)
    #plt.show()

    for f in ["rank","percolator_psm_score","q_value","PEP","percolator_psm_score_scan","q_value_scan","PEP_scan"]:
        plt.figure(figsize=(11.5,9))
        plt.scatter(df_test["tr"]-calib_test_preds,df_test[f],s=1)
        plt.title("Test data (rank contenders)")
        plt.xlabel("Error tr_pred - tr_obs")
        plt.ylabel(f)
        plt.savefig("test_error_%s.png" % (f), dpi=150)


if __name__ == "__main__":
    main()