"""
This code is used to test run lc_pep

For the library versions see the .yml file
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2019"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

from lc_pep_fit_model import fit_xgb, fit_xgb_leaf
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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--",c="grey")

def make_cv(df,X,y,ratio_test=0.9,seed=42):
    random.seed(seed)
    cv_dict = {}
    for seq,ident in zip(df["seq"],df.index):
        try:
            cv_dict[seq].append(ident)
        except:
            cv_dict[seq] = [ident]
    key_list = list(cv_dict.keys())
    random.shuffle(key_list)
    
    train_peps = key_list[0:int(len(key_list)*ratio_test)]
    test_peps = key_list[int(len(key_list)*ratio_test):]

    train_idents = list(itertools.chain(*[cv_dict[tp] for tp in train_peps]))
    test_idents = list(itertools.chain(*[cv_dict[tp] for tp in test_peps]))
    
    return train_idents, test_idents

def filter_df(df):
    df.fillna("",inplace=True)
    mods = [m.split("|") for m in df["modifications"]]
    
    mods_only = []
    for m in mods:
        mods_only_temp = []
        for i,m_temp in enumerate(m):
            if i % 2 != 0:
                mods_only_temp.append(m_temp)
        mods_only.append("|".join(mods_only_temp))
    df["mods_only"] = mods_only
    filter_single_mod = [rn for rn,r in df.iterrows() if len(r["mods_only"].split("|")) < 2]
    
    df = df.loc[filter_single_mod,:]
    df["idents"]= [s+"|"+m for s,m in zip(df["seq"],df["mods_only"])]
    print(set(df["mods_only"]))
    df.drop_duplicates(subset=["idents"],inplace=True)
    df.index = df["idents"]

    return df

def get_params_combinations(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return(combinations)

def rem_code():
    pepper = LCPep(config_file=config_file)
 
    #df = pd.read_csv("datasets/unmod.csv",sep=",")
    df = pd.read_csv("datasets/dia.csv",sep=",")
    #df = pd.read_csv("datasets/SCX.csv",sep=",")
    #df = pd.read_csv("datasets/Xbridge.csv",sep=",")
    #df = pd.read_csv("datasets/LUNA_SILICA.csv",sep=",")
    #df = pd.read_csv("datasets/LUNA_SILICA.csv",sep=",")
     
    df = filter_df(df)
 
    X = pepper.do_f_extraction_pd_parallel(df)
    X = pd.concat([X,df["tr"]],axis=1)
 
    y = X.pop("tr")
 
    param_dist =   {
            #'objective' : 'gpu:reg:linear',
            'tree_method':['gpu_hist'],
            "grow_policy" : ["lossguide"],
            #'learning_rate': [0.02], 
            'gamma' : [4,10], #0.1,0.25,0.5,1,2,
            "reg_alpha" : [4,10], #0.1,0.25,0.5,1,2,
            "reg_lambda" : [4,10], #0.1,0.25,0.5,1,2,
            #'min_child_weight' : [3],
            'nthread' : [8],
            'max_depth' : [25],
            'subsample' : [0.9], 
            'colsample_bytree' : [0.8], 
            'seed': [2100], 
            'eval_metric' : ["rmse"],
            #'num_boost_round' : [300],
            #'n_estimators': [999],
            'max_leaves': [4,8,12,24,48]
        }
 
    print(len(get_params_combinations(param_dist)))
    train_indices,test_indices = make_cv(df,X,y)
    highest_cor = 0.0
    for params in get_params_combinations(param_dist):
        train_preds,train_cross_preds,test_preds,xgb_model = fit_xgb_leaf(X.loc[train_indices,:],
                                                                      y.loc[train_indices],
                                                                      X.loc[test_indices,:],
                                                                      y.loc[test_indices],
                                                                      param_dist =  params)
        print(params,test_preds,scipy.stats.pearsonr(y.loc[test_indices],train_cross_preds),highest_cor,np.mean([abs(y-y_hat) for y,y_hat in zip(y.loc[test_indices],train_cross_preds)]),np.percentile([abs(y-y_hat) for y,y_hat in zip(y.loc[test_indices],train_cross_preds)],95))
        if scipy.stats.pearsonr(y.loc[test_indices],train_cross_preds)[0] > highest_cor:
            highest_cor = scipy.stats.pearsonr(y.loc[test_indices],train_cross_preds)[0]
            opt_params = params
            xgb_model_best = xgb_model
 
    print("The 95th percentile error:")
    print(np.percentile([abs(a-b) for a,b in zip(y,train_cross_preds)],95))
    print("The pearson correlation:")
    print(scipy.stats.pearsonr(y.loc[test_indices],train_cross_preds))
 
    plt.scatter(y.loc[test_indices],train_cross_preds)
    plt.show()
 
    df = pd.read_csv("datasets/seqs_exp.csv",sep=",")
    df.index = ["Pep_"+str(dfi) for dfi in df.index]
     
    pepper = LCPep(config_file=config_file,path_model=os.path.join(os.getcwd(),"mods/lcpep_synt.pickle"))
 
     
    random_picks = list(set(np.random.choice(len(df.index), tot_select_cal)))
     
    df["tr"] = df["tr"]/60
     
    pepper.calibrate_preds(df["seq"].iloc[random_picks],df["modifications"].iloc[random_picks],df.index,df["tr"].iloc[random_picks])
 
 
    plt.scatter(df["tr"],pepper.make_preds(seq_df=df))
    plt.scatter(df["tr"],pepper.make_preds(seq_df=df,calibrate=False))
    abline(1.0,0.0)
    plt.show()
 
    print(scipy.stats.pearsonr(df["tr"],pepper.make_preds(seq_df=df))[0])
    print(scipy.stats.pearsonr(df["tr"],pepper.make_preds(seq_df=df,calibrate=False))[0])

def main(config_file = "config.ini"):
    df = pd.read_csv("datasets/seqs_exp.csv",sep=",")

    df.index = ["Pep_"+str(dfi) for dfi in df.index]
    
    f_extractor = FeatExtractor(chem_descr_feat=False,
                                verbose=False)

    pepper = LCPep(config_file=config_file,
                   path_model=os.path.join(os.getcwd(),"mods/lcpep_synt.pickle"),
                   f_extractor=f_extractor,
                   verbose=False)

    df["tr"] = df["tr"]**0.85

    pepper.calibrate_preds(seq_df=df)

    print("Predictions (calibrated): ",pepper.make_preds(seq_df=df))
    print("Predictions (uncalibrated): ",pepper.make_preds(seq_df=df,calibrate=False))

    plt.scatter(df["tr"],pepper.make_preds(seq_df=df),label="Calibrated",s=1)
    plt.scatter(df["tr"],pepper.make_preds(seq_df=df,calibrate=False),label="Uncalibrated",s=1)
    abline(1.0,0.0)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
