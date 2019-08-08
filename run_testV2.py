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

from lc_pep_fit_model import fit_xgb, fit_xgb_leaf, fit_svr, fit_lasso
from lc_pep import LCPep
from feat_extractor import FeatExtractor
from sklearn import preprocessing

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

import scipy
import xgboost as xgb

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
    df.drop_duplicates(subset=["idents"],inplace=True)
    df.index = df["idents"]

    return df

def get_params_combinations(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return(combinations)

def train_wrapper(X,y,X_test,y_test,df): 
    param_dist =   {
            #'objective' : 'gpu:reg:linear',
            'tree_method':["gpu_hist"], #
            "grow_policy" : ["lossguide"],
            #'learning_rate': [0.02], 
            'gamma' : [0.5,1,2,4], #0.1,0.25,0.5,1,2,
            "reg_alpha" : [0.5,1,2,4], #0.1,0.25,0.5,1,2,
            "reg_lambda" : [0.5,1,2,4], #0.1,0.25,0.5,1,2,
            #'min_child_weight' : [3],
            'nthread' : [8],
            'max_depth' : [25], #25
            'subsample' : [0.9], 
            'colsample_bytree' : [0.8], 
            'seed': [2100], 
            'eval_metric' : ["rmse"],
            #'num_boost_round' : [300],
            #'n_estimators': [999],
            'max_leaves': [4,8,12,24,48],
            'n_jobs': [32]
        }
 
    print(len(get_params_combinations(param_dist)))
    train_indices,test_indices = make_cv(df.loc[X.index],X,y)
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
    return(xgb_model_best.predict(xgb.DMatrix(X_test,label = y_test)),y_test,X_test.index)

def change_df_shift(df,mods_to_include=[]):
    new_df = {}

    for um in mods_to_include:
        sub_df = df.loc[[rid for rid,r in df.iterrows() if um in r["modifications"]]]
        #peptides = df.loc[df["modifications"] == um]
        #print(peptides)
        shifts = []
        for rid,p in sub_df.iterrows():
            native_tr = list(df[(df["seq"] == p["seq"]) & (df["modifications"] == "")]["tr"])
            if len(native_tr) < 1:
                continue
            
            native_tr = native_tr[0]
            p["tr"] = native_tr-p["tr"]
            new_df[rid] = p
            #shifts.append(native_tr-p["tr"])
            #print(um,native_tr,p["tr"],native_tr-p["tr"])
            #print("=====")
    return(pd.DataFrame(new_df).T)
        

def main(config_file = "config.ini"):

    params = [
        [False,False,False,False],
        [True,True,True,False],
        [False,False,False,True],
        [True,True,True,True]
    ]

    #df = pd.read_csv("datasets/unmod.csv",sep=",")
    #df = pd.read_csv("datasets/dia.csv",sep=",")
    #df = pd.read_csv("datasets/SCX.csv",sep=",")
    #df = pd.read_csv("datasets/Xbridge.csv",sep=",")
    #df = pd.read_csv("datasets/LUNA_SILICA.csv",sep=",")
    #df = pd.read_csv("datasets/LUNA_SILICA.csv",sep=",")
    df = pd.read_csv("datasets/seqs_exp.csv",sep=",")

    df.fillna("",inplace=True)
    unique_mods = list(set("|".join(list(df["modifications"])).split("|")))
    
    um_list = []
    for um in unique_mods:
        try:
            int(um)
        except:
            if len(um) == 0: continue
            um_list.append(um)

     
    df.index = ["Pep_"+str(dfi) for dfi in df.index]
    #df = change_df_shift(df,mods_to_include=um_list)

    outfile = open("special_cv_preds_shift.csv","w")
    outfile.write("ident,mod,atom_all,atom_sum,atom_min,chemdesc,prediction,observed\n")

    for uml in um_list:
        perfs = []
        for i in range(len(params)):
            add_sum_feat,ptm_add_feat,ptm_subtract_feat,chem_descr_feat = params[i]
            
            f_extractor = FeatExtractor(standard_feat=True,
                                        add_sum_feat=add_sum_feat,
                                        ptm_add_feat=ptm_add_feat,
                                        ptm_subtract_feat=ptm_subtract_feat,
                                        chem_descr_feat=chem_descr_feat,
                                        verbose=False)

            pepper = LCPep(config_file=config_file,
                        path_model=os.path.join(os.getcwd(),"mods/lcpep_synt.pickle"),
                        f_extractor=f_extractor,
                        verbose=False)
            X = pepper.do_f_extraction_pd_parallel(df)

            scaler = preprocessing.StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=X.columns)

            X_train = X.loc[[rid for rid,r in df.iterrows() if uml not in r["modifications"]]]
            X_test = X.loc[[rid for rid,r in df.iterrows() if uml in r["modifications"]]]
            y_train = df.loc[[rid for rid,r in df.iterrows() if uml not in r["modifications"]]]["tr"]
            y_test = df.loc[[rid for rid,r in df.iterrows() if uml in r["modifications"]]]["tr"]

            preds,y_vals,ids = train_wrapper(X_train,y_train,X_test,y_test,df)

            y_vals = y_test
            ids  = y_test.index

            param_str = ",".join(map(str,params[i]))

            for p,v,i in zip(preds,y_vals,ids):
                outfile.write("%s,%s,%s,%s,%s\n" % (i,uml,param_str,p,v))
            outfile.flush()

if __name__ == "__main__":
    main()
