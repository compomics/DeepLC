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

from lc_pep_fit_model import fit_xgb
from lc_pep import LCPep

# Native imports
import pickle
import sys
import os

# Pandas
import pandas as pd

# Matplotlib
from matplotlib import pyplot as plt

# SciPy
import scipy

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

def main(tot_select_cal = 1000,points_to_smooth = 50, config_file = "config.ini"):
    pepper = LCPep(config_file=config_file)

    df = pd.read_csv("parse_pride/seqs_exp.csv",sep=",")
    df.index = ["Pep_"+str(dfi) for dfi in df.index]

    X = pepper.do_f_extraction_pd_parallel(df)
    X = pd.concat([X,df["tr"]],axis=1)
    
    y = X.pop("tr")

    train_preds,train_cross_preds,test_preds,xgb_model = fit_xgb(X,y,X,y,config_file=config_file)
    print("The 95th percentile error:")
    print(np.percentile([abs(a-b) for a,b in zip(y,train_cross_preds)],95))
    print("The pearson correlation:")
    print(scipy.stats.pearsonr(y,train_cross_preds))

    plt.scatter(y,train_cross_preds)
    plt.show()

    print("The 95th percentile error:")
    print(np.percentile([abs(a-b) for a,b in zip(y,train_cross_preds)],95))
    print("The pearson correlation:")
    print(scipy.stats.pearsonr(y,train_cross_preds))
    plt.scatter(y,train_preds)
    plt.show()

    with open("xgb_production_lcpep.pickle","wb") as handle:
        pickle.dump(xgb_model, handle)
    
    pepper = LCPep(config_file=config_file,path_model=os.path.join(os.getcwd(),"xgb_production_lcpep.pickle"))
    df = pd.read_csv("parse_pride/seqs_exp.csv",sep=",")
    
    #df = df.iloc[list(set(np.random.choice(len(df.index), 5000)))]
    
    df.index = ["Pep_"+str(dfi) for dfi in df.index]
    random_picks = list(set(np.random.choice(len(df.index), tot_select_cal)))
    
    df["tr"] = df["tr"]**0.85
    
    pepper.calibrate_preds(df["seq"].iloc[random_picks],df["modifications"].iloc[random_picks],df.index,df["tr"].iloc[random_picks])

    plt.scatter(df["tr"],pepper.make_preds(df["seq"],df["modifications"],df.index))
    plt.scatter(df["tr"],pepper.make_preds(df["seq"],df["modifications"],df.index,calibrate=False))
    abline(1.0,0.0)
    plt.show()

if __name__ == "__main__":
    main()