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

# Native imports
import os
import time
import pickle
from operator import itemgetter
import sys
from configparser import ConfigParser
import time

# Pandas
import pandas as pd

# Numpy
import numpy as np

# XGBoost
import xgboost as xgb

# Feature extraction
from feat_extractor import FeatExtractor

# Multiproc
from multiprocessing import Pool

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class LCPep():
    def __init__(self,
                main_path=os.getcwd(),
                path_model=None,
                verbose=True,
                bin_dist=1,
                dict_cal_divider = 100,
                split_cal = 25,
                n_jobs=32,
                config_file=None,
                f_extractor=None):
        
        # if a config file is defined overwrite standard parameters
        if config_file:
            cparser = ConfigParser()
            cparser.read(config_file)
            dict_cal_divider = cparser.getint("lcPep","dict_cal_divider")
            split_cal = cparser.getint("lcPep","split_cal")
            n_jobs = cparser.getint("lcPep","n_jobs")

        self.main_path = main_path
        self.verbose = verbose
        self.bin_dist = bin_dist
        self.calibrate_dict = {}
        self.calibrate_min = float('inf')
        self.calibrate_max = 0

        self.dict_cal_divider = dict_cal_divider
        self.split_cal = split_cal
        self.n_jobs = n_jobs

        if path_model:
            with open(path_model, "rb") as handle:
                self.model = pickle.load(handle)
        
        if f_extractor:
            self.f_extractor = f_extractor
        else:
            self.f_extractor = FeatExtractor()
    
    def __str__(self):
        return("""
  _     ____                    
 | |   / ___|  _ __   ___ _ __  
 | |  | |     | '_ \ / _ \ '_ \ 
 | |__| |___  | |_) |  __/ |_) |
 |_____\____| | .__/ \___| .__/ 
              |_|        |_|    
              """)
        

    def do_f_extraction(self,
                        seqs,
                        mods,
                        identifiers):
        """
        Extract all features we can extract; without parallelization; use if you want to run feature extraction
        with a single core

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        return self.f_extractor.full_feat_extract(seqs,mods,identifiers)

    def do_f_extraction_pd(self,
                        df_instances):
        """
        Extract all features we can extract; without parallelization; use if you want to run feature extraction
        with a single thread; and use a defined dataframe

        Parameters
        ----------
        df_instances : pd.DataFrame
            dataframe containing the sequences (column:seq), modifications (column:modifications) and naming (column:index)

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        return self.f_extractor.full_feat_extract(df_instances["seq"],df_instances["modifications"],df_instances.index)
    
    def do_f_extraction_pd_parallel(self,
                        df_instances):
        """
        Extract all features we can extract; with parallelization; use if you want to run feature extraction
        with multiple threads; and use a defined dataframe

        Parameters
        ----------
        df_instances : pd.DataFrame
            dataframe containing the sequences (column:seq), modifications (column:modifications) and naming (column:index)
        
        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        df_instances_split = np.array_split(df_instances, self.n_jobs)
        pool = Pool(self.n_jobs)
        df = pd.concat(pool.map(self.do_f_extraction_pd, df_instances_split))
        pool.close()
        pool.join()
        return df

    def make_preds(self,
                seqs=[],
                mods=[],
                identifiers=[],
                calibrate=True,
                seq_df=None):
        """
        Make predictions for sequences

        Parameters
        ----------
        seq_df : pd.DataFrame
            dataframe containing the sequences (column:seq), modifications (column:modifications) and naming (column:index);
            will use parallel by default! TODO if n_jobs is set to 1 even with pd df use single thread
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        calibrate : boolean
            calibrate predictions or just return the predictions

        Returns
        -------
        np.array
            predictions
        """
        # TODO make feature extraction run in paralel
        try: 
            seq_df.index
            X = self.do_f_extraction_pd_parallel(seq_df)
            X = X.loc[seq_df.index]
        except KeyError:
            X = self.f_extractor.full_feat_extract(seqs,mods,identifiers)
            X = X.loc[identifiers]        

        X = X[self.model.feature_names]
        
        if calibrate:
            cal_preds = []
            #X = xgb.DMatrix(X)

            # first get uncalibrated prediction
            uncal_preds = self.model.predict(X)

            for uncal_pred in uncal_preds:
                try:
                    slope,intercept,x_correction = self.calibrate_dict[str(round(uncal_pred,self.bin_dist))]
                    cal_preds.append(slope * (uncal_pred-x_correction) + intercept)
                except KeyError:
                    # outside of the prediction range ... use the last calibration curve
                    if uncal_pred <= self.calibrate_min:
                        slope,intercept,x_correction = self.calibrate_dict[str(round(self.calibrate_min,self.bin_dist))]
                        cal_preds.append(slope * (uncal_pred-x_correction) + intercept)
                    elif uncal_pred >= self.calibrate_max:
                        slope,intercept,x_correction = self.calibrate_dict[str(round(self.calibrate_max,self.bin_dist))]
                        cal_preds.append(slope * (uncal_pred-x_correction) + intercept)
                    else:
                        slope,intercept,x_correction = self.calibrate_dict[str(round(self.calibrate_max,self.bin_dist))]
                        cal_preds.append(slope * (uncal_pred-x_correction) + intercept)
            return np.array(cal_preds)
        else:
            #X = xgb.DMatrix(X)            
            return self.model.predict(X)

    def calibrate_preds(self,
                        seqs=[],
                        mods=[],
                        identifiers=[],
                        measured_tr=[],
                        seq_df=None,
                        use_median=True):
        """
        Make calibration curve for predictions TODO make similar function for pd.DataFrame

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        measured_tr : list
            measured tr of the peptides; should correspond to seqs, identifiers, and mods

        Returns
        -------
        
        """
        # TODO there is already a prediction function... use that.
        #try: 
        X = self.do_f_extraction_pd_parallel(seq_df)
        
        X = X.loc[seq_df.index]
        measured_tr = seq_df["tr"]
        #X = X[self.model.feature_names]
        #except KeyError:
        #    print(seqs,mods,identifiers)
        #    X = self.f_extractor.full_feat_extract(seqs,mods,identifiers)
        #    print(X)
        #    X = X.loc[identifiers]
        
        #X = xgb.DMatrix(X)
        
        predicted_tr = self.model.predict(X)
        if self.verbose: t0 = time.time()
        
        # sort two lists, predicted and observed based on measured tr
        tr_sort = [(mtr,ptr) for mtr,ptr in sorted(zip(measured_tr,predicted_tr), key=lambda pair: pair[0])]
        measured_tr = [mtr for mtr,ptr in tr_sort]
        predicted_tr = [ptr for mtr,ptr in tr_sort]

        mtr_mean = []
        ptr_mean = []

        # smooth between observed and predicted
        for mtr,ptr in zip(self.split_seq(measured_tr,self.split_cal),self.split_seq(predicted_tr,self.split_cal)):
            if use_median:
                mtr_mean.append(sum(mtr)/len(mtr))
                ptr_mean.append(sum(ptr)/len(ptr))
            else:
                mtr_mean.append(np.median(mtr))
                ptr_mean.append(np.median(ptr))

        # calculate calibration curves
        for i in range(0,len(ptr_mean)):
            if i >= len(ptr_mean)-1: continue
            delta_ptr = ptr_mean[i+1]-ptr_mean[i]
            delta_mtr = mtr_mean[i+1]-mtr_mean[i]

            slope = delta_mtr/delta_ptr
            intercept = mtr_mean[i]
            x_correction = ptr_mean[i]

            # optimized predictions using a dict to find calibration curve very fast
            for v in np.arange(round(ptr_mean[i],self.bin_dist),round(ptr_mean[i+1],self.bin_dist),1/((self.bin_dist)*self.dict_cal_divider)):
                if v < self.calibrate_min:
                    self.calibrate_min = v
                if v > self.calibrate_max:
                    self.calibrate_max = v
                self.calibrate_dict[str(round(v,1))] = [slope,intercept,x_correction]

        if self.verbose: print("Time to calibrate: %s seconds" % (time.time() - t0))

    def split_seq(self,
                a,
                n):
        """
        Split a list (a) into multiple chunks (n)

        Parameters
        ----------
        a : list
            list to split
        n : list
            number of chunks

        Returns
        -------
        list
            chunked list
        """

        # since chunking is not alway possible do the modulo of residues
        k, m = divmod(len(a), n)
        return(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
