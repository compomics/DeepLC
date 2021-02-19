"""
Main code used to generate LC retention time predictions. This provides the main
interface.

For the library versions see the .yml file
"""

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__credits__ = ["Robbin Bouwmeester", "Ralf Gabriels", "Prof. Lennart Martens", "Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]


# Default models, will be used if no other is specified. If no best model is
# selected during calibration, the first model in the list will be used.
import os
deeplc_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODELS = [
    "mods/full_hc_hela_hf_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5", 
    "mods/full_hc_hela_hf_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5", 
    "mods/full_hc_hela_hf_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
    "mods/full_hc_PXD005573_mcp_cb975cfdd4105f97efa0b3afffe075cc.hdf5"
]
DEFAULT_MODELS = [os.path.join(deeplc_dir, dm) for dm in DEFAULT_MODELS]


# Native imports
from configparser import ConfigParser
from operator import itemgetter
import copy
import gc
import logging
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import sys
import time

# Pandas
import pandas as pd

# Numpy
import numpy as np

# Keras
import tensorflow as tf

from tensorflow.keras.models import load_model

# "Costum" activation function
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1, max_value=20.0)

try: from tensorflow.compat.v1.keras.backend import set_session
except ImportError: from tensorflow.keras.backend import set_session
try: from tensorflow.compat.v1.keras.backend import clear_session
except ImportError: from tensorflow.keras.backend import clear_session
try: from tensorflow.compat.v1.keras.backend import get_session
except ImportError: from tensorflow.keras.backend import get_session


# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set for TF V1.0 (counters some memory problems of nvidia 20 series GPUs)
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

# Set for TF V2.0 (counters some memory problems of nvidia 20 series GPUs)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

# Feature extraction
from deeplc.feat_extractor import FeatExtractor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    #sess = get_session()
    gc.collect()
    # Set to force CPU calculations
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DeepLC():
    """
    DeepLC predictor.

    Parameters
    ----------
    main_path : str
        main path of module
    path_model : str
        path to model file: leave empty to use default models
    verbose : bool
        turn logging on/off
    bin_dist : float
        TODO
    dict_cal_divider : int
        TODO
    split_cal : int
        TODO
    n_jobs : int or None
        number of threads to use; if None, use maximum available
    config_file : str or None
        path to configuration file
    f_extractor : object :: deeplc.FeatExtractor or None
        deeplc.FeatExtractor object to use
    cnn_model : bool
        use CNN model or not
    batch_num : int
        number of peptides per batch; lower for lower memory footprint

    Methods
    -------
    calibrate_preds(seqs=[], mods=[], identifiers=[], measured_tr=[], correction_factor=1.0, seq_df=None, use_median=True)
        Find best model and calibrate
    make_preds(seqs=[], mods=[], identifiers=[], calibrate=True, seq_df=None, correction_factor=1.0, mod_name=None)
        Make predictions

    """

    def __init__(self,
                 main_path=os.path.dirname(os.path.realpath(__file__)),
                 path_model=None,
                 verbose=True,
                 bin_dist=2,
                 dict_cal_divider=100,
                 split_cal=25,
                 n_jobs=None,
                 config_file=None,
                 f_extractor=None,
                 cnn_model=False,
                 batch_num=350000):
        
        # if a config file is defined overwrite standard parameters
        if config_file:
            cparser = ConfigParser()
            cparser.read(config_file)
            dict_cal_divider = cparser.getint("DeepLC", "dict_cal_divider")
            split_cal = cparser.getint("DeepLC", "split_cal")
            n_jobs = cparser.getint("DeepLC", "n_jobs")

        self.main_path = main_path
        self.verbose = verbose
        self.bin_dist = bin_dist
        self.calibrate_dict = {}
        self.calibrate_min = float("inf")
        self.calibrate_max = 0
        self.cnn_model = cnn_model

        self.batch_num = batch_num
        self.dict_cal_divider = dict_cal_divider
        self.split_cal = split_cal
        self.n_jobs = n_jobs

        max_threads = multiprocessing.cpu_count()
        if not self.n_jobs:
            self.n_jobs = max_threads
        elif self.n_jobs > max_threads:
            self.n_jobs = max_threads

        tf.config.threading.set_intra_op_parallelism_threads(n_jobs)
        tf.config.threading.set_inter_op_parallelism_threads(n_jobs)
        
        if "NUMEXPR_MAX_THREADS" not in os.environ:
            os.environ['NUMEXPR_MAX_THREADS'] = str(n_jobs)

        if path_model:
            if self.cnn_model:
                self.model = path_model
            else:
                with open(path_model, "rb") as handle:
                    self.model = pickle.load(handle)
        else:
            # Use default models
            self.cnn_model = True
            self.model = DEFAULT_MODELS

        if f_extractor:
            self.f_extractor = f_extractor
        else:
            self.f_extractor = FeatExtractor()

    def __str__(self):
        return("""
  _____                  _      _____
 |  __ \                | |    / ____|
 | |  | | ___  ___ _ __ | |   | |
 | |  | |/ _ \/ _ \ '_ \| |   | |
 | |__| |  __/  __/ |_) | |___| |____
 |_____/ \___|\___| .__/|______\_____|
                  | |
                  |_|
              """)

    def do_f_extraction(self,
                        seqs,
                        mods,
                        identifiers,
                        charges=[]):
        """
        Extract all features we can extract; without parallelization; use if you
        want to run feature extraction with a single core

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
        if len(charges) > 0:
            return self.f_extractor.full_feat_extract(seqs, mods, identifiers,charges=charges)
        else:
            return self.f_extractor.full_feat_extract(seqs, mods, identifiers)

    def do_f_extraction_pd(self,
                           df_instances):
        """
        Extract all features we can extract; without parallelization; use if
        you want to run feature extraction with a single thread; and use a
        defined dataframe

        Parameters
        ----------
        df_instances : object :: pd.DataFrame
            dataframe containing the sequences (column:seq), modifications
            (column:modifications) and naming (column:index)

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        if "charges" in df_instances.columns:
            return self.f_extractor.full_feat_extract(
                df_instances["seq"],
                df_instances["modifications"],
                df_instances.index,
                charges=df_instances["charges"]
                )
        else:
            return self.f_extractor.full_feat_extract(
                df_instances["seq"],
                df_instances["modifications"],
                df_instances.index)

    def do_f_extraction_pd_parallel(self,
                                    df_instances):
        """
        Extract all features we can extract; with parallelization; use if you
        want to run feature extraction with multiple threads; and use a defined
        dataframe

        Parameters
        ----------
        df_instances : object :: pd.DataFrame
            dataframe containing the sequences (column:seq), modifications
            (column:modifications) and naming (column:index)

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        df_instances_split = np.array_split(df_instances, self.n_jobs)
        if multiprocessing.current_process().daemon:
            logging.warn("DeepLC is running in a daemon process. Disabling multiprocessing as daemonic processes can't have children.")
            pool = multiprocessing.dummy.Pool(1)
        else:
            pool = multiprocessing.Pool(self.n_jobs)
        if self.n_jobs == 1:
            df = self.do_f_extraction_pd(df_instances)
        else:
            df = pd.concat(
                pool.map(
                    self.do_f_extraction_pd,
                    df_instances_split))
        pool.close()
        pool.join()
        return df

    def calibration_core(self,uncal_preds,cal_dict,cal_min,cal_max):
        cal_preds = []
        for uncal_pred in uncal_preds:
            try:
                slope, intercept = cal_dict[str(
                    round(uncal_pred, self.bin_dist))]
                cal_preds.append(
                    slope * (uncal_pred) + intercept)
            except KeyError:
                # outside of the prediction range ... use the last
                # calibration curve
                if uncal_pred <= cal_min:
                    slope, intercept = cal_dict[str(
                        round(cal_min, self.bin_dist))]
                    cal_preds.append(
                        slope * (uncal_pred) + intercept)
                elif uncal_pred >= cal_max:
                    slope, intercept = cal_dict[str(
                        round(cal_max, self.bin_dist))]
                    cal_preds.append(
                        slope * (uncal_pred) + intercept)
                else:
                    slope, intercept = cal_dict[str(
                        round(cal_max, self.bin_dist))]
                    cal_preds.append(
                        slope * (uncal_pred) + intercept)
        return np.array(cal_preds)

    def make_preds_core(self,
                        seq_df=None,
                        seqs=[],
                        mods=[],
                        identifiers=[],
                        calibrate=True,
                        correction_factor=1.0,
                        mod_name=None):
        """
        Make predictions for sequences

        Parameters
        ----------
        seq_df : object :: pd.DataFrame
            dataframe containing the sequences (column:seq), modifications
            (column:modifications) and naming (column:index); will use parallel
            by default!
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        calibrate : boolean
            calibrate predictions or just return the predictions
        correction_factor : float
            correction factor to apply to predictions
        mod_name : str or None
            specify a model to use instead of the model assigned originally to
            this instance of the object

        Returns
        -------
        np.array
            predictions
        """

        # See if we got a list; if not assume we got a df
        if len(seqs) == 0:
            # Make a copy, because we do not want to change to original df
            seq_df = seq_df.copy()
        else:
            # Make a df out of provided lists
            seq_df = pd.DataFrame([seqs, mods]).T
            seq_df.columns = ["seq", "modifications"]
            seq_df.index = identifiers

        # Only run on unique peptides, defined by seq+mod
        # TODO sort the mods in the peprec on both position and alphabet mod;
        # to not let duplicates through!
        if "charges" in seq_df.columns:
            seq_df["idents"] = seq_df["seq"] + "|" + seq_df["modifications"] + "|" + seq_df["charges"].astype(str)
        else:
            seq_df["idents"] = seq_df["seq"] + "|" + seq_df["modifications"]

        identifiers = list(seq_df.index)

        # Save a row identifier to seq+mod mapper so output has expected return
        # shapes
        identifiers_to_seqmod = dict(zip(seq_df.index, seq_df["idents"]))

        # Drop duplicated seq+mod
        seq_df.drop_duplicates(subset=["idents"], inplace=True)

        if self.verbose:
            cnn_verbose = 1
        else:
            cnn_verbose = 0

        # If we need to apply deep NN
        if self.cnn_model:
            if self.verbose:
                logging.debug("Extracting features for the CNN model ...")
            X = self.do_f_extraction_pd_parallel(seq_df)
            X = X.loc[seq_df.index]

            X_sum = np.stack(X["matrix_sum"])
            X_global = np.concatenate((np.stack(X["matrix_all"]),
                                       np.stack(X["pos_matrix"])),
                                      axis=1)
            X_hc = np.stack(X["matrix_hc"])
            X = np.stack(X["matrix"])
        else:
            if self.verbose:
                logging.debug(
                    "Extracting features for the predictive model ...")
            seq_df.index
            X = self.do_f_extraction_pd_parallel(seq_df)
            X = X.loc[seq_df.index]

            X = X[self.model.feature_names]

        ret_preds = []

        # If we need to calibrate
        if calibrate:
            assert self.calibrate_dict, "DeepLC instance is not yet calibrated.\
 Calibrate before making predictions, or use calibrate=False"

            if self.verbose:
                logging.debug("Predicting with calibration...")

            

            # Load the model differently if we are going to use a CNN
            if self.cnn_model:
                # TODO this is madness! Only allow dicts to come through this function...
                if isinstance(self.model, dict):
                    ret_preds = []
                    for m_group_name,m_name in self.model.items():
                        mod = load_model(m_name,
                                         custom_objects = {'<lambda>': lrelu})
                        uncal_preds = mod.predict(
                            [X, X_sum, X_global, X_hc], batch_size=5120).flatten() / correction_factor
                        
                        p = list(self.calibration_core(uncal_preds,self.calibrate_dict[m_name],self.calibrate_min[m_name],self.calibrate_max[m_name]))
                        ret_preds.append(p)
                    ret_preds = np.array([sum(a)/len(a) for a in zip(*ret_preds)])
                elif not mod_name:
                    mod = load_model(self.model,
                                     custom_objects = {'<lambda>': lrelu})
                    uncal_preds = mod.predict(
                        [X, X_sum, X_global, X_hc], batch_size=5120).flatten() / correction_factor
                    ret_preds = self.calibration_core(uncal_preds,self.calibrate_dict,self.calibrate_min,self.calibrate_max)
                else:
                    mod = load_model(mod_name,
                                     custom_objects = {'<lambda>': lrelu})
                    uncal_preds = mod.predict(
                        [X, X_sum, X_global, X_hc], batch_size=5120).flatten() / correction_factor
                    ret_preds = self.calibration_core(uncal_preds,self.calibrate_dict,self.calibrate_min,self.calibrate_max)
            else:
                # first get uncalibrated prediction
                uncal_preds = self.model.predict(X) / correction_factor

            
        else:
            if self.verbose:
                logging.debug("Predicting without calibration...")

            # Load the model differently if we use CNN
            if self.cnn_model:
                if not mod_name:
                    if isinstance(self.model, dict):
                        ret_preds = []
                        for m_group_name,m_name in self.model.items():
                            mod = load_model(m_name,
                                            custom_objects = {'<lambda>': lrelu})
                            p = mod.predict(
                                [X, X_sum, X_global, X_hc], batch_size=5120).flatten() / correction_factor
                            ret_preds.append(p)
                        ret_preds = np.array([sum(a)/len(a) for a in zip(*ret_preds)])
                    elif isinstance(self.model, list):
                        mod_name = self.model[0]
                        mod = load_model(mod_name,
                                         custom_objects = {'<lambda>': lrelu})
                        ret_preds = mod.predict([X,
                                                X_sum,
                                                X_global,
                                                X_hc],
                                                batch_size=5120,
                                                verbose=cnn_verbose).flatten() / correction_factor
                    elif isinstance(self.model, str):
                        mod_name = self.model
                        mod = load_model(mod_name,
                                         custom_objects = {'<lambda>': lrelu})
                        ret_preds = mod.predict([X,
                                                X_sum,
                                                X_global,
                                                X_hc],
                                                batch_size=5120,
                                                verbose=cnn_verbose).flatten() / correction_factor
                    else:
                        logging.critical('No CNN model defined.')
                        exit(1)
                else:
                    mod = load_model(mod_name,
                                    custom_objects = {'<lambda>': lrelu})
                    ret_preds = mod.predict([X,
                                            X_sum,
                                            X_global,
                                            X_hc],
                                            batch_size=5120,
                                            verbose=cnn_verbose).flatten() / correction_factor
            else:
                ret_preds = self.model.predict(X) / correction_factor

        pred_dict = dict(zip(seq_df["idents"], ret_preds))

        # Map from unique peptide identifiers to the original dataframe
        ret_preds_shape = []
        for ident in identifiers:
            ret_preds_shape.append(pred_dict[identifiers_to_seqmod[ident]])

        if self.verbose:
            logging.debug("Predictions done ...")

        # Below can cause freezing on some systems
        # It is meant to clear any remaining vars in memory
        reset_keras()
        del mod

        return ret_preds_shape


    def make_preds(self,
                   seqs=[],
                   mods=[],
                   identifiers=[],
                   calibrate=True,
                   seq_df=None,
                   correction_factor=1.0,
                   mod_name=None):
        """
        Make predictions for sequences, in batches if required.

        Parameters
        ----------
        seq_df : object :: pd.DataFrame
            dataframe containing the sequences (column:seq), modifications
            (column:modifications) and naming (column:index); will use parallel
            by default!
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        calibrate : boolean
            calibrate predictions or just return the predictions
        correction_factor : float
            correction factor to apply to predictions
        mod_name : str or None
            specify a model to use instead of the model assigned originally to
            this instance of the object

        Returns
        -------
        np.array
            predictions
        """
        if self.batch_num == 0:
            return self.make_preds_core(seqs=seqs,
                                        mods=mods,
                                        identifiers=identifiers,
                                        calibrate=calibrate,
                                        seq_df=seq_df,
                                        correction_factor=correction_factor,
                                        mod_name=mod_name)
        else:
            ret_preds = []
            if len(seqs) > 0:
                seq_df = pd.DataFrame({"seq": seqs,
                                       "modifications": mods},
                                      index=identifiers)
            for g, seq_df_t in seq_df.groupby(
                    np.arange(len(seq_df)) // self.batch_num):
                temp_preds = self.make_preds_core(
                    identifiers=identifiers,
                    calibrate=calibrate,
                    seq_df=seq_df_t,
                    correction_factor=correction_factor,
                    mod_name=mod_name)
                ret_preds.extend(temp_preds)

                # if self.verbose:
                logging.debug(
                    "Finished predicting retention time for: %s/%s" %
                    (len(ret_preds), len(seq_df)))
            return ret_preds

    def calibrate_preds_func(self,
                             seqs=[],
                             mods=[],
                             identifiers=[],
                             measured_tr=[],
                             correction_factor=1.0,
                             seq_df=None,
                             use_median=True,
                             mod_name=None):
        """
        Make calibration curve for predictions

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        measured_tr : list
            measured tr of the peptides; should correspond to seqs, identifiers,
            and mods
        correction_factor : float
            correction factor that needs to be applied to the supplied measured
            trs
        seq_df : object :: pd.DataFrame
            a pd.DataFrame that contains the sequences, modifications and
            observed retention times to fit a calibration curve
        use_median : boolean
            flag to indicate we need to use the median valuein a window to
            perform calibration
        mod_name
            specify a model to use instead of the model assigned originally to
            this instance of the object

        Returns
        -------
        float
            the minimum value where a calibration curve was fitted, lower values
            will be extrapolated from the minimum fit of the calibration curve
        float
            the maximum value where a calibration curve was fitted, higher values
            will be extrapolated from the maximum fit of the calibration curve
        dict
            dictionary with keys for rounded tr, and the values concern a linear
            model that should be applied to do calibration (!!! what is the
            shape of this?)
        """
        if len(seqs) == 0:
            seq_df.index
            predicted_tr = self.make_preds(
                seq_df=seq_df,
                calibrate=False,
                correction_factor=correction_factor,
                mod_name=mod_name)
            measured_tr = seq_df["tr"]
        else:
            predicted_tr = self.make_preds(
                seqs=seqs,
                mods=mods,
                identifiers=identifiers,
                calibrate=False,
                correction_factor=correction_factor,
                mod_name=mod_name)

        # sort two lists, predicted and observed based on measured tr
        tr_sort = [(mtr, ptr) for mtr, ptr in sorted(
            zip(measured_tr, predicted_tr), key=lambda pair: pair[1])]
        measured_tr = np.array([mtr for mtr, ptr in tr_sort])
        predicted_tr = np.array([ptr for mtr, ptr in tr_sort])

        mtr_mean = []
        ptr_mean = []

        calibrate_dict = {}
        calibrate_min = float('inf')
        calibrate_max = 0

        if self.verbose:
            logging.debug(
                "Selecting the data points for calibration (used to fit the\
linear models between)"
            )

        # smooth between observed and predicted
        split_val = predicted_tr[-1]/self.split_cal
        for range_calib_number in np.arange(0.0,predicted_tr[-1],split_val):
            ptr_index_start = np.argmax(predicted_tr>=range_calib_number)
            ptr_index_end = np.argmax(predicted_tr>=range_calib_number+split_val)
            
            # no points so no cigar... use previous points
            if ptr_index_start >= ptr_index_end:
                logging.warning("Skipping calibration step, due to no points in the predicted range (are you sure about the split size?): %s,%s" % (range_calib_number,range_calib_number+split_val))
                continue

            mtr = measured_tr[ptr_index_start:ptr_index_end]
            ptr = predicted_tr[ptr_index_start:ptr_index_end]

            if use_median:
                mtr_mean.append(np.median(mtr))
                ptr_mean.append(np.median(ptr))
            else:
                mtr_mean.append(sum(mtr) / len(mtr))
                ptr_mean.append(sum(ptr) / len(ptr))

        if self.verbose:
            logging.debug("Fitting the linear models between the points")
        
        if self.split_cal >= len(measured_tr):
            logging.error("There are not enough measured tr (%s) for the number of splits chosen (%s)" % (len(measured_tr),self.split_cal))
            logging.error("Choose a smaller split_cal parameter or provide more peptides for fitting the calibration curve")
            sys.exit(1)
        if len(mtr_mean) == 0:
            logging.error("The measured tr list is empty, not able to calibrate")
            sys.exit(1)
        if len(ptr_mean) == 0:
            logging.error("The predicted tr list is empty, not able to calibrate")
            sys.exit(1)

        # calculate calibration curves
        for i in range(0, len(ptr_mean)):
            if i >= len(ptr_mean) - 1:
                continue
            delta_ptr = ptr_mean[i + 1] - ptr_mean[i]
            delta_mtr = mtr_mean[i + 1] - mtr_mean[i]

            slope = delta_mtr / delta_ptr
            intercept = (-1*(ptr_mean[i]*slope))+mtr_mean[i]

            # optimized predictions using a dict to find calibration curve very
            # fast
            for v in np.arange(
                round(ptr_mean[i], self.bin_dist),
                round(ptr_mean[i + 1], self.bin_dist),
                1 / ((self.bin_dist) * self.dict_cal_divider)
            ):
                if v < calibrate_min:
                    calibrate_min = v
                if v > calibrate_max:
                    calibrate_max = v
                calibrate_dict[str(round(v, self.bin_dist))] = [slope, intercept]

        return calibrate_min, calibrate_max, calibrate_dict


    def calibrate_preds(self,
                        seqs=[],
                        mods=[],
                        identifiers=[],
                        measured_tr=[],
                        correction_factor=1.0,
                        seq_df=None,
                        use_median=True):
        """
        Find best model and calibrate.

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : list
            identifiers of the peptides; should correspond to seqs and mods
        measured_tr : list
            measured tr of the peptides; should correspond to seqs, identifiers,
            and mods
        correction_factor : float
            correction factor that needs to be applied to the supplied measured
            trs
        seq_df : object :: pd.DataFrame
            a pd.DataFrame that contains the sequences, modifications and
            observed retention times to fit a calibration curve
        use_median : boolean
            flag to indicate we need to use the median valuein a window to
            perform calibration

        Returns
        -------

        """
        if isinstance(self.model, str):
            self.model = [self.model]

        if self.verbose:
            logging.debug("Start to calibrate predictions ...")
        if self.verbose:
            logging.debug(
                "Ready to find the best model out of: %s" %
                (self.model))

        best_perf = float("inf")
        best_calibrate_min = 0.0
        best_calibrate_max = 0.0
        best_calibrate_dict = {}
        mod_calibrate_dict = {}
        mod_calibrate_min_dict = {}
        mod_calibrate_max_dict = {}
        pred_dict = {}
        mod_dict = {}
        best_models = []

        for m in self.model:
            if self.verbose:
                logging.debug("Trying out the following model: %s" % (m))
            calibrate_output = self.calibrate_preds_func(
                seqs=seqs,
                mods=mods,
                identifiers=identifiers,
                measured_tr=measured_tr,
                correction_factor=correction_factor,
                seq_df=seq_df,
                use_median=use_median,
                mod_name=m)

            self.calibrate_min, self.calibrate_max, self.calibrate_dict = calibrate_output

            if len(self.calibrate_dict.keys()) == 0:
                continue

            preds = self.make_preds(seqs=seqs,
                                    mods=mods,
                                    identifiers=identifiers,
                                    calibrate=True,
                                    seq_df=seq_df,
                                    correction_factor=correction_factor,
                                    mod_name=m)
            m_name = m.split("/")[-1]
            m_group_name = "_".join(m_name.split("_")[:-1])

            try:
                pred_dict[m_group_name][m] = preds
                mod_dict[m_group_name][m] = m
                mod_calibrate_dict[m_group_name][m] = self.calibrate_dict
                mod_calibrate_min_dict[m_group_name][m] = self.calibrate_min
                mod_calibrate_max_dict[m_group_name][m] = self.calibrate_max
            except:
                pred_dict[m_group_name] = {}
                mod_dict[m_group_name] = {}
                mod_calibrate_dict[m_group_name] = {}
                mod_calibrate_min_dict[m_group_name] = {}
                mod_calibrate_max_dict[m_group_name] = {}

                pred_dict[m_group_name][m] = preds
                mod_dict[m_group_name][m] = m
                mod_calibrate_dict[m_group_name][m] = self.calibrate_dict
                mod_calibrate_min_dict[m_group_name][m] = self.calibrate_min
                mod_calibrate_max_dict[m_group_name][m] = self.calibrate_max

        for m_name in pred_dict.keys():
            preds = [sum(a)/len(a) for a in zip(*list(pred_dict[m_name].values()))]
            if len(measured_tr) == 0:
                perf = sum(abs(seq_df["tr"] - preds))
            else:
                perf = sum(abs(measured_tr - preds))

            if self.verbose:
                logging.debug(
                    "For current model got a performance of: %s" %
                    (perf / len(preds)))

            if perf < best_perf:
                m_group_name =  "_".join(m.split("_")[:-1]).split("/")[-1]
                # TODO is deepcopy really required?

                best_calibrate_dict = copy.deepcopy(mod_calibrate_dict[m_group_name])
                best_calibrate_min = copy.deepcopy(mod_calibrate_min_dict[m_group_name])
                best_calibrate_max = copy.deepcopy(mod_calibrate_max_dict[m_group_name])

                best_model = copy.deepcopy(mod_dict[m_group_name])
                best_perf = perf

        self.calibrate_dict = best_calibrate_dict
        self.calibrate_min = best_calibrate_min
        self.calibrate_max = best_calibrate_max
        self.model = best_model

        logging.debug("Model with the best performance got selected: %s" %(best_model))


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
        result = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        return result
