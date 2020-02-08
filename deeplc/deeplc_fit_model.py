"""
This code is used to train an XGBoost model. This code is no longer used and
substituted for a deep learning alternative.

For the library versions see the .yml file
"""

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__credits__ = ["Robbin Bouwmeester", "Ralf Gabriels", "Prof. Lennart Martens", "Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]

# Native imports
from configparser import ConfigParser
import logging

# SciPy
from scipy.stats import randint
from scipy.stats import uniform

# SKLearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# XGBoost
import xgboost as xgb

from sklearn.svm import SVR

# Data analysis imports
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import expon
from numpy import logspace
import numpy as np


def fit_lasso(X_train,
              y_train,
              X_test,
              y_test,
              nfolds=10,
              n_jobs=7):
    model = Lasso()
    params = {
        'alpha': [
            0.005,
            0.01,
            0.1,
            1.0,
            5.0,
            10.0,
            100.0,
            500.0,
            750.0,
            1000.0],
        'copy_X': [True],
        'fit_intercept': [
            True,
            False],
        'normalize': [
            True,
            False],
        'precompute': [False]}

    cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    n_iter_search = 40
    random_search = RandomizedSearchCV(model,
                                       param_distributions=params,
                                       n_iter=n_iter_search,
                                       verbose=10,
                                       scoring="neg_mean_absolute_error",
                                       n_jobs=n_jobs,
                                       cv=cv)

    random_search = random_search.fit(X_train, y_train)

    xgb_model = random_search.best_estimator_

    test_preds = xgb_model.predict(X_test)

    train_preds = xgb_model.predict(X_train)

    model = Lasso(**random_search.best_params_)
    train_cross_preds = cross_val_predict(model,
                                          X_train,
                                          y_train,
                                          cv=cv)

    random_search.feats = X_train.columns

    return train_preds, train_cross_preds, test_preds


def fit_svr(X_train,
            y_train,
            X_test,
            y_test,
            nfolds=10):

    model = SVR()
    params = {
        'alpha': [
            0.0005,
            0.001,
            0.005,
            0.01,
            0.1,
            1.0,
            5.0,
            10.0,
            100.0,
            500.0,
            750.0,
            1000.0],
        'copy_X': [True],
        'fit_intercept': [
            True,
            False],
        'normalize': [
            True,
            False],
        'precompute': [
            True,
            False]}

    cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    n_iter_search = 50
    random_search = RandomizedSearchCV(model,
                                       param_distributions=params,
                                       n_iter=n_iter_search,
                                       verbose=1,
                                       scoring="neg_mean_absolute_error",
                                       n_jobs=32,
                                       cv=cv)

    random_search = random_search.fit(X_train, y_train)

    xgb_model = random_search.best_estimator_

    test_preds = xgb_model.predict(X_test)

    train_preds = xgb_model.predict(X_train)

    model = SVR(**random_search.best_params_)
    train_cross_preds = cross_val_predict(model,
                                          X_train,
                                          y_train,
                                          cv=cv)

    random_search.feats = X_train.columns

    return train_preds, train_cross_preds, test_preds


def fit_xgb_leaf(X,
                 y,
                 X_test,
                 y_test,
                 cv=10,
                 param_dist={}):

    dtrain = xgb.DMatrix(X,
                         label=y)
    dtest = xgb.DMatrix(X_test,
                        label=y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    bst = xgb.train(param_dist,
                    dtrain,
                    num_boost_round=2000,
                    evals=watchlist,
                    early_stopping_rounds=10,
                    verbose_eval=False)

    pred = bst.predict(dtest)

    return bst.predict(dtrain), bst.predict(dtest), bst.best_score, bst


def fit_xgb(X_train, y_train, X_test, y_test, config_file="config.ini"):
    """
    Extract all features we can extract; without parallelization; use if you want to run feature extraction
    with a single core

    Parameters
    ----------
    X_train : pd.DataFrame
        feature matrix
    y_train : pd.DataFrame/Series
        objective values for training
    X_test : pd.DataFrame
        feature matrix for testing/evaluating
    y_test : pd.DataFrame/Series
        objective values for testing/evaluating
    config_file : str
        location of the configuration file that contains the hyperparemeter spaces

    Returns
    -------
    list
        predictions for the train set
    list
        cross-validation predictions (hyperparameters still determined on the training set; not the model parameters)
    list
        test predictions
    sklearn.model_selection.RandomizedSearchCV
        object containing the model and training settings
    """

    cparser = ConfigParser()
    cparser.read(config_file)

    # get hyperparameter space to sample from
    n_estimators = eval(cparser.get("fitXGB", "n_estimators"))
    max_depth = eval(cparser.get("fitXGB", "max_depth"))
    learning_rate = eval(cparser.get("fitXGB", "learning_rate"))
    gamma = eval(cparser.get("fitXGB", "gamma"))
    reg_alpha = eval(cparser.get("fitXGB", "reg_alpha"))
    reg_lambda = eval(cparser.get("fitXGB", "reg_lambda"))

    random_state = cparser.getint("fitXGB", "random_state")
    nfolds = cparser.getint("fitXGB", "nfolds")
    n_iter_search = cparser.getint("fitXGB", "n_iter_search")
    verbose = cparser.getint("fitXGB", "verbose")
    n_jobs = cparser.getint("fitXGB", "n_jobs")
    eval_metric = cparser.get("fitXGB", "eval_metric").strip('"')

    model = xgb.XGBRegressor()

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_jobs': [n_jobs]
    }

    cv = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=n_iter_search,
        verbose=verbose,
        scoring=eval_metric,
        cv=cv,
        random_state=random_state)

    random_search = random_search.fit(X_train, y_train)

    xgb_model = random_search.best_estimator_

    train_preds = xgb_model.predict(X_train)

    # train using the best hyperparameters and make cv preds
    model = xgb.XGBRegressor(**random_search.best_params_)

    if verbose > 0:
        logging.debug("Predicting tR with CV now...")
    train_cross_preds = cross_val_predict(model, X_train, y_train, cv=cv)

    random_search.feats = X_train.columns

    test_preds = xgb_model.predict(X_test)

    if verbose > 0:
        logging.debug("=====")
        logging.debug(random_search.best_params_)
        logging.debug(random_search.best_score_)
        logging.debug("=====")

    return train_preds, train_cross_preds, test_preds, random_search
