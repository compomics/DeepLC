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
from configparser import ConfigParser

#SciPy
from scipy.stats import randint
from scipy.stats import uniform

#SKLearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# XGBoost
import xgboost as xgb

def fit_xgb(X_train,y_train,X_test,y_test,config_file="config.ini"):
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
    n_estimators = eval(cparser.get("fitXGB","n_estimators"))
    max_depth  = eval(cparser.get("fitXGB","max_depth"))
    learning_rate  = eval(cparser.get("fitXGB","learning_rate"))
    gamma  = eval(cparser.get("fitXGB","gamma"))
    reg_alpha  = eval(cparser.get("fitXGB","reg_alpha"))
    reg_lambda  = eval(cparser.get("fitXGB","reg_lambda"))

    random_state  = cparser.getint("fitXGB","random_state")
    nfolds  = cparser.getint("fitXGB","nfolds")
    n_iter_search  = cparser.getint("fitXGB","n_iter_search")
    verbose  = cparser.getint("fitXGB","verbose")
    n_jobs  = cparser.getint("fitXGB","n_jobs")
    eval_metric  = cparser.get("fitXGB","eval_metric").strip('"')

    model = xgb.XGBRegressor()

    params = {
        'n_estimators' : n_estimators,
        'max_depth' : max_depth,
        'learning_rate' : learning_rate,
        'gamma' : gamma,
        'reg_alpha' : reg_alpha,
        'reg_lambda' : reg_lambda,
        'n_jobs' : [n_jobs]
    }

    cv = KFold(n_splits=nfolds, shuffle=True,random_state=random_state)
    
    random_search = RandomizedSearchCV(model, param_distributions=params,
                                       n_iter=n_iter_search,verbose=verbose,scoring=eval_metric,
                                       cv=cv,random_state=random_state)

    random_search = random_search.fit(X_train, y_train)

    xgb_model = random_search.best_estimator_
    
    train_preds = xgb_model.predict(X_train)
    
    # train using the best hyperparameters and make cv preds
    model = xgb.XGBRegressor(**random_search.best_params_)

    if verbose > 0: print("Going to perform CV predictions now...")
    train_cross_preds = cross_val_predict(model,X_train,y_train,cv=cv)

    random_search.feats = X_train.columns
    
    test_preds = xgb_model.predict(X_test)
    
    if verbose > 0:
        print("=====")
        print(random_search.best_params_)
        print(random_search.best_score_)
        print("=====")
    
    return train_preds,train_cross_preds,test_preds,random_search
