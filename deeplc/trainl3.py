"""
Robbin Bouwmeester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
This code is used to train retention time predictors and store
predictions from a CV procedure for further analysis.

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from numpy import arange
from scipy.stats import pearsonr

from operator import itemgetter
from numpy import median
from collections import Counter
    
def train_en(X,y,n_jobs=16,cv=None):
    """
    Function that trains Layer 3 of CALLC (elastic net)
    
    Parameters
    ----------
    X : pd.DataFrame
        dataframe with molecular descriptors
    y : pd.Series
        vector with observed retention times
    n_jobs : int
        number of jobs to spawn
    cv : sklearn.model_selection.KFold
        cv object
    
    Returns
    -------
    sklearn.linear_model.ElasticNet
        elastic net model trained in Layer 3
    list
        list with predictions
    list
        list with features used to train Layer 3
    """
    preds = []

    model = ElasticNet()
    crossv_mod = clone(model)
    ret_mod = clone(model)

    set_reg = [0.01,1.0,10.0,100.0,1000.0,10000.0,10000.0,100000.0,1000000.0,1000000000,1000000]
    set_reg.extend([x/2 for x in set_reg])
    set_reg.extend([x/3 for x in set_reg])
    
    params = {
       'alpha': set_reg,
       'l1_ratio' : [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
       'copy_X':[True],
       'normalize' : [False],
       'positive' : [True],
       'fit_intercept'  : [True,False]
    }

    grid = GridSearchCV(model, params,cv=cv,scoring='neg_mean_absolute_error',verbose=0,n_jobs=n_jobs,refit=True)
    grid.fit(X,y)
    
    cv_pred = cv
    crossv_mod.set_params(**grid.best_params_)
    preds = cross_val_predict(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0)

    ret_mod.set_params(**grid.best_params_)
    ret_mod.fit(X,y)

    coef_indexes = [i for i,coef in enumerate(ret_mod.coef_) if coef > 0.0]
   
    return ret_mod