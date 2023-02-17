"""
Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
"""
# Import modules for feature engineering and modelling

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from hiclass2 import LocalClassifierPerParentNode

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

from config import CV_SPLITS
import utils
import numpy as np 
import pandas as pd




class EstimatorSelectionHelper:

    def __init__(self, models, params, root_classes=None):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.root_classes = root_classes
        self.random_searches = {}

    def cv(self, n_splits=CV_SPLITS, random_state=None, shuffle=True):
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


    def fit(self, X, y, cv=None, n_jobs=-1, verbose=1, scoring=None, refit=False):
        if cv is None:
            cv = self.cv()
        for key in self.keys:
            utils.logger.info("Running RandomizedSearchCV for %s." % key)

            estimator = LocalClassifierPerParentNode(self.models[key], n_jobs=n_jobs, root_classes = self.root_classes, replace_classifiers=False)

            rs = RandomizedSearchCV(estimator, self.params[key], cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True, refit=False).fit(X, y)
            self.random_searches[key] = rs
            print(rs)

    def score_summary(self, X, sort_by='mean_score'):
        all_dfs = []
        for k in self.random_searches:
            df = pd.DataFrame(self.random_searches[k].cv_results_)
            df['estimator'] = k
            all_dfs.append(df)
            
        all_dfs = pd.concat(all_dfs)
        return all_dfs
