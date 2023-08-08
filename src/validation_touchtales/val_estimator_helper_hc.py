"""
Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
"""
# Import modules for feature engineering and modelling

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from hiclass2 import LocalClassifierPerParentNode


from config import CV_SPLITS
import utils
import numpy as np 
import pandas as pd

import mlflow



class EstimatorSelectionHelper:

    def __init__(self, models, params, root_classes=None, key=None):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.root_classes = root_classes
        self.random_searches = {}
        self.key = key

    def cv(self, n_splits=CV_SPLITS, random_state=None, shuffle=True):
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


    def fit(self, X, y, X_test, y_test, cv=None, n_jobs=-1, verbose=1, scoring=None, refit=False):
        if cv is None:
            cv = self.cv()
        for key in self.keys:
            utils.logger.info("Running RandomizedSearchCV for %s." % key)

            estimator = LocalClassifierPerParentNode(self.models[key], n_jobs=n_jobs, root_classes = self.root_classes, replace_classifiers=False)

            with mlflow.start_run():
                mlflow.log_param("key", self.key)
                estimator.fit(X, y)
            y_hat = estimator.predict(X_test)
            

            rs = {}
            for fname, f in scoring.items():
                print(fname, f(y_test, y_hat))
                rs["test_" + fname] = f(y_test, y_hat)
            #rs = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            self.random_searches[key] = rs
            print(rs)

    def score_summary(self, X, sort_by='mean_score'):
        def row(key, f1, prec, recall):
            d = {
                'estimator': key,
                'f1': f1,
                'precision': prec,
                'recall': recall
            }
            return pd.Series({**d})

        rows = []
        for k in self.random_searches:

            f1 = self.random_searches[k]['test_f1']
            prec = self.random_searches[k]['test_prec']
            recall = self.random_searches[k]['test_recall']

            rows.append((row(k, f1, prec, recall)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'f1', 'precision', 'recall']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
