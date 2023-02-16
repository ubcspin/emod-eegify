"""
Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
"""
# Import modules for feature engineering and modelling

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
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

            rs = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            self.random_searches[key] = rs
            print(rs)

    def score_summary(self, X, sort_by='mean_score'):
        def row(key, f1, prec, recall):
            d = {
                'estimator': key,
                'min_f1': min(f1),
                'max_f1': max(f1),
                'mean_f1': np.mean(f1),
                'std_f1': np.std(f1),

                'min_precision': min(prec),
                'max_precision': max(prec),
                'mean_precision': np.mean(prec),
                'std_precision': np.std(prec),

                'min_recall': min(recall),
                'max_recall': max(recall),
                'mean_recall': np.mean(recall),
                'std_recall': np.std(recall)
            }
            return pd.Series({**d})

        rows = []
        for k in self.random_searches:

            f1 = self.random_searches[k]['test_f1']
            prec = self.random_searches[k]['test_prec']
            recall = self.random_searches[k]['test_recall']

            rows.append((row(k, f1, prec, recall)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_f1', 'mean_f1', 'max_f1', 'std_f1', 'min_precision', 'mean_precision', 'max_precision', 'std_precision', 'min_recall', 'mean_recall', 'max_recall', 'std_recall']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
