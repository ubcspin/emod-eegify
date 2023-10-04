# """
# Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
# """
# # Import modules for feature engineering and modelling
# from sklearn.model_selection import KFold, GridSearchCV
# from hiclass2 import LocalClassifierPerParentNode

# from config import CV_SPLITS
# import utils
# import numpy as np 
# import pandas as pd




# class EstimatorSelectionHelper:

#     def __init__(self, models, params, root_classes=None):
#         if not set(models.keys()).issubset(set(params.keys())):
#             missing_params = list(set(models.keys()) - set(params.keys()))
#             raise ValueError(
#                 "Some estimators are missing parameters: %s" % missing_params)
#         self.models = models
#         self.params = params
#         self.keys = models.keys()
#         self.root_classes = root_classes
#         self.random_searches = {}

#     def cv(self, n_splits=CV_SPLITS, random_state=None, shuffle=True):
#         return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


#     def fit(self, X, y, cv=None, n_jobs=-1, verbose=1, scoring=None, refit=False):
#         if cv is None:
#             cv = self.cv()
#         for key in self.keys:
#             utils.logger.info("Running GridSearchCV for %s." % key)

#             estimator = LocalClassifierPerParentNode(self.models[key], n_jobs=n_jobs, root_classes = self.root_classes, replace_classifiers=False)

#             rs = GridSearchCV(estimator, self.params[key], cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True, refit=False).fit(X, y)
#             self.random_searches[key] = rs
#             print(rs)

#     def score_summary(self, X, sort_by='mean_score'):
#         all_dfs = []
#         for k in self.random_searches:
#             df = pd.DataFrame(self.random_searches[k].cv_results_)
#             df['estimator'] = k
#             all_dfs.append(df)
            
#         all_dfs = pd.concat(all_dfs)
#         return all_dfs


"""
Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
"""
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


CV_SPLITS = 3


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.rfes = {}

    def cv(self, n_splits=CV_SPLITS, random_state=0, shuffle=True):
        return StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    def rfe(self, clf, cv=None, step=1, scoring='f1_macro'):
        if cv is None:
            cv = self.cv()
        return RFECV(estimator=clf, step=step, cv=cv, scoring=scoring)

    def pipeline(self, rfecv, gs_cv):
        return Pipeline([('feature_sele', rfecv), ('clf_cv', gs_cv)])

    def fit(self, X, y, cv=None, n_jobs=3, verbose=1, scoring=None, refit=False):
        if cv is None:
            cv = self.cv()
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            rfecv = self.rfe(model)
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            pipe = self.pipeline(rfecv, gs)
            pipe.fit(X, y)
            self.grid_searches[key] = gs
            self.rfes[key] = rfecv

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            print("120")

            print({**params, **d})
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = []

            keys = self.grid_searches[k].cv_results_.keys()
            print("keys: ", keys)
            for i in range(self.grid_searches[k].cv.get_n_splits()):
                key = "split{}_test_f1".format(i)
                if key in keys:
                    r = self.grid_searches[k].cv_results_[key]
                    scores.append(r.reshape(len(params), 1))
                else:
                    print(f"key: {key}, keys: {keys}")

            all_scores = np.hstack(scores)
            try:
                print(f"all score {all_scores}, params {params}, type: {type(all_scores)}/{type(params)}")
                
                for p, s in zip(params, all_scores):
                    print(p, s)
                    rows.append((row(k, s, p)))
            except Exception as e:
                print(f"147 all score {all_scores}, params {params}")
                print("error! ", e)

        print("151", sort_by)

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        print("152")

        columns = ['estimator', 'min_score',
                   'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        print("158, ", columns)

        return df[columns]