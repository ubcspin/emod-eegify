"""
Modified from: https://gist.github.com/DimaK415/428bbeb0e79551f780bb990e7c26f813
"""
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


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
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv.get_n_splits()):
                key = "split_{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score',
                   'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]