from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

MODELS = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier(),
    'SVC': SVC()  # takes a long time to train with linear kernels
}

PARAMS = {
    'ExtraTreesClassifier': {'n_estimators': [16, 32]},
    'RandomForestClassifier': {'n_estimators': [16, 32]},
    'AdaBoostClassifier':  {'n_estimators': [16, 32]},
    'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
    'XGBClassifier': {'max_depth': (4, 6, 8), 'min_child_weight': (1, 5, 10)},
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ],
}