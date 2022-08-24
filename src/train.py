import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from estimator_helper import EstimatorSelectionHelper

from models import MODELS, PARAMS
from utils import pickle_data, load_pickle
from config import TIME_INDEX, TIME_INTERVAL

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'training_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'results.pk'


KEY = 'ewm_a6'
LABEL_TYPES = ['pos', 'angle', 'acc']

LE = LabelEncoder()


def fit_helper(X, y, models=MODELS, params=PARAMS, n_jobs=-1, scoring='f1_macro'):
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X, y, scoring=scoring, n_jobs=n_jobs)
    scores = helper.score_summary(sort_by='max_score')
    return helper, scores


def train(training_data, key=KEY):

    subject_results = []

    for subject in training_data:
        res = {}

        X = subject['features'].loc[:,
                                    subject['features'].columns.str.contains(KEY)]

        res['X'] = X
        res['pnum'] = subject['pnum']

        for label_type in LABEL_TYPES:
            y = subject[label_type]
            if label_type in 'cw':
                y = LE.fit_transform(
                    subject['calibrated_values'].astype('str'))
            res['y_' + label_type] = y

            helper, scores = fit_helper(X, y)
            res['helper_' + label_type] = helper
            res['scores_' + label_type] = scores

            del helper, del scores

        subject_results.append(res)

    return subject_results


if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        training_data = utils.load_pickle(file_path=input_pickle_file_path)

    training_results = train(training_data)

    if OUTPUT_PICKLE_FILe:
        output_pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=training_results,
                          file_path=output_pickle_file_path)
