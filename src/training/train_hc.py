import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

import os
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer

from estimator_helper_hc import EstimatorSelectionHelper

from models import MODELS, PARAMS
from config import EXP_PARAMS

from hiclass2 import metrics



INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = '_featurized_data.pk'
INPUT_LABEL_NAME = '_labels.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'RESULTS'
OUTPUT_PICKLE_NAME = 'results.pk'

SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']
FEATURE_TYPES = ['alpha', 'beta', 'delta', 'theta', 'gamma']
LABEL_TYPES = [
    'angle'
]



def fit_helper(X, y, models=MODELS, params=PARAMS, n_jobs=-1, scoring={ "f1": make_scorer(metrics.f1), "prec" : make_scorer(metrics.precision), "recall": make_scorer(metrics.recall)}, cw_classes=None):
    helper = EstimatorSelectionHelper(models, params, cw_classes)
    helper.fit(X, y, scoring=scoring, n_jobs=n_jobs)
    try:
        scores = helper.score_summary(X, sort_by='max_f1')
        return helper, scores
    except ValueError:
        return -1, -1


def train(feature_dict, label_dict, label_types=LABEL_TYPES, feature_types=FEATURE_TYPES, pnum='p01'):

    # read the features back into a numpy array
    bands = feature_dict[feature_types].to_numpy()
    features = np.empty((bands.shape[0], bands.shape[1], 64, 64))
    for band in range(len(feature_types)):
        band_arr = bands[:,band]
        for window in range(band_arr.shape[0]):
            band_im = np.array(band_arr[window])
            features[window, band, :, :] = band_im
    X = features.astype(np.float32)

    res = {}

    res['pnum'] = pnum

    Y = label_dict

    LE = LabelEncoder() # transform string to class values
    LE.fit(Y['cw_mode'])

    y_cw_str = Y['cw_mode']
    y_cw = LE.transform(y_cw_str)

    for label_type in label_types: # train every label type
        utils.logger.info(f'Training label type {label_type}')

        y = np.array(list(zip(y_cw, Y[label_type])))
        y_str = np.array(list(zip(y_cw_str, Y[label_type] )))

        res['y_hc_' + label_type] = y_str
        helper, scores = fit_helper(X, y, cw_classes=len(LE.classes_))
        res['scores_hc_' + label_type] = scores
        del helper
        del scores

    return res


if __name__ == '__main__':

    for i in range(0, 1):
        for window_size in EXP_PARAMS['WINDOW_SIZE']:
            if INPUT_PICKLE_FILE:
                participant_results = {}
                for subject_id in tqdm(SUBJECT_IDS):
                    utils.logger.info(f'Training participant {subject_id}')

                    training_data_filename = subject_id + "_" + str(window_size) + 'ms' + INPUT_PICKLE_NAME
                    input_pickle_file_path = os.path.join(INPUT_DIR, training_data_filename)
                    input_label_file_path = os.path.join(INPUT_DIR, str(window_size) + 'ms' + INPUT_LABEL_NAME)

                    features = utils.load_pickle(pickled_file_path=input_pickle_file_path)
                    labels = utils.load_pickle(pickled_file_path=input_label_file_path)


                    utils.logger.info(f'Window size {window_size} - Iteration {i}')
                    training_results = train(features[subject_id], labels[subject_id], pnum=subject_id)


                    if SAVE_PICKLE_FILE:
                        iter_ = subject_id + "_" + str(i) + '_' + str(window_size) + 'ms_hc_cw_' + OUTPUT_PICKLE_NAME
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        output_pickle_file_path = os.path.join(OUTPUT_DIR, iter_)
                        utils.pickle_data(data=training_results, file_path=output_pickle_file_path)


    sys.path.remove(str(_parentdir))