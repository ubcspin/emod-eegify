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

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config_touchtale import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS
sys.path.remove(str(_parentdir))

from hiclass2 import metrics



INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = '_featurized_data.pk'
INPUT_LABEL_NAME = '_labels.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'RESULTS'
OUTPUT_PICKLE_NAME = 'results.pk'

SUBJECT_IDS = [x[:3] if 'p0' in x and 'cleaned' in x else '' for x in os.listdir('COMBINED_DATA_TOUCHTALE/')]
SUBJECT_IDS = list(filter(lambda a: a != '', SUBJECT_IDS))
print(SUBJECT_IDS)

# FEATURE_TYPES = ['window_id', 'max_window_id', 'max_BPM', 'max_flag', 'max_GSR', 'max_feeltrace', 'max_calibrated_values', \
#                 'min_window_id', 'min_BPM', 'min_flag', 'min_GSR', 'min_feeltrace', 'min_calibrated_values', 'auc_window_id', \
#                 'auc_BPM', 'auc_flag', 'auc_GSR', 'auc_feeltrace', 'auc_calibrated_values', 'var_window_id', 'var_BPM', \
#                 'var_flag', 'var_GSR', 'var_feeltrace', 'var_calibrated_values', 'sum_diffs_window_id', 'sum_diffs_BPM', \
#                 'sum_diffs_flag', 'sum_diffs_GSR', 'sum_diffs_feeltrace', 'sum_diffs_calibrated_values', 'touch framesum', \
#                 'timedelta', 'freqs_max_calibrated_values', 'freqs_max_amp_calibrated_values', 'freqs_min_amp_calibrated_values', \
#                 'freqs_var_calibrated_values', 'freqs_mean_calibrated_values', 'freqs_peak_count_calibrated_values', \
#                 'freqs_max_feeltrace', 'freqs_max_amp_feeltrace', 'freqs_min_amp_feeltrace', 'freqs_var_feeltrace', \
#                 'freqs_mean_feeltrace', 'freqs_peak_count_feeltrace', 'freqs_max_GSR', 'freqs_max_amp_GSR', 'freqs_min_amp_GSR', \
#                 'freqs_var_GSR', 'freqs_mean_GSR', 'freqs_peak_count_GSR', 'freqs_max_flag', 'freqs_max_amp_flag', \
#                 'freqs_min_amp_flag', 'freqs_var_flag', 'freqs_mean_flag', 'freqs_peak_count_flag', 'freqs_max_BPM', \
#                 'freqs_max_amp_BPM', 'freqs_min_amp_BPM', 'freqs_var_BPM', 'freqs_mean_BPM', 'freqs_peak_count_BPM', \
#                 'freqs_max_window_id', 'freqs_max_amp_window_id', 'freqs_min_amp_window_id', 'freqs_var_window_id', \
#                 'freqs_mean_window_id', 'freqs_peak_count_window_id']
LABEL_TYPES = ['pos', 'angle', 'acc', 'cw']



def fit_helper(X, y, models=MODELS, params=PARAMS, n_jobs=-1, scoring={ "f1": make_scorer(metrics.f1), "prec" : make_scorer(metrics.precision), "recall": make_scorer(metrics.recall)}, cw_classes=None):
    helper = EstimatorSelectionHelper(models, params, cw_classes)
    helper.fit(X, y, scoring=scoring, n_jobs=n_jobs)
    try:
        scores = helper.score_summary(X, sort_by='max_f1')
        return helper, scores
    except ValueError:
        return -1, -1


def train(feature_dict, label_dict, label_types=LABEL_TYPES, pnum='p01'):

    # read the features back into a numpy array
    # bands = feature_dict[feature_types].to_numpy()
    # features = np.empty((bands.shape[0], bands.shape[1], 64, 64))
    # for band in range(len(feature_types)):
    #     band_arr = bands[:,band]
    #     for window in range(band_arr.shape[0]):
    #         band_im = np.array(band_arr[window])
    #         features[window, band, :, :] = band_im
    features = feature_dict.drop('timedelta', axis=1).to_numpy()
    X = features.astype(np.float32)

    res = {}
   
    res['pnum'] = pnum

    Y = label_dict.loc[label_dict['window_id'].isin(feature_dict['window_id'].unique())]

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

                    # print("features", features[subject_id])
                    # print("labels", labels[subject_id])
                    # break

                    utils.logger.info(f'Window size {window_size} - Iteration {i}')
                    training_results = train(features[subject_id], labels[subject_id], pnum=subject_id)


                    if SAVE_PICKLE_FILE:
                        iter_ = subject_id + "_" + str(i) + '_' + str(window_size) + 'ms_hc_cw_' + OUTPUT_PICKLE_NAME
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        output_pickle_file_path = os.path.join(OUTPUT_DIR, iter_)
                        utils.pickle_data(data=training_results, file_path=output_pickle_file_path)


    # sys.path.remove(str(_parentdir))