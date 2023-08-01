import os

import numpy as np
import pandas as pd

from calculate_de_features import calculate_de_features
from calculate_freq_features import calc_freq_features
from calculate_stat_features import calculate_statistical_features

from tqdm import tqdm

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS
sys.path.remove(str(_parentdir))

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = 'cleaned_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

COLUMNS = None

SUBJECT_IDS = [x[:3] if 'p0' in x else '' for x in os.listdir('COMBINED_DATA_TOUCHTALE/')]
SUBJECT_IDS.remove('')


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, fs=FS):
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    columns = ['BPM', 'flag', 'GSR', 'feeltrace', 'calibrated_values']
    statistical_features = calculate_statistical_features(
        grouped_data[columns])
    statistical_features = statistical_features.reset_index()
    frequency_features = calc_freq_features(
        df, columns, time_interval=time_interval, time_index=time_index)

    if columns is None:
        columns = ['window_id'] + ['T' + str(i) for i in range(1, 101)]

    utils.logger.info(f'Applying windows of length {time_index}')
    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()

    n_samples = window_size / 1e3 * fs
    n_samples = int(n_samples)
    utils.logger.info(f'Removing windows less than {window_size} ms ({n_samples})')

    print("group data shape: ", np.array(grouped_data).shape)
    windows = []
    for window in grouped_data:
        print("window shape: ", np.array(window).shape)
        if window.shape[0] < n_samples:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {n_samples}')
            continue
        elif window.shape[0] > n_samples:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {n_samples}')
            window = window[:n_samples, :]
        print("window shape: ", np.array(window).shape)
        windows.append(window)

    print(np.array(windows).shape)

    for i, window in enumerate(windows):
        if i > 0:
            channel_means = np.mean(windows[i-1][:,1:], axis=0, keepdims=True) # calc the mean for each channel from the previous window
            windows[i][:,1:] = windows[i][:,1:] - channel_means # subtract the mean of the previous window from from the current window

    utils.logger.info(f'Calculating features for each window')
    de_features = calculate_de_features(
        windows)

    all_features = de_features

    return all_features

def calculate_features():
    for pnum in tqdm(SUBJECT_IDS):
        input_pickle_file_path = os.path.join(INPUT_DIR, f"{pnum}_" + INPUT_PICKLE_NAME)

        try:
            merged_data = utils.load_pickle(
                pickled_file_path=input_pickle_file_path)
        except: 
            merged_data = utils.load_pickle(
                pickled_file_path=os.getcwd()+'/'+input_pickle_file_path)

        utils.logger.info(f'Calculating features for {pnum}')

        window_sizes = EXP_PARAMS["WINDOW_SIZE"]

        for wsize in window_sizes:
            utils.logger.info(f'Calculating labels for window size: {wsize} ms')
            features = calculate_features_per_participant(merged_data[pnum], time_interval=f"{wsize}ms", window_size=wsize, fs=50)


            participant_feature = {}
            participant_feature[pnum] = features

            if SAVE_PICKLE_FILE:
                utils.logger.info('Saving data')
                output = f"{pnum}_" + str(wsize) + 'ms_' + OUTPUT_PICKLE_NAME
                output_pickle_file_path = os.path.join(OUTPUT_DIR, output)

                utils.pickle_data(data=participant_feature,
                          file_path=output_pickle_file_path)



if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        calculate_features()