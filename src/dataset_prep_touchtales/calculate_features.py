import os

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from calculate_de_features import calculate_de_features
from calculate_freq_features import calc_freq_features
from calculate_stat_features import calculate_statistical_features

from tqdm import tqdm

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config_touchtale import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS
sys.path.remove(str(_parentdir))

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = 'cleaned_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

COLUMNS = None

SUBJECT_IDS = [x[:3] if 'p0' in x and 'cleaned' in x else '' for x in os.listdir('COMBINED_DATA_TOUCHTALE/')]
SUBJECT_IDS = list(filter(lambda a: a != '', SUBJECT_IDS))
print(SUBJECT_IDS)


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, fs=FS):
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    columns = ['window_id', 'BPM', 'flag', 'GSR']
    grouped_data = df[columns].groupby('window_id')

    statistical_features = calculate_statistical_features(
        grouped_data[columns])
    statistical_features = statistical_features.reset_index()
    frequency_features = calc_freq_features(
        df, columns, time_interval=time_interval, time_index=time_index)
    
    print("stat feature shape: ", statistical_features.shape, time_interval)
    print("freq feature shape: ", frequency_features.shape)
    
    columns = ['window_id'] + ['T' + str(i) for i in range(1, 101)]

    utils.logger.info(f'Applying windows of length {time_index}')
    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()

    n_samples = window_size / 1e3 * fs
    n_samples = int(n_samples)
    utils.logger.info(f'Removing windows less than {window_size} ms ({n_samples})')

    print("group data shape: ", np.array(grouped_data).shape)
    windows = []
    for window in grouped_data:
        # print("window shape: ", np.array(window).shape)
        if window.shape[0] < n_samples:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {n_samples}')
            continue
        elif window.shape[0] > n_samples:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {n_samples}')
            window = window[:n_samples, :]
        # print("window shape: ", np.array(window).shape)
        windows.append(window)

    # print("windows shape: ", np.array(windows).shape, window_size, fs)
    print("window id unique: ", df['window_id'].unique())

    framesum_total = []
    for i, window in enumerate(windows):
        # if i > 0:
        #     channel_means = np.mean(windows[i-1][:,1:], axis=0, keepdims=True) # calc the mean for each channel from the previous window
        #     windows[i][:,1:] = windows[i][:,1:] - channel_means # subtract the mean of the previous window from from the current window

        # Calculate framesum of touch data
        framesum_total.append(np.mean(np.sum(window[:, 1:], axis=1)))
    
    framesum_df = pd.DataFrame({'window_id': np.arange(0, len(framesum_total)), 'touch framesum': framesum_total})

    utils.logger.info(f'Calculating features for each window')
    # de_features = calculate_de_features(
        # windows)

    statistical_features = pd.merge_asof(statistical_features, framesum_df, on=[
    'window_id'], direction='nearest')

    all_features = pd.merge_asof(statistical_features, frequency_features, on=[
        'window_id'], direction='nearest')

    print("framesum shape: ", framesum_df.shape)
    print("all features shape: ", all_features.shape)

    print(list(all_features.columns.values))

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
            features = calculate_features_per_participant(merged_data[pnum])
            # features = calculate_features_per_participant(merged_data[pnum], time_interval=f"{wsize}ms", window_size=wsize)


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