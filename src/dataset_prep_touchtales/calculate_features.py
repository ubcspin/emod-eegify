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
from config_touchtale import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS, N_SAMPLES, MODE, TOUCH_FEATURES_ONLY, OTHER_FEATURES_ONLY, ALL_FEATURES_INCLUDED, SUBJECT_IDS
sys.path.remove(str(_parentdir))

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = 'cleaned_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

COLUMNS = None


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, fs=FS):
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    columns = ['window_id', 'BPM', 'flag', 'GSR']
    grouped_data = df[columns].groupby('window_id', group_keys=False).apply(np.array).to_numpy()
    # print(len(grouped_data))

    other_feat_windows_arr = []
    for window in grouped_data:
        # window = window[:mean_windowsize, :]
        if window.shape[0] < N_SAMPLES:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {N_SAMPLES}')
            continue
        elif window.shape[0] > N_SAMPLES:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {N_SAMPLES}')
            window = window[:N_SAMPLES, :]
        # print("window shape: ", np.array(window).shape, n_samples)
        other_feat_windows_arr.append(window)

    windows = np.array(other_feat_windows_arr)
    windows = windows.reshape((windows.shape[0]*windows.shape[1]), windows.shape[2])
    other_df = pd.DataFrame(windows, columns=columns)

    statistical_features = calculate_statistical_features(
        other_df.groupby('window_id'))
    statistical_features = statistical_features.reset_index().astype('float64')
    frequency_features = calc_freq_features(
        df, columns, time_interval=time_interval, time_index=time_index)
    
    frequency_features = frequency_features.drop(columns=['timedelta']).astype('float64')
    
    # print(statistical_features.columns.values)    
    # print("stat feature shape: ", statistical_features.shape, time_interval)
    # print("freq feature shape: ", frequency_features.shape)
    
    ### Calculate Stat features for touch
    columns = ['window_id', 'framesum']
    df['framesum'] = df[['T' + str(i) for i in range(1, 101)]].sum(axis=1)
    grouped_data_touch = df[columns].groupby('window_id').apply(np.array).to_numpy()

    windows = []
    for window in grouped_data_touch:
        # window = window[:mean_windowsize, :]
        if window.shape[0] < N_SAMPLES:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {N_SAMPLES}')
            continue
        elif window.shape[0] > N_SAMPLES:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {N_SAMPLES}')
            window = window[:N_SAMPLES, :]
        # print("window shape: ", np.array(window).shape, n_samples)
        windows.append(window)

    windows = np.array(windows)
    windows = windows.reshape((windows.shape[0]*windows.shape[1]), windows.shape[2])
    touch_df = pd.DataFrame(windows, columns=columns)
    
    statistical_features_touch = calculate_statistical_features(
        touch_df.groupby('window_id'))
    statistical_features_touch = statistical_features_touch.reset_index().astype('float64')

    # print(statistical_features, type(statistical_features))

    # utils.logger.info(f'Applying windows of length {time_index}')


    # ### Calculate touch framesum
    # columns = ['window_id'] + ['T' + str(i) for i in range(1, 101)]
    # grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()

    # utils.logger.info(f'Removing windows less than {window_size} ms ({N_SAMPLES})')

    remaining = len(grouped_data) - len(other_feat_windows_arr)
    print(f"{len(other_feat_windows_arr)} out of {len(grouped_data)} windows remaining. Discarded {remaining}, {np.round(remaining/len(grouped_data), 2)*100}%.")

    # framesum_total = []
    # for i, window in enumerate(other_feat_windows_arr):
    #     # Calculate framesum of touch data
    #     framesum_total.append(np.mean(np.sum(window[:, 1:], axis=1)))
    
    # framesum_df = pd.DataFrame({'window_id': np.arange(0, len(framesum_total)), 'touch framesum': framesum_total})    

    # utils.logger.info(f'Calculating features for each window')

    if MODE == TOUCH_FEATURES_ONLY:
        all_features = statistical_features_touch
    elif MODE == OTHER_FEATURES_ONLY:
        # print(statistical_features)
        # print(frequency_features)
        all_features = pd.merge_asof(statistical_features, frequency_features, on=[
            'window_id'], direction='nearest')
    elif MODE == ALL_FEATURES_INCLUDED:
         # merge touch features with other features
        statistical_features = pd.merge_asof(statistical_features, statistical_features_touch, on=[
            'window_id'], direction='nearest')

        all_features = pd.merge_asof(statistical_features, frequency_features, on=[
            'window_id'], direction='nearest')
    else: 
        return None

    
    
    # print(all_features.columns.values)

    # print("framesum shape: ", framesum_df.shape)
    # print("all features shape: ", all_features.shape)

    # print(all_features)
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
            # features = calculate_features_per_participant(merged_data[pnum])
            features = calculate_features_per_participant(merged_data[pnum], time_interval=f"{wsize}ms", window_size=wsize)


            participant_feature = {}
            participant_feature[pnum] = features

            print("feature dimensions: ", features.shape)

            if SAVE_PICKLE_FILE:
                utils.logger.info('Saving data')
                output = f"{pnum}_" + str(wsize) + 'ms_' + OUTPUT_PICKLE_NAME
                output_pickle_file_path = os.path.join(OUTPUT_DIR, output)

                utils.pickle_data(data=participant_feature,
                          file_path=output_pickle_file_path)


if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        calculate_features()