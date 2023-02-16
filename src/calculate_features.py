import os
import utils

import numpy as np
import pandas as pd

from calculate_de_features import calculate_de_features
from tqdm import tqdm

from config import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS


INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'merged_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

COLUMNS = None


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, fs=FS):
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    if columns is None:
        columns = ['window_id'] + [ 'E' + str(i+1) for i in range(64)] 

    utils.logger.info(f'Applying windows of length {time_index}')
    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()

    n_samples = window_size / 1e3 * fs
    n_samples = int(n_samples)
    utils.logger.info(f'Removing windows less than {window_size} ms ({n_samples})')


    windows = []
    for window in grouped_data:
        if window.shape[0] < n_samples:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {n_samples}')
            continue
        elif window.shape[0] > n_samples:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {n_samples}')
            window = window[:n_samples, :]
        windows.append(window)
    

    utils.logger.info(f'Calculating features for each window')
    de_features = calculate_de_features(
        windows)

    all_features = de_features

    return all_features

def calculate_features(merged_data: dict):

    for pnum in tqdm(merged_data.keys()):
        utils.logger.info(f'Calculating features for {pnum}')

        window_sizes = EXP_PARAMS["WINDOW_SIZE"]

        for wsize in window_sizes:
            utils.logger.info(f'Calculating labels for window size: {wsize} ms')
            features = calculate_features_per_participant(merged_data[pnum], time_interval=f"{wsize}ms", window_size=wsize)


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
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        merged_data = utils.load_pickle(
            pickled_file_path=input_pickle_file_path)
            

    calculate_features(merged_data)