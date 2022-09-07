import os
import utils

import numpy as np
import pandas as pd

from calculate_de_features import calculate_de_features
from tqdm import tqdm

from config import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'merged_data.pk'

SAVE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_FILE_NAME = 'featurized_data.npy'

COLUMNS = None


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE):
    
    if columns is None:
        columns = ['window_id'] + [ 'E' + str(i+1) for i in range(64)] 

    df[columns[1:]] = (df[columns[1:]] - df[columns[1:]].min()) / (df[columns[1:]].max() - df[columns[1:]].min()) # min/max normalization
    
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    utils.logger.info(f'Applying windows of length {time_index}')
    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()

    utils.logger.info(f'Removing windows less than {window_size} ms')
    windows = []
    for window in grouped_data:
        if window.shape[0] < window_size:
            continue
        windows.append(np.expand_dims(window[:,1:], axis=0) ) # drop window_id

    all_features = np.vstack(windows) # (N, window_size, 64)
    return all_features

def calculate_features(merged_data: dict):

    for pnum in tqdm(merged_data.keys()):
        utils.logger.info(f'Calculating features for {pnum}')
        features = calculate_features_per_participant(merged_data[pnum])

        participant_feature = features

        if SAVE_FILE:
            output_file_path = os.path.join(OUTPUT_DIR, f"{pnum}_" + OUTPUT_FILE_NAME)
            utils.save_numpy(data=participant_feature,
                          file_path=output_file_path)



if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        merged_data = utils.load_pickle(
            pickled_file_path=input_pickle_file_path)

    calculate_features(merged_data)