import os
import utils

import pandas as pd

from calculate_freq_features import calc_freq_features

from config import TIME_INDEX, TIME_INTERVAL

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'merged_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

COLUMNS = None


def calculate_features_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS):
    df = df.assign(window_id=df.groupby(pd.Grouper(
        key=time_index, freq=time_interval)).ngroup())

    if columns is None:
        columns = list(map(lambda x: 'a' + str(x), range(7)))

    grouped_data = df.groupby('window_id')

    keystroke_features = list(map(lambda g: calculate_keystroke_features(
        g[1], columns=columns), grouped_data))
    keystroke_features = pd.concat(keystroke_features)
    keystroke_features_grouped = keystroke_features.groupby('window_id').mean()
    keystroke_features_grouped = keystroke_features_grouped.reset_index()

    statistical_features = calculate_statistical_features(
        grouped_data[columns])
    statistical_features = statistical_features.reset_index()
    frequency_features = calc_freq_features(
        df, columns, time_interval=time_interval, time_index=time_index)

    statistical_features = pd.merge_asof(statistical_features, keystroke_features_grouped, on=[
        'window_id'], direction='nearest')

    all_features = pd.merge_asof(statistical_features, frequency_features, on=[
        'window_id'], direction='nearest')

    return all_features


def merge_features(statistical_features, frequency_features):
    index = statistical_features.index.get_indexer(
        frequency_features.index, 'nearest')
    features = statistical_features
    features.loc[index, frequency_features.columns] = frequency_features
    return features


def calculate_features(merged_data: dict):
    all_participants_data = {}

    for pnum in merged_data.keys():
        utils.logger.info(f'Calculating features for {pnum}')
        features = calculate_features_per_participant(merged_data[pnum])
        all_participants_data[pnum] = features

    return all_participants_data


if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        merged_data = utils.load_pickle(
            pickled_file_path=input_pickle_file_path)

    featurized_data = calculate_features(merged_data)

    if SAVE_PICKLE_FILE:
        output_pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=featurized_data,
                          file_path=output_pickle_file_path)