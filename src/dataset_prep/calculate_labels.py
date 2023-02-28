import os
import utils

import numpy as np
import pandas as pd

from tqdm import tqdm

from config import TIME_INDEX, TIME_INTERVAL, LABEL_CLASS_COUNT, WINDOW_SIZE, EXP_PARAMS, FS


INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'merged_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'labels.pk'

COLUMNS = None

LABELS = ['pos', 'angle', 'acc', 'cw_mode']

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def calculate_labels_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, labels=LABELS, n_labels=LABEL_CLASS_COUNT, fs=FS):
    df = df.assign(window_id=df.groupby(pd.Grouper(key=time_index, freq=time_interval)).ngroup())

    if columns is None:
        columns = ['window_id'] + ['continuous_annotation'] + ['calibrated_words']

    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()
    windows = []
    n_samples = window_size / 1e3 * fs
    n_samples = int(n_samples)
    utils.logger.info(f'Removing windows less than {window_size} ms ({n_samples})')
    for window in grouped_data:
        if window.shape[0] < n_samples:
            utils.logger.info(f'Window removed {window.shape[0]}, expected size {n_samples}')
            continue
        elif window.shape[0] > n_samples:
            utils.logger.info(f'Window cropped {window.shape[0]}, expected size {n_samples}')
            window = window[:n_samples, :]
        windows.append(window)


    results = {}
    for label_type in labels:
        utils.logger.info(f'Calculating {label_type} labels')

        if 'cw' in label_type:
            if 'mode' in label_type: 
                results[label_type] = np.vstack([ mode(x[:, 2])[0] for x in windows]).squeeze()
                continue


        results[label_type], _ = get_label(
            windows, n_labels=n_labels, label_type=label_type)

    results['window_id'] = np.arange(len(windows))
    results = pd.DataFrame(results)

    return results


def calculate_labels(merged_data: dict, window_size):
    all_labels = {}

    if not isinstance(window_size, str):
        time_interval = str(window_size) + 'ms'

    for pnum in tqdm(merged_data.keys()):
        utils.logger.info(f'Calculating labels for {pnum}')
        labels = calculate_labels_per_participant(merged_data[pnum], time_interval=time_interval, window_size=window_size)

        all_labels[pnum] = labels

    return all_labels


def get_label(data, n_labels=3, label_type='angle'):
    if label_type == 'angle':
        # angle/slope mapped to [0,1] in a time window
        labels = stress_2_angle(np.vstack([x[:, 1].astype(float).T for x in data]))
    elif label_type == 'pos':
        # mean value within the time window
        labels = np.vstack([x[:, 1].astype(float).mean() for x in data])
    elif label_type == 'acc':
        # accumulator mapped to [0,1] in a time window
        labels = stress_2_accumulator(np.vstack([x[:, 1].astype(float).T for x in data]))
    else:
        raise ValueError

    label_dist = stress_2_label(labels, n_labels=n_labels).squeeze()
    return label_dist, labels.squeeze()


def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(n_labels)) - 1


def stress_2_angle(stress_windows):
    '''
    do a linear least squares fit in the time window
    stress_window: (N_samples, time_window)
    '''
    xvals = np.arange(stress_windows.shape[-1])/1e3/60  # time in (minutes)
    slope = np.polyfit(xvals, stress_windows.T, 1)[
        0]  # take slope linear term # 1/s
    angle = np.arctan(slope) / (np.pi/2) * 0.5 + 0.5  # map to [0,1]
    return angle


def stress_2_accumulator(stress_windows):
    '''
    apply an integral to the time window
    stress_window: (N_samples, time_window)
    '''
    max_area = stress_windows.shape[-1]
    xvals = np.arange(stress_windows.shape[-1])  # time in (ms)
    integral = np.trapz(stress_windows, x=xvals)
    return integral/max_area  # map to [0,1]


def run():
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        merged_data = utils.load_pickle(
            pickled_file_path=input_pickle_file_path)

    window_sizes = EXP_PARAMS["WINDOW_SIZE"]

    for wsize in window_sizes:
        utils.logger.info(f'Calculating labels for window size: {wsize} ms')
        label_data = calculate_labels(merged_data, wsize)

        if SAVE_PICKLE_FILE:
            utils.logger.info('Saving data')
            output = str(wsize) + 'ms_' + OUTPUT_PICKLE_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=label_data,
                              file_path=output_pickle_file_path)