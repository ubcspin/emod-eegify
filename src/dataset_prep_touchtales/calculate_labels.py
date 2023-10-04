import os

import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config_touchtale import TIME_INDEX, TIME_INTERVAL, LABEL_CLASS_COUNT, WINDOW_SIZE, EXP_PARAMS, FS, N_SAMPLES, SUBJECT_IDS
sys.path.remove(str(_parentdir))

from sklearn.preprocessing import normalize
from scipy.special import erfinv

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = 'cleaned_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'labels.pk'

COLUMNS = None

# LABELS = ['pos', 'angle', 'acc']
LABELS = ['pos', 'angle', 'acc', 'cw_mode']


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def calculate_labels_per_participant(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=COLUMNS, window_size=WINDOW_SIZE, labels=LABELS, n_labels=LABEL_CLASS_COUNT, fs=FS, pnum=0):
    df = df.assign(window_id=df.groupby(pd.Grouper( key=time_index, freq=time_interval)).ngroup())

    if columns is None:
        columns = ['window_id'] + ['feeltrace'] + ['calibrated_words']

    grouped_data = df[columns].groupby('window_id').apply(np.array).to_numpy()
    windows = grouped_data
    # n_samples = N_SAMPLES
    # utils.logger.info(f'Removing windows less than {window_size} ms ({n_samples})')

    # print("group data shape: ", np.array(grouped_data).shape)
    # windows = []
    # for window in grouped_data:
    #     # window = window[:mean_windowsize, :]
    #     if window.shape[0] < n_samples:
    #         utils.logger.info(f'Window removed {window.shape[0]}, expected size {n_samples}')
    #         continue
    #     elif window.shape[0] > n_samples:
    #         utils.logger.info(f'Window cropped {window.shape[0]}, expected size {n_samples}')
    #         window = window[:n_samples, :]
    #     # print("window shape: ", np.array(window).shape, n_samples)
    #     windows.append(window)

    # print(len(windows))

    results = {}
    for label_type in labels:
        utils.logger.info(f'Calculating {label_type} labels')

        if 'cw' in label_type:
            if 'mode' in label_type: 
                results[label_type] = np.vstack([ mode(x[:, 2])[0] for x in windows]).squeeze()
                continue

        results[label_type], _ = get_label(
            windows, n_labels=n_labels, label_type=label_type, pnum=pnum)

    results['window_id'] = np.arange(len(windows))
    results = pd.DataFrame(results)

    # print("RESULTS: ", results)

    return results


def calculate_labels(window_size):
    all_labels = {}

    if not isinstance(window_size, str):
        time_interval = str(window_size) + 'ms'


    for pnum in tqdm(SUBJECT_IDS):
        if INPUT_PICKLE_FILE:
            input_pickle_file_path = os.path.join(INPUT_DIR, f"{pnum}_" + INPUT_PICKLE_NAME)
            merged_data = utils.load_pickle(
                pickled_file_path=input_pickle_file_path)
            
        print(f'Calculating labels for {pnum}')
        labels = calculate_labels_per_participant(merged_data[pnum], time_interval=time_interval, window_size=window_size, pnum=pnum)
        # labels = calculate_labels_per_participant(merged_data[pnum])

        l_header = ['pos', 'angle', 'acc', 'cw_mode']
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(7, 8)

        count = 0
        for row in ax:
            for col in row:
                col.hist(labels[l_header[count]], bins='auto')
                col.set_title(l_header[count])
                count += 1

        fig.suptitle(f"label distribution for {pnum} with window size {time_interval}")
        plt.savefig(f'images/{pnum}_{time_interval}.png')
        
        # print('label dimensions: ', labels.shape)
        all_labels[pnum] = labels

    return all_labels


def get_label(data, n_labels=3, label_type='angle', pnum=0):
    # print(data)
    if label_type == 'angle':
        # angle/slope mapped to [0,1] in a time window
        labels = stress_2_angle(np.array([x[:, 1].astype(float).T for x in data], dtype=object), n_labels=n_labels)
        label_dist = labels.squeeze()
        # print(label_dist)
    elif label_type == 'pos':
        # mean value within the time window
        v = np.array([x[:, 1].astype(float).mean() for x in data])
        labels = (v - v.min()) / (v.max() - v.min())
        label_dist = stress_2_label(labels, n_labels=n_labels).squeeze()
    elif label_type == 'acc':
        # accumulator mapped to [0,1] in a time window
        v = stress_2_accumulator(np.array([x[:, 1].astype(float).T for x in data], dtype=object))
        # print(sorted(v))
        N_STD = 3
        try:
            non_outliers = v[abs(v - np.mean(v)) < N_STD * np.std(v)]
            smaller_outliers_index, larger_outliers_index = v - np.mean(v) < -N_STD * np.std(v), v - np.mean(v) > N_STD * np.std(v)
            v[smaller_outliers_index] = non_outliers.min()
            v[larger_outliers_index] = non_outliers.max()
            
            labels = (v - v.min()) / (v.max() - v.min())
        except ValueError:  #raised if `y` is empty.
            print(v, len(v))
            raise ValueError

        # print(labels, erfinv(np.linspace(labels.std()*2, min(1, labels.std()*4), n_labels)), pnum)
        # print(labels.mean(), labels.std(), np.count_nonzero(labels < labels.mean()), len(labels))
        label_dist = np.digitize(labels, erfinv(np.linspace(labels.std(), min(1, labels.std()*3), n_labels))) 
        # label_dist = stress_2_label(labels, n_labels=n_labels).squeeze()
    else:
        raise ValueError
    
    # print(label_type, labels)


    # print("dist", label_dist, label_type)
    return label_dist, labels.squeeze()


def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(0, n_labels)) - 1


def stress_2_angle(stress_windows, n_labels=5):
    '''
    do a linear least squares fit in the time window
    stress_window: (N_samples, time_window)
    '''
    xvals = [np.arange(len(l)) for l in stress_windows]  # time in (minutes)
    slope = []
    for i, l in enumerate(stress_windows):
        try:
            if len(xvals[i]) != 1:
                slope.append(np.polyfit(xvals[i], l, 1)[0])
            else:
                slope.append(0)
        except Exception as e:
            print("error", xvals[i], stress_windows[i])
            print(e)
    angle = [np.arctan(s) / (np.pi/2) * 100 for s in slope]  # map to [0,1]
    # print(angle)
    
    return np.digitize(np.array(angle) * n_labels, erfinv(np.linspace(-1,1,n_labels))) 


def stress_2_accumulator(stress_windows):
    '''
    apply an integral to the time window
    stress_window: (N_samples, time_window)
    '''
    max_area = len(stress_windows)
    xvals = [np.arange(len(l)) for l in stress_windows]  # time in (ms)
    integral = []
    for i, l in enumerate(stress_windows):
        try:
            if len(xvals[i]) != 1:
                integral.append(np.trapz(l, x=xvals[i]))
            else:
                integral.append(0)
        except Exception as e:
            print("error", xvals[i], stress_windows[i])
            print(e)

    # integral = np.array(integral)
    # ret = (integral - integral.mean()) / (integral.max() - integral.min())
    return np.array(integral)  # map to [0,1]


if __name__ == "__main__":
    window_sizes = EXP_PARAMS["WINDOW_SIZE"]

    for wsize in window_sizes:
        print(f'Calculating labels for window size: {wsize} ms')
        label_data = calculate_labels(wsize)

        print(label_data)

        if SAVE_PICKLE_FILE:
            output = str(wsize) + 'ms_' + OUTPUT_PICKLE_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=label_data,
                              file_path=output_pickle_file_path)
            print(f'Saved data to {output_pickle_file_path}')