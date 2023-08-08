import pandas as pd
import numpy as np

from scipy import signal
from scipy.fft import fftshift

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
from config_touchtale import TIME_INDEX, TIME_INTERVAL, FS, WINDOW_TYPE, WINDOW_SIZE

COLUMNS = None


def calc_freq_features(df, columns, fs=FS, window_size=WINDOW_SIZE, window_type=WINDOW_TYPE, time_index=TIME_INDEX, time_interval=TIME_INTERVAL):
    freq_features = pd.DataFrame()

    print(df['window_id'].unique())

    for column in columns:
        res = calculate_freq_features_per_column(
            df[column], fs=fs, window_size=window_size, window=window_type, time_index=time_index)
        res = res.add_suffix('_' + column)
        freq_features = res.join(freq_features)
    print(freq_features.shape)

    freq_features = freq_features.reset_index()
    freq_features = freq_features.assign(window_id=freq_features.groupby(
        pd.Grouper(key=time_index, freq=time_interval)).ngroup())

    print(freq_features.shape)
    return freq_features


def calculate_freq_features_per_column(data, fs=FS, window_size=WINDOW_SIZE, window=WINDOW_TYPE, time_index=TIME_INDEX, time_interval=TIME_INTERVAL):
    if window == 'hann':
        window = signal.get_window('hann', int(fs * window_size))

    results = {}

    f, t, Sxx = signal.spectrogram(data, fs, window=window)

    # print(f.shape, t.shape, Sxx.shape)

    # max freqs
    results['freqs_max'] = f[np.argmax(Sxx, 0)]

    # max amplitudes
    results['freqs_max_amp'] = np.max(Sxx, 0)

    # min amplitudes
    results['freqs_min_amp'] = np.min(Sxx, 0)

    # variance
    results['freqs_var'] = np.var(Sxx, 0)

    # mean
    results['freqs_mean'] = np.mean(Sxx, 0)

    # peak count
    peaks = np.apply_along_axis(signal.find_peaks, 0, Sxx)
    peaks = peaks[0]
    peak_count = list(map(lambda x: len(x), peaks))
    results['freqs_peak_count'] = peak_count

    results = pd.DataFrame(results, index=pd.TimedeltaIndex(t, unit='s'))
    results.index.rename(time_index, inplace=True)

    # print(results.assign(window_id=results.groupby(
    #     pd.Grouper(key=time_index, freq=time_interval)).ngroup()))

    return results