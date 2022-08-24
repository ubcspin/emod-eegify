from scipy import signal
import numpy as np
import pandas as pd


from config import TIME_INDEX, TIME_INTERVAL, FS, WINDOW_TYPE, WINDOW_SIZE

COLUMNS = None




def generate_eeg_features(dataset):
    sample_freq = 1000
    # get FFT
    psd_windows = [signal.periodogram(x[:,2:], sample_freq, window='hamming', axis=0) for x in dataset ] # get the power spectral density for each window

    # frequency bands
    bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4), 'theta': (4, 7), 'gamma': (30, 50)}
    band_freqs = [bands[x] for x in ['alpha', 'beta', 'delta', 'theta', 'gamma']]

    features = []
    for window in psd_windows: # calculate the power in each band for channel for each window
        freqs, psd = window
        idx_bands = [np.logical_and(freqs >= low, freqs <= high) for low,high in band_freqs]

        freq_res = freqs[1] - freqs[0]
        band_powers = np.array([sp.integrate.simpson(psd[idx,:], dx=freq_res, axis=0) for idx in idx_bands]) # (5,64)

        normed_powers = band_powers
        diff_entropy = np.log(normed_powers)
        # (5, 1, 64)
        # (5, 64, 1)
        diff_de = np.expand_dims(diff_entropy, axis=2) - np.expand_dims(diff_entropy, axis=1) # (5,64,64)
        diff_de = (diff_de  - diff_de.min(axis=(1,2), keepdims=True))/(diff_de.max(axis=(1,2), keepdims=True) - diff_de.min(axis=(1,2), keepdims=True))
        
        features.append(diff_de)
    return features


def calc_freq_features(df, columns, fs=FS, window_size=WINDOW_SIZE, window_type=WINDOW_TYPE, time_index=TIME_INDEX, time_interval=TIME_INTERVAL):
    freq_features = pd.DataFrame()

    for column in columns:
        res = calculate_freq_features_per_column(
            df[column], fs=fs, window_size=window_size, window=window_type, time_index=time_index)
        res = res.add_suffix('_' + column)
        freq_features = res.join(freq_features)

    freq_features = freq_features.reset_index()
    freq_features = freq_features.assign(window_id=freq_features.groupby(
        pd.Grouper(key=time_index, freq=time_interval)).ngroup())

    return freq_features


def calculate_freq_features_per_column(data, fs=FS, window_size=WINDOW_SIZE, window=WINDOW_TYPE, time_index=TIME_INDEX):
    if window == 'hann':
        window = signal.get_window('hann', int(fs * window_size))

    results = {}

    f, t, Sxx = signal.spectrogram(data, fs, window=window)

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

    return results
