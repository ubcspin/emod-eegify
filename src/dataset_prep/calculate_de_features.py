from scipy import signal
import scipy as sp
import numpy as np
import pandas as pd


from config import FS, WINDOW_TYPE


COLUMNS = None
BANDS = {'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4), 'theta': (4, 7), 'gamma': (30, 50)}



def calculate_de_features(dataset, fs=FS, window_type=WINDOW_TYPE):
    # get the power spectral density for each window
    psd_windows = [signal.periodogram(x[:,1:], fs, window=window_type, axis=0) for x in dataset ]

    # frequency bands
    chosen_bands = ['alpha', 'beta', 'delta', 'theta', 'gamma']
    band_freqs = [BANDS[x] for x in chosen_bands]

    features = []
    for window in psd_windows: # calculate the power in each band for channel for each window
        freqs, psd = window
        # find the freqs between low and high
        idx_bands = [np.logical_and(freqs >= low, freqs <= high) for low,high in band_freqs]

        # frequency delta
        freq_res = freqs[1] - freqs[0]
        # calculate frequency power for each band
        band_powers = np.array([sp.integrate.simpson(psd[idx,:], dx=freq_res, axis=0) for idx in idx_bands]) # (5,64)

        normed_powers = band_powers
        diff_entropy = np.log(normed_powers)
        # take the channel wise difference by expanding the matrices into (5, 1, 64) and (5, 64, 1) --> (5,64,64)
        diff_de = np.expand_dims(diff_entropy, axis=2) - np.expand_dims(diff_entropy, axis=1) # (5,64,64)
        # normalize to [0,1]
        diff_de = (diff_de  - diff_de.min(axis=(1,2), keepdims=True))/(diff_de.max(axis=(1,2), keepdims=True) - diff_de.min(axis=(1,2), keepdims=True))
        
        features.append(diff_de)

    features = np.array(features)
    results = {}
    for index, band in enumerate(chosen_bands):
        results[band] = features[:,index,:,:].tolist()

    results['window_id'] = np.arange(len(features))
    results = pd.DataFrame(results)
    return results
