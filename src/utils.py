import contextlib
import glob
import os

import joblib

import mne
import numpy as np
import pandas as pd
import scipy.io as sp_io
from joblib import Parallel, delayed
from mne.preprocessing import ICA
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    """This is a random helper function"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def create_dataset(src_dir: str, out_dir = 'feeltrace', num_workers=2) -> None:
    """
    :param src_dir: the directory containing all the EEG and Feeltrace data in folders for each subject
    :param out_dir: output directory to write to
    :param num_workers: number of parallel processes to run

    Creates the normalized and cropped dataset in the EEG_FT_DATA directory, throws an error if
    EEG_FT_DATA does not exist
    """
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))
    subject_data_dir.sort(key=lambda x:int(x.split("p")[-1])) # sort in non decreasing order

    subject_data = [glob.glob(os.path.join(x, '*')) for x in subject_data_dir]

    all_eeg_data = [ next(filter(lambda item: 'eeg.mat' in item and 'eeg_eeg.mat' not in item, x))for x in subject_data] # find all eeg.mat
    all_joystick_data = [ next(filter(lambda item: 'joystick.mat' in item and 'joystick_joystick.mat' not in item, x)) for x in subject_data] # final all joystick.mat
    all_scene_data = [ next(filter(lambda item: 'scenes.mat' in item, x)) for x in subject_data] # final all joystick.mat


    # the next steps takes a bit of time!!
    eeg_ft_pairs = [(eeg, ft, scene) for eeg, ft, scene in zip(all_eeg_data, all_joystick_data, all_scene_data)]

    with tqdm_joblib(tqdm(desc="Dataset Creation", total=len(eeg_ft_pairs))) as progress_bar:
        Parallel(n_jobs=num_workers)(delayed(write_to_csv_dataset_loop)(i, x, y, z, out_dir) for i, (x, y, z) in enumerate(eeg_ft_pairs))
    print(f'Created dataset initial dataset csv in {out_dir}')


def write_to_csv_dataset_loop(index: int, x: str, y: str, z:str, out_dir) -> None:
    """
    Should not be called by the user, for pair at index, create the pandas dataframe and write to a csv file
    :param y: FeelTrace filename
    :param x: EEG filename
    :param index: index of the pair to write
    """

    eeg_column_headers = ['t'] + [f'channel_{i}' for i in range(64)]
    ft_column_headers = ['t', 'stress']
    scenes_column_headers = ['t', 'scene']

    eeg = sp_io.loadmat(x)['var']
    ft = sp_io.loadmat(y)['var']
    #scenes_data = sp_io.loadmat(z)['var'][:,[-2,-4]] # extract the time and label columns
    #scenes_data = np.apply_along_axis(lambda x: [x[0].item(),x[1].item()], -1, scenes_data) # clean up nested arrays

    normalized_eeg, normalized_ft = filter_normalize_crop(eeg, ft)

    eeg_df = pd.DataFrame(data=normalized_eeg, columns=eeg_column_headers, dtype='float64')
    ft_df = pd.DataFrame(data=normalized_ft, columns=ft_column_headers, dtype='float64')
    #scenes_df = pd.DataFrame(data=scenes_data, columns=scenes_column_headers)
    #scenes_df = scenes_df.astype({'t':'float64', 'scene':'str'})

    ft_df = interpolate_df(ft_df, resample_period='1ms') # resample every milliseconds using zero-hold

    eeg_df['t'] = pd.to_datetime(eeg_df['t'], unit='ms').astype('datetime64[ms]') # change precision to ms
    #scenes_df['t'] = pd.to_datetime(scenes_df['t'], unit='ms').astype('datetime64[ms]') # change precision to ms


    merged_df = pd.merge(ft_df, eeg_df, on='t', copy=False) # merge where times overlap
    merged_df['t'] = merged_df['t'].astype('int64') / 1e9 # nano second to seconds
    #scenes_df['t'] = scenes_df['t'].astype('int64') / 1e9 # nano second to seconds
    merged_df.to_csv(os.path.join(out_dir, f'eeg_ft_{index}.csv'), index=False)
    #scenes_df.to_csv(os.path.join(out_dir, f'scenes_{index}.csv'), index=False)

    del merged_df
    del eeg_df
    del ft_df
    #del scenes_df


def filter_normalize_crop(eeg: np.array, ft: np.array) -> tuple:
    """
    EEG -> Apply a notch filter at 60Hz, remove eye blinks through ICA, crop and normalize between [0,1]
    Feeltrace -> crop and normalize between [0,1]

    :param eeg: raw eeg signal
    :param ft: raw feeltrace signal
    :return: eeg and ft signals after processing
    """

    # process EEG  here to remove last noisy channel
    channel_names = [ 'E' + str(i+1) for i in range(64)] + ['Cz']
    sampling_rate = 1000 # Hz
    ch_types = 'eeg'
    info = mne.create_info(channel_names, sampling_rate, ch_types)

    # load eeg into mne package
    montage = mne.channels.make_standard_montage('GSN-HydroCel-65_1.0')
    raw = mne.io.RawArray(eeg[:, 1:].transpose()/(10 ** 6), info) # divide by 10^6 since uV
    raw.set_montage(montage)
    raw.set_channel_types({'E62': 'eog'})
    raw.drop_channels('Cz')
    # notch 60Hz
    raw.notch_filter(np.arange(60, 301, 60), filter_length='auto', phase='zero') # make filter non-causal to remove phase

    # EOG artifact removal through ICA (eye blinking removal)
    ica = ICA(n_components=15)
    ica.fit(raw)
    ica.plot_components(show=False)

    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='E62', measure='correlation', threshold=0.5)
    ica.exclude = eog_indices
    ica.apply(raw)
    
    new_eeg = eeg[:,:-1] # drop last channel
    new_eeg[:, 1:] = raw.get_data().transpose()

    # normalize to be between [0,1]
    # min/max determined from data
    min_ft = 0
    max_ft = 225

    ft[:, 1] = (ft[:, 1] - min_ft) / (max_ft - min_ft)

    return new_eeg, ft


#### TODO: comment the functions below


def interpolate_df(df, timestamps='t', resample_period='33ms'):
    '''
    resample and fill in nan values using zero hold
    '''
    df[timestamps] = pd.to_datetime(df[timestamps], unit='ms')
    df_time_indexed = df.set_index(timestamps)
    df_padded = df_time_indexed.resample(resample_period, origin='start') # resample
    df_padded = df_padded.ffill() # fill nan values with the previous valid value
    df_padded.reset_index(inplace=True)
    return df_padded

