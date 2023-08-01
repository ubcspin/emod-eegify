import os
from tabnanny import verbose
import mne
import pandas as pd
import numpy as np

print(pd. __version__)

from tqdm import tqdm

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config import DEBUG, FS, SAMPLE_PERIOD, MAX_CONTINUOUS_ANNOTATION
sys.path.remove(str(_parentdir))

import gc

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = 'subject_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'cleaned_data.pk'


FILE_ORDER = [
    "bpm.csv",
    "flag.csv",
    "gsr.csv",
    "feeltrace.csv",
    "touch.csv",
    "calibrated_words.csv"
]

def parse_data(subject_data: dict, file_order=FILE_ORDER):
    channel_names = ['T' + str(i) for i in range(1, 101)]
    order = {v: i for i, v in enumerate(file_order)}

    for pnum in tqdm(subject_data.keys()):
        utils.logger.info(f'Parsing data for {pnum}')

        subject_data[pnum].sort(key=lambda x: order[x['filename']])

        subject_data_iter = iter(subject_data[pnum])

        df = pd.DataFrame.from_dict(next(subject_data_iter)['df'])
        
        df['timestamps'] = pd.to_datetime(df.timestamps, unit='ms').astype('datetime64[ms]')
        # print(df.timestamps)
        for subject in subject_data_iter:
            subject_df = subject['df']

            utils.logger.info(f'Fix sampling to {FS}Hz')
            subject_df['timestamps'] =  pd.to_datetime(subject_df.timestamps, unit='ms').astype('datetime64[ms]')

            subject_df = subject_df.drop_duplicates(subset=['timestamps'])
            subject_df = subject_df.set_index('timestamps', verify_integrity=True)
            # l = list(subject_df.index)
            # print(len(set(subject_df.index)), len(subject_df.index), type(subject_df.index))
            # print(len([x for x in l if l.count(x) > 1]))
            subject_df = subject_df.resample(SAMPLE_PERIOD, origin='start')
            
            utils.logger.info('Filling NaNs')
            subject_df = subject_df.ffill() # fill nan values with the previous valid value
            # subject_df.reset_index(inplace=True)

            df = pd.merge(df, subject_df, on='timestamps')
        
        df['pnum'] = pnum
        df['timestamps'] = df.timestamps.astype('int64') / 1e9

        utils.logger.info('Sorting DataFrame')
        df.sort_values(by='timestamps', inplace=True)
        df.reset_index(inplace=True, drop=True)

        df['timedelta'] = pd.TimedeltaIndex(df.timestamps, unit='s')

        utils.logger.info('Scaling continuous_annotation to 0-1 range')
        maxfeeltrace = df.loc[2:, df.columns.str.contains('feeltrace')].max()
        df.loc[:, df.columns.str.contains(
            'continuous_annotation')] = df.loc[:, df.columns.str.contains('feeltrace')] / maxfeeltrace

        utils.logger.info('Scaling calibrated values to 0-1 range')
        maxcalibrated = df.loc[:, df.columns.str.contains('calibrated_values')].max()
        df.loc[:, df.columns.str.contains(
            'calibrated_values')] = df.loc[:, df.columns.str.contains('calibrated_values')] / maxcalibrated

        # # eeg functions
        # ch_types = 'eeg'
        # verbose = 'DEBUG' if DEBUG else 'WARNING'
        # mne.set_log_level(verbose=verbose)
        # info = mne.create_info(channel_names, FS, ch_types)
        # # load eeg into mne package
        # montage = mne.channels.make_standard_montage('GSN-HydroCel-65_1.0')
        # raw = mne.io.RawArray(df[channel_names].to_numpy().transpose()/(1e6), info) # divide by 1e6 since uV
        # raw.set_montage(montage)
        # raw.set_channel_types({'E62': 'eog'})
        # raw.drop_channels('Cz') # drop reference channel

        # utils.logger.info('Applying bandpass (1Hz to 50Hz) to EEG')
        # raw.filter(l_freq=1, h_freq=50, filter_length='auto', phase='zero') # apply bandpass filter, no phase so non-causal

        # # apply baseline correction
        # utils.logger.info('Applying baseline correction')
        # eeg_arr = raw.get_data()
        # comment out baseline correction for now: BG 2023-06-02
        # mne.baseline.rescale(eeg_arr, raw.times, (None, None), mode='mean', copy=False) # baseline correction
        
        # df.drop(columns=[channel_names[-1]], inplace=True)
        # df[channel_names[:-1]] = eeg_arr.transpose() # replace eeg data with baseline corrected data and filtered data

        merged_data = {}
        merged_data[pnum] = df

        if SAVE_PICKLE_FILE:
            utils.logger.info('Saving data')
            output = f"{pnum}_" + OUTPUT_PICKLE_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR,  output)
            utils.pickle_data(data=merged_data,
                          file_path=output_pickle_file_path)
        gc.collect()


if __name__ == "__main__":
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        subject_data = utils.load_pickle(input_pickle_file_path)
        parse_data(subject_data)
