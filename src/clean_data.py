import os
import mne
import utils

import pandas as pd
import numpy as np

from config import TIME_INDEX, TIME_INTERVAL, FS, SAMPLE_PERIOD, MAX_FEELTRACE

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'subject_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'merged_data.pk'

FILE_ORDER = [
    'eeg.csv',
    'joystick.csv'
]


def parse_data(subject_data: dict, file_order=FILE_ORDER):
    merged_data = {}
    channel_names = [ 'E' + str(i+1) for i in range(64)] + ['Cz']
    cols = ['timestamps', 'feeltrace'] + channel_names
    order = {v: i for i, v in enumerate(file_order)}

    for pnum in subject_data.keys():
        utils.logger.info(f'Parsing data for {pnum}')

        subject_data[pnum].sort(key=lambda x: order[x['filename']])

        subject_data_iter = iter(subject_data[pnum])

        df = pd.DataFrame.from_dict(next(subject_data_iter)['df'])

        for subject in tqdm(subject_data_iter):
            subject_df = subject['df']
            df = pd.merge_ordered(df, subject_df, on='timestamps', suffixes=(
                None, '_' + subject['filename'].split('.')[0]))
        df['pnum'] = pnum

        utils.logger.info('Sorting DataFrame')
        df.sort_values(by='timestamps', inplace=True)
        df.reset_index(inplace=True, drop=True)

        utils.logger.info('Filling NaNs')
        df[cols] = df[cols].bfill().ffill()
        df = df[~df.timestamps.duplicated()]


       
        # utils.logger.info('Creating combined keypress keys')
        # fsr_cols = list(map(lambda x: 'a' + str(x), range(5)))
        # df['a5'] = np.sum(df[fsr_cols], axis=1)
        # df['a6'] = np.max(df[fsr_cols], axis=1)

        utils.logger.info('Fix sampling to {FS}Hz')
        df['timedelta'] = pd.TimedeltaIndex(df.timestamps, unit='ms')
        df = df.set_index('timedelta').resample(SAMPLE_PERIOD).nearest()
        df.reset_index(inplace=True, drop=False)
        df['timestamps'] = df.timedelta.astype('int64') / 1e9

        utils.logger.info('Scaling feeltrace to 0-1 range')
        df.loc[:, df.columns.str.contains(
            'feeltrace')] = df.loc[:, df.columns.str.contains('feeltrace')] / MAX_FEELTRACE


        # eeg functions
        ch_types = 'eeg'
        info = mne.create_info(channel_names, FS, ch_types)
        # load eeg into mne package
        montage = mne.channels.make_standard_montage('GSN-HydroCel-65_1.0')
        raw = mne.io.RawArray(df[channel_names].to_numpy().transpose()/(1e6), info) # divide by 1e6 since uV
        raw.set_montage(montage)
        raw.set_channel_types({'E62': 'eog'})
        raw.drop_channels('Cz') # drop reference channel

        utils.logger.info('Applying bandpass (0.5Hz to 50Hz) to EEG')
        raw.filter(l_freq=0.5, h_freq=50, filter_length='auto', phase='zero') # apply bandpass filter, no phase so non-causal
        
        df.drop(columns=[channel_names[-1]])
        df[channel_names[:-1]] = raw.get_data().transpose()

        merged_data[pnum] = df

    return merged_data


if __name__ == "__main__":
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        subject_data = utils.load_pickle(input_pickle_file_path)

    merged_data = parse_data(subject_data)

    if SAVE_PICKLE_FILE:
        output_pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=merged_data,
                          file_path=output_pickle_file_path)
