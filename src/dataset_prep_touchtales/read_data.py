import os
import re
import glob


import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
sys.path.remove(str(_parentdir))

import pandas as pd

from tqdm import tqdm

# Filenames and columns
FILES_DICT = {
    "bpm.csv": ['timestamps', 'BPM'],
    "flag.csv": ['timestamps', 'flag'],
    "gsr.csv": ['timestamps', 'GSR'],
    "feeltrace.csv": ['timestamps', 'feeltrace'],
    "touch.csv": ['timestamps'] + ['T'+str(i) for i in range(1, 101)],
    "calibrated_words.csv": ['timestamps', 'calibrated_words', 'calibrated_values']
}

RAW_DATA_PATH = 'touchtale'
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = 'subject_data.pk'
SAVE_PICKLE_FILE = True

def read_dataset(src_dir=RAW_DATA_PATH, output_dir=OUTPUT_DIR, file_dict=FILES_DICT):
    utils.logger.info(f'Reading data from {src_dir}')

    os.makedirs(output_dir, exist_ok=True)
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))
    all_subjects_files = [glob.glob(os.path.join(x, '*'))
                          for x in subject_data_dir]
    
    print(subject_data_dir)
    
    subjects = {}

    for subject_files in all_subjects_files:
        subject_files = sorted(subject_files)
        pnum = extract_pnum(subject_files[0])

        utils.logger.info(f'Parsing files for {pnum}')

        subjects[pnum] = parse_files(subject_files, file_dict)

    return subjects


def extract_pnum(filename: str):
    match = re.search('[0-9]?[0-9]', filename)
    return 'p' + "%02d" % int(match.group(0))


def parse_files(subject_files: list, file_dict=FILES_DICT):
    subject_data = []


    for file in tqdm(subject_files):
        utils.logger.info(f'Assessing {file}')
        filename = file.split('/')[-1]


        if not filename in file_dict.keys():
            utils.logger.info(f'Skipping {filename}')
            continue

        utils.logger.info(f'Reading {filename}')

        print(file_dict[filename])

        x = pd.read_csv(file, names=file_dict[filename], header=0, low_memory=False)
        subject_data.append({'filename': filename, 'df': x})

    return subject_data


if __name__ == "__main__":
    subject_data = read_dataset(RAW_DATA_PATH)

    if SAVE_PICKLE_FILE:
        pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=subject_data, file_path=pickle_file_path)