import read_data
import clean_data
import calculate_features

from config import DEBUG, FS, SAMPLE_PERIOD, MAX_CONTINUOUS_ANNOTATION


# Function to read in the data and save it as a pickle file
# Filenames and columns
FILES_DICT = {
    "eeg.csv": ['timestamps'] + [ 'E' + str(i+1) for i in range(64)] + ['Cz'],
    "joystick.csv": ['timestamps', 'continuous_annotation'],
    "calibrated_words.csv": ['timestamps', 'calibrated_words', 'calibrated_values']
}

RAW_DATA_PATH = '../../data/FEEL'
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'subject_data.pk'
SAVE_PICKLE_FILE = True

read_data.run(RAW_DATA_PATH, OUTPUT_DIR, FILES_DICT, OUTPUT_PICKLE_NAME, SAVE_PICKLE_FILE)



# Function to clean the data and save it as a pickle file
INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'subject_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'merged_data.pk'

FILE_ORDER = [
    'eeg.csv',
    'joystick.csv',
    'calibrated_words.csv'
]

clean_data.run(INPUT_PICKLE_FILE, INPUT_DIR, INPUT_PICKLE_NAME, SAVE_PICKLE_FILE,
               OUTPUT_DIR, OUTPUT_PICKLE_NAME, FILE_ORDER, SAMPLE_PERIOD,
               FS, MAX_CONTINUOUS_ANNOTATION, DEBUG)

from config import TIME_INDEX, EXP_PARAMS, FS


INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'merged_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'featurized_data.pk'

calculate_features.run(INPUT_PICKLE_FILE, INPUT_DIR, INPUT_PICKLE_NAME, SAVE_PICKLE_FILE,
        OUTPUT_DIR, OUTPUT_PICKLE_NAME, FS, TIME_INDEX, EXP_PARAMS)


import calculate_labels
calculate_labels.run()

import calculate_validation
calculate_validation.run()