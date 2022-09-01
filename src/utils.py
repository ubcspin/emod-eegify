import pickle
import logging

from config import DEBUG

from sklearn.model_selection import StratifiedKFold

import numpy as np


logging.basicConfig(
    filename='emod-eeg_log.log', format='%(asctime)-6s: %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', level=logging.DEBUG)


logger = logging.getLogger(__name__)

def pickle_data(file_path: str, data: any):
    logging.info(f'Saving file {file_path}')

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    if DEBUG:
        data = load_pickle(file_path)


def load_pickle(pickled_file_path: str):
    logging.info(f'Loading file {pickled_file_path}')

    data = pickle.load(open(pickled_file_path, 'rb'))

    if DEBUG:
        try:
            assert data != []
        except AssertionError:
            logging.error('What a pickle! Pickled file is empty, proceed with caution.')
    
    return data


def save_numpy(file_path: str, data: np.array):
    logging.info(f'Saving file {file_path}')

    with open(file_path, 'wb') as f:
        np.save(f, data)

    if DEBUG:
        data = load_numpy(file_path)


def load_numpy(file_path: str):
    logging.info(f'Loading file {file_path}')

    data = np.load(open(file_path, 'rb'), allow_pickle=True)

    if DEBUG:
        try:
            assert data != []
        except AssertionError:
            logging.error('What a pickle! Numpy file is empty, proceed with caution.')
    
    return data



def split_dataset(labels, k=5):
    '''
    split the features and labels into k groups for k fold validation
    we use StratifiedKFold to ensure that the class distributions within each sample is the same as the global distribution
    '''
    
    kf = StratifiedKFold(n_splits=k, shuffle=True)

    # only labels are required for generating the split indices so we ignore it
    temp_features = np.zeros_like(labels)
    indices = [(train_index, test_index) for train_index, test_index in kf.split(temp_features, labels)]
    return indices