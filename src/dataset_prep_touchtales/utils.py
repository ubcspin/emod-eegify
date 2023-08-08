import pickle
import logging
import numpy as np

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

from sklearn import metrics
from config_touchtale import DEBUG


logging.basicConfig(
    filename='emod-fsr_log.log', format='%(asctime)-6s: %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', level=logging.DEBUG)


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
            logging.error(
                'What a pickle! Pickled file is empty, proceed with caution.')

    return data


def auc_group(data=None, x=None, y=None):
    if x is None:
        x = data.index.astype('int64')
        y = data.values
    return metrics.auc(x, y)


def sum_abs_diffs(x):
    return np.sum(np.abs(np.diff(x)))