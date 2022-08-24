import pickle
import logging

from config import DEBUG


logging.basicConfig(
    filename='emod-eeg_log.log', format='%(asctime)-6s: %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', level=logging.DEBUG)


logger = logging.getLogger(__name__)

def pickle_data(file_path: str, data: Any):
    logging.info(f'Saving file {file_path}')

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    if DEBUG:
        data = load_pickle(file_path)


def load_pickle(pickled_file_path: str):
    logging.info(f'Loading file {file_path}')

    data = pickle.load(open(file_path, 'rb'))

    if DEBUG:
        try:
            assert data != []
        except AssertionError:
            logging.error('What a pickle! Pickled file is empty, proceed with caution.')
    
    return data