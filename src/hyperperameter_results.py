import os
import utils
from config import EXP_PARAMS


INPUT_DIR = 'RESULTS'
INPUT_PICKLE_NAME = 'results.pk'


SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']

def get_hyperparameter_results():
    for window_size in EXP_PARAMS["WINDOW_SIZE"]:
        for subject_id in SUBJECT_IDS:
            input_pickle_file_path = os.path.join(INPUT_DIR, subject_id + "_" + "0" + '_' + str(window_size) + 'ms_hc_cw_' + INPUT_PICKLE_NAME)
            features = utils.load_pickle(pickled_file_path=input_pickle_file_path)
            features.head()


if __name__ == '__main__':
    get_hyperparameter_results()
