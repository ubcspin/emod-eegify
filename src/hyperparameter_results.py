import os
import utils
from config import EXP_PARAMS
import pandas as pd


INPUT_DIR = 'RESULTS'
INPUT_PICKLE_NAME = 'results.pk'


SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']

def get_hyperparameter_results():
    hps = {}
    for window_size in EXP_PARAMS["WINDOW_SIZE"]:
        for subject_id in SUBJECT_IDS:
            input_pickle_file_path = os.path.join(INPUT_DIR, subject_id + "_" + "0" + '_' + str(window_size) + 'ms_hc_cw_' + INPUT_PICKLE_NAME)
            features = utils.load_pickle(pickled_file_path=input_pickle_file_path)
            best_idx = features['scores_hc_angle']['mean_test_f1'].argmax()
            best_params = features['scores_hc_angle']['params'][best_idx]
            hps[subject_id + "_" + str(window_size) + "ms"] =  best_params

    hps = pd.DataFrame.from_dict(hps)
    print(hps)
    hps.to_csv(os.path.join('validation','hyperparams.csv'))  
if __name__ == '__main__':
    get_hyperparameter_results()
