import os
import utils
from config import EXP_PARAMS
import pandas as pd
from calculate_chance import calc_chance


INPUT_DIR = 'RESULTS'
INPUT_PICKLE_NAME = 'val_results.pk'


SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']

def get_validation_results():
    columns = ['pnum', 'window_size', 'run', 'model', 'f1', 'precision', 'recall', 'chance_f1', 'chance_precision', 'chance_recall', 'chance_std_f1', 'chance_std_precision', 'chance_std_recall']
    results = []
    chance_results = {}

    for i in range(30):
        for window_size in EXP_PARAMS["WINDOW_SIZE"]:
            for subject_id in SUBJECT_IDS:
                input_pickle_file_path = os.path.join(INPUT_DIR, subject_id + "_" + str(i) + '_' + str(window_size) + 'ms_hc_cw_' + INPUT_PICKLE_NAME)
                res = utils.load_pickle(pickled_file_path=input_pickle_file_path)


                if i == 0:
                    labels = res['y_hc_test_angle']
                    chance_metrics = calc_chance(labels, n=1000)
                    chance_results[subject_id] = chance_metrics
                else:
                    chance_metrics = chance_results[subject_id]

                item = [res['pnum'], window_size, i] + res['scores_hc_angle'].values.squeeze().tolist() + chance_metrics
                results.append(item)
    
    results = pd.DataFrame(results, columns=columns)
    results.to_csv(os.path.join('validation','final_results.csv'), index=False)  
if __name__ == '__main__':
    get_validation_results()
