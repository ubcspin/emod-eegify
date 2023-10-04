import os
import utils

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config_touchtale import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS
sys.path.remove(str(_parentdir))

import pandas as pd
from calculate_chance import calc_chance


INPUT_DIR = '/Users/poyuchen/Desktop/UBC/Engineering-Physics/Fifth-Year/Summer/SPIN/emod-eegify.nosync/src/RESULTS/'
INPUT_PICKLE_NAME = 'results.pk'

NUMSETS = 1


SUBJECT_IDS = [x[:3] if 'p0' in x and 'cleaned' in x else '' for x in os.listdir('/Users/poyuchen/Desktop/UBC/Engineering-Physics/Fifth-Year/Summer/SPIN/emod-eegify.nosync/src/COMBINED_DATA_TOUCHTALE/')]
SUBJECT_IDS = list(filter(lambda a: a != '', SUBJECT_IDS))
print(SUBJECT_IDS)


def get_validation_results():
    columns = ['pnum', 'window_size', 'run', 'model', 'f1', 'precision', 'recall', 'chance_f1', 'chance_precision', 'chance_recall', 'chance_std_f1', 'chance_std_precision', 'chance_std_recall']
    results = []
    chance_results = {}

    for i in range(NUMSETS):
        for window_size in EXP_PARAMS["WINDOW_SIZE"]:
            for subject_id in SUBJECT_IDS:
                input_pickle_file_path = os.path.join(INPUT_DIR, subject_id + "_" + str(i) + '_' + str(window_size) + 'ms_hc_cw_' + INPUT_PICKLE_NAME)
                res = utils.load_pickle(pickled_file_path=input_pickle_file_path)

                # print(res)
                # break


                if i == 0:
                    labels = res['y_hc_angle']
                    # print(labels)
                    chance_metrics = calc_chance(labels, n=1000)
                    chance_results[subject_id + "_" + str(window_size)]  = chance_metrics
                else:
                    chance_metrics = chance_results[subject_id + "_" + str(window_size)] 

                item = [res['pnum'], window_size, i] + res['scores_hc_angle'].values.squeeze().tolist() + chance_metrics
                results.append(item)
    
    results = pd.DataFrame(results, columns=columns)
    results.to_csv(os.path.join('validation', 'final_results.csv'), index=False)

if __name__ == '__main__':
    get_validation_results()
