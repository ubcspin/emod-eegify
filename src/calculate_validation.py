import os
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ShuffleSplit




INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = '_featurized_data.pk'
INPUT_LABEL_NAME = '_labels.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = '_val_featurized_data.pk'
OUTPUT_LABEL_NAME = '_val_labels.pk'

SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']



if __name__ == '__main__':

    for window_size in [500]:

        train_labels = {}
        val_labels = {}

        for subject_id in tqdm(SUBJECT_IDS):
            training_data_filename = subject_id + "_" + str(window_size) + 'ms' + INPUT_PICKLE_NAME
            input_pickle_file_path = os.path.join(INPUT_DIR, training_data_filename)
            input_label_file_path = os.path.join(INPUT_DIR, str(window_size) + 'ms' + INPUT_LABEL_NAME)

            features = utils.load_pickle(pickled_file_path=input_pickle_file_path)
            labels = utils.load_pickle(pickled_file_path=input_label_file_path)


            splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=None)

            for train_index, val_index in splitter.split(features[subject_id]):
                continue
                 
            train_features = features[subject_id].iloc[train_index]
            val_features = features[subject_id].iloc[val_index]

            train_l = labels[subject_id].iloc[train_index]
            val_l = labels[subject_id].iloc[val_index]


            train_feature = {}
            train_feature[subject_id] = train_features

            val_feature = {}
            val_feature[subject_id] = val_features

            train_labels[subject_id] = train_l
            val_labels[subject_id] = val_l

   
            if SAVE_PICKLE_FILE:
                output = subject_id + "_" + str(window_size) + 'ms' + INPUT_PICKLE_NAME
                output_pickle_file_path = os.path.join(OUTPUT_DIR, output)

                utils.pickle_data(data=train_feature,
                          file_path=output_pickle_file_path)
                
                output = subject_id + "_" + str(window_size) + 'ms' + OUTPUT_PICKLE_NAME
                output_pickle_file_path = os.path.join(OUTPUT_DIR, output)

                utils.pickle_data(data=val_feature,
                          file_path=output_pickle_file_path)



        if SAVE_PICKLE_FILE:
            output = str(window_size) + 'ms' + INPUT_LABEL_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=train_labels,
                              file_path=output_pickle_file_path)
            
            output = str(window_size) + 'ms' + OUTPUT_LABEL_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=val_labels,
                              file_path=output_pickle_file_path)