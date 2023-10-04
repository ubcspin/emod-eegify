import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ShuffleSplit

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
import utils
from config_touchtale import EXP_PARAMS, SUBJECT_IDS
sys.path.remove(str(_parentdir))



INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
INPUT_PICKLE_NAME = '_featurized_data.pk'
INPUT_LABEL_NAME = '_labels.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA_TOUCHTALE'
OUTPUT_PICKLE_NAME = '_val_featurized_data.pk'
OUTPUT_LABEL_NAME = '_val_labels.pk'


def check_val_is_subset(train_labels, val_labels):
    train_angle, train_word = train_labels[['angle','cw_mode']].to_numpy().T
    val_angle, val_word = val_labels[['angle','cw_mode']].to_numpy().T

    train_word = list(train_word) # convert numpy string to list of strings
    val_word = list(val_word) # convert numpy string to list of strings

    train_dist = set([(l1,l2) for l1,l2 in zip(train_word, train_angle)]) # create a distribution of the training data
    val_dist = set([(l1,l2) for l1,l2 in zip(val_word, val_angle)]) # create a distribution of the validation data

    return val_dist.issubset(train_dist) # check if the validation distribution is a subset of the training distribution

if __name__ == "__main__":
    for window_size in EXP_PARAMS['WINDOW_SIZE']:

        train_labels = {}
        val_labels = {}

        for subject_id in tqdm(SUBJECT_IDS):
            training_data_filename = subject_id + "_" + str(window_size) + 'ms' + INPUT_PICKLE_NAME
            input_pickle_file_path = os.path.join(INPUT_DIR, training_data_filename)
            input_label_file_path = os.path.join(INPUT_DIR, str(window_size) + 'ms' + INPUT_LABEL_NAME)

            print(input_label_file_path)

            try:
                features = utils.load_pickle(pickled_file_path=input_pickle_file_path)
                labels = utils.load_pickle(pickled_file_path=input_label_file_path)
            except:
                features = utils.load_pickle(pickled_file_path=os.getcwd()+'/src/'+input_pickle_file_path)
                labels = utils.load_pickle(pickled_file_path=os.getcwd()+'/src/'+input_label_file_path)
                print(f"Loaded from {os.getcwd()+'/src/'+input_label_file_path}")

            print("feature dim: ", features[subject_id].shape)
            print("labels dim: ", labels[subject_id].shape)

            # print(features)
            # print(labels[subject_id])

            split_data = True
            while split_data:
                splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=None)

                for train_index, val_index in splitter.split(features[subject_id]):
                    continue

                # print(max(train_index), max(val_index), len(labels[subject_id]))

                train_l = labels[subject_id].iloc[train_index]
                val_l = labels[subject_id].iloc[val_index]

                # print(train_l)
                # unique, counts = np.unique(train_l, return_counts=True)
                # print(unique)

                # print(train_l, val_l, subject_id)

                split_data = check_val_is_subset(train_l, val_l) == False
                if split_data == False:
                    utils.logger.info(f'Successfully completed validation split for {subject_id}')

                 
            train_features = features[subject_id].iloc[train_index]
            val_features = features[subject_id].iloc[val_index]

            print(f"train feature dim: {train_features.shape}, val feature dim: {val_features.shape}")
            print(f"train label dim: {train_l.shape}, val feature dim: {val_l.shape}")


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

        print(train_labels)

        if SAVE_PICKLE_FILE:
            output = str(window_size) + 'ms' + INPUT_LABEL_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=train_labels,
                              file_path=output_pickle_file_path)
            
            output = str(window_size) + 'ms' + OUTPUT_LABEL_NAME
            output_pickle_file_path = os.path.join(OUTPUT_DIR, output)
            utils.pickle_data(data=val_labels,
                              file_path=output_pickle_file_path)