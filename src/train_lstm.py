import os

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import lstm_classifier, lstm_classifier_dataset, autoencoder, autoencoder_dataset, PARAMS

from tqdm import tqdm

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'


SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']
INPUT_FEATURE_NAME = 'featurized_data.npy'
INPUT_LABEL_NAME = 'labels.pk'


SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'eeg_lstm_results.pk'

LABEL_TYPES = ['pos', 'angle']



def train_encoder(model, features=None, params=PARAMS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['classifier_learning_rate'], weight_decay=params['weight_decay'])
    else:
        raise ValueError(f'Unexpected value for optimizer parameter: {params}')

    criterion = nn.MSELoss()
    
    train_dataset = autoencoder_dataset(features)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=params['batch_size'],
                                               num_workers=8,
                                               shuffle=True)
    
    train_metrics = []
    for epoch in range(params['encoder_train_epochs']):
        utils.logger.info(f'Starting {epoch}')
        # reset metrics
        cur_train_loss = 0 # loss
        
        # set to train mode
        model.train()
        
        # loop over dataset
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            cur_train_loss += loss.detach().cpu()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_loss = cur_train_loss/train_loop_size
        utils.logger.info(f'Current Encoder Loss {cur_train_loss}')
        train_metrics.append([cur_train_loss])
        
    return train_metrics


def train_lstm(model, features=None, labels=None, params=PARAMS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['classifier_learning_rate'], weight_decay=params['weight_decay'])
    else:
        raise ValueError(f'Unexpected value for optimizer parameter: {params}')
    
    if params['loss'] == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unexpected value for loss function: {params}')

    train_dataset = lstm_classifier_dataset(features, labels)
    
    # figure out class distribution to over sample less represented classes

    train_labels = labels
    
    # get the weights of each class as 1/occurrence
    train_class_weight = np.bincount(train_labels, minlength=params['num_classes'])
    utils.logger.info(f'Training label distribution {train_class_weight}')

    train_class_weight = 1/train_class_weight
    
    # get the per sample weight, which is the likelihood os sampling
    train_sample_weights = [train_class_weight[x] for x in train_labels]
    
    # sampler, weighted by the inverse of the occurrence
    train_sampler = torch.utils.data.WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=params['batch_size'],
                                               num_workers=8,
                                               sampler=train_sampler)
    
    train_metrics = []
    for epoch in range(params['classifier_train_epochs']):
        utils.logger.info(f'Starting {epoch}')
        # reset metrics
        cur_train_acc = 0 # accuracy
        cur_train_pc = 0 # precision
        cur_train_rc = 0 # recall
        cur_train_f1 = 0 # f1
        cur_train_loss = 0 # loss
        
        # set to train mode
        model.train()
        
        # loop over dataset
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            y_hat_np = F.softmax(y_hat.detach(), dim=1).argmax(axis=1).cpu().numpy().squeeze().reshape(-1,) # predictions
            y_np = y.detach().cpu().numpy().squeeze().reshape(-1,) # labels
            
            # metrics
            prf = precision_recall_fscore_support(y_np, y_hat_np, average='macro', zero_division=0)
            
            cur_train_acc += np.mean(y_hat_np == y_np)
            cur_train_pc += prf[0]
            cur_train_rc += prf[1]
            cur_train_f1 += prf[2]
            cur_train_loss += loss.detach().cpu()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_acc  = cur_train_acc/train_loop_size
        cur_train_pc   = cur_train_pc/train_loop_size
        cur_train_rc   = cur_train_rc/train_loop_size
        cur_train_f1   = cur_train_f1/train_loop_size
        cur_train_loss = cur_train_loss/train_loop_size
        
        utils.logger.info(f'Current Classifier Loss {cur_train_loss}')
        utils.logger.info(f'Current Classifier f1 {cur_train_f1}')
        train_metrics.append([cur_train_acc, cur_train_pc, cur_train_rc, cur_train_f1, cur_train_loss])
        
    return train_metrics

def test(model, features=None, labels=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()
        x  = torch.from_numpy(features).float().to(device)
        y = labels
        y_hat = model(x)
        y_hat = F.softmax(y_hat.detach(), dim=-1).argmax(dim=-1).cpu().numpy()
    return y, y_hat


def encode_classifier_data(encoder_model, classifier_features, params=PARAMS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.eval()
    with torch.no_grad():
        prev_shape = classifier_features.shape # (N, window_size, 64)
        x = torch.from_numpy(classifier_features).float().reshape(-1,64).to(device) # encode the 64 channels (Nxwindow_size, 64)
        x_encoded = encoder_model.encode(x).reshape(prev_shape[0], prev_shape[1], params['encoder_features']).detach().cpu().numpy()
        return x_encoded


def train_participant(features, label_dict, params=PARAMS, label_types=LABEL_TYPES):
    results = {}

    for label_type in label_types: # train every label type
        utils.logger.info(f'Training label type {label_type}')
        labels = label_dict[label_type].to_numpy()
        #split data into train/test indices using kFold validation
        indices = utils.split_dataset(labels, k=params['k_fold'])
        fold_results = {}
        for fold in range(params['k_fold']): # with k cross validation
            utils.logger.info(f'Running fold number {fold}')
            train_index, test_index = indices[fold]


            lstm_train_X, encoder_X, lstm_y, _ = train_test_split(features[train_index], labels[train_index], test_size=params['encoder_split']) 

            encoder_model =  autoencoder(num_features=params['encoder_features'])
            classifier_model = lstm_classifier(num_features=params['encoder_features'], dropout=params['lstm_dropout'], n_labels=params['num_classes'])

            train_encoder(encoder_model, features=encoder_X)
            encoded_features_train = encode_classifier_data(encoder_model, lstm_train_X, params=params)

            train_lstm(classifier_model, features=encoded_features_train, labels=lstm_y, params=params)
            
            encoded_features_test = encode_classifier_data(encoder_model, features[test_index], params=params)

            y, y_hat = test(classifier_model, features=encoded_features_test, labels=labels[test_index])

            fold_results[str(fold)] = {'y': y, 'y_hat': y_hat}
        results[label_type] = pd.DataFrame(fold_results)
    return results


if __name__ == '__main__':
    if INPUT_PICKLE_FILE:
        participant_results = {}
        for subject_id in tqdm(SUBJECT_IDS):
            utils.logger.info(f'Training participant {subject_id}')

            input_feature_file_path = os.path.join(INPUT_DIR, subject_id + "_" + INPUT_FEATURE_NAME)
            input_label_file_path = os.path.join(INPUT_DIR, subject_id + "_" + INPUT_LABEL_NAME)
            
            features = utils.load_numpy(file_path=input_feature_file_path)
            labels = utils.load_pickle(pickled_file_path=input_label_file_path)

            result = train_participant(features, labels[subject_id])
            participant_results[subject_id] = result

        if SAVE_PICKLE_FILE:
            output_pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
            utils.pickle_data(data=participant_results,
                          file_path=output_pickle_file_path)
