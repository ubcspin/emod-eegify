import torch.nn as nn
import torch

import numpy as np

from config import LABEL_CLASS_COUNT


class cnn_classifier(nn.Module):
    def __init__(self, dropout=0.2, n_labels=3):
        super(cnn_classifier, self).__init__()
        
        self.n_classes = n_labels


        self.cnn = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding='same', padding_mode='circular'), # loop the sides during convolution
            nn.ReLU()
        )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64* 64 * 8, self.n_classes)
            )
    
    def forward(self,x):
        x = self.cnn(x)
        x = self.classify(x) 
        return x

class cnn_classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        'Initialization'
        super(cnn_classifier_dataset, self).__init__()

        self.x = features # (N, n_bands, 64, 64)
        self.labels = labels # (N, 1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float() # (n_bands, 64, 64)
        y = torch.from_numpy(np.array(self.labels[index])).long() # feel trace labels int value [0,n_labels]
        return x, y


PARAMS = {
    'k_fold': 5,  # number of cross validation folds
    'num_classes': LABEL_CLASS_COUNT, # number of classes for output
    'dropout': 0.8, # dropout regularization to reduce over fitting
    'optimizer': 'AdamW', # optimizer for training the classifier
    'classifier_learning_rate': 1e-3, # AdamW learning rate
    'classifier_train_epochs': 30, # length of training epochs
    'batch_size': 128, # training batch size
    'weight_decay': 1e-4, # L2 regularization for AdamW optimizer
    'loss': 'cross_entropy' # loss function
}


