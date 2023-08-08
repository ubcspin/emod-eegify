import torch.nn as nn
import torch

from skorch import NeuralNetClassifier

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier


import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
from config_touchtale import TIME_INDEX, TIME_INTERVAL, WINDOW_SIZE, EXP_PARAMS, FS, LABEL_CLASS_COUNT


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
        x = torch.softmax(x, dim=1)
        return x

class cnn_classifier_original(nn.Module):
    def __init__(self, dropout=0.2, n_labels=3):
        super(cnn_classifier_original, self).__init__()
        
        self.n_classes = n_labels


        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout),
            nn.Conv2d(32, 16, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16* 16 * 16, self.n_classes)
            )
    
    def forward(self,x):
        x = self.cnn(x)
        x = self.classify(x)
        x = torch.softmax(x, dim=1)
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


# model hyper-parameters
OPTIMIZER = torch.optim.AdamW
LR = 1e-4
MAX_EPOCHS = 5
CRITERION = nn.CrossEntropyLoss
BATCH_SIZE = 2048
DROPOUT = 0.25

# device
DEVICE = 'cuda' if not torch.cuda.is_available() else 'cpu'

MODELS = {
    # 'CNN': NeuralNetClassifier(module=cnn_classifier_original,
    #     module__dropout=DROPOUT,
    #     module__n_labels=LABEL_CLASS_COUNT,
    #     optimizer=OPTIMIZER,
    #     lr=LR,
    #     max_epochs=MAX_EPOCHS,
    #     criterion=CRITERION,
    #     batch_size=BATCH_SIZE,
    #     iterator_train__shuffle=True,
    #     train_split=None,
    #     device=DEVICE,
    #     verbose=0)
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier(),
    'SVC': SVC()  # takes a long time to train with linear kernels
}

PARAMS = {

    # 'CNN': {
    # 'local_classifier__batch_size': [128, 256, 512],
    # 'local_classifier__lr': [LR*0.1, LR,  LR*10],
    # 'local_classifier__max_epochs' : [MAX_EPOCHS,  MAX_EPOCHS*2, MAX_EPOCHS*4]
    # }
    'ExtraTreesClassifier': {'n_estimators': [16, 32]},
    'RandomForestClassifier': {'n_estimators': [16, 32]},
    'AdaBoostClassifier':  {'n_estimators': [16, 32]},
    'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
    'XGBClassifier': {'max_depth': (4, 6, 8), 'min_child_weight': (1, 5, 10)},
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ],
}