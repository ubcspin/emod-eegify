import torch.nn as nn
import torch

from skorch import NeuralNetClassifier

import numpy as np

from config import LABEL_CLASS_COUNT
from config import EXP_PARAMS


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
BATCH_SIZE = 256
DROPOUT = 0.25

# device
DEVICE = 'cuda' if not torch.cuda.is_available() else 'cpu'

def get_model(batch_size, lr, max_epochs, dropout, n_labels):
    return NeuralNetClassifier(module=cnn_classifier_original,
        module__dropout=dropout,
        module__n_labels=n_labels,
        optimizer=OPTIMIZER,
        lr=lr,
        max_epochs=max_epochs,
        criterion=CRITERION,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        train_split=None,
        device=DEVICE,
        callbacks=[
                MLflowLogger() # log metrics to mlflow
                ],
        verbose=0)

hps = pd.read_csv('hyperparams.csv')
SUBJECT_IDS = ['p02', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p12', 'p13', 'p15', 'p17', 'p19', 'p20', 'p22', 'p23']

MODELS = {}
for window_size in EXP_PARAMS["WINDOW_SIZE"]:
        for subject_id in SUBJECT_IDS:
            id = subject_id + "_" + str(window_size) + "ms"
            batch_size = hps[id]['local_classifier__batch_size']
            lr = hps[id]['local_classifier__lr']
            max_epochs = hps[id]['local_classifier__max_epochs']
            MODELS[id] = {"CNN" : get_model(batch_size, lr, epochs, DROPOUT, LABEL_CLASS_COUNT)}



PARAMS = {

    'CNN': {
    'local_classifier__batch_size': [128, 256, 512],
    'local_classifier__lr': [LR*0.1, LR,  LR*10],
    'local_classifier__max_epochs' : [MAX_EPOCHS,  MAX_EPOCHS*2, MAX_EPOCHS*4]
    }
}

