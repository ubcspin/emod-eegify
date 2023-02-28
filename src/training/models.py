import torch.nn as nn
import torch

from skorch import NeuralNetClassifier

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

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {
    'CNN': NeuralNetClassifier(module=cnn_classifier,
        module__dropout=0.25,
        module__n_labels=LABEL_CLASS_COUNT,
        optimizer=OPTIMIZER,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        criterion=CRITERION,
        batch_size=BATCH_SIZE,
        iterator_train__shuffle=True,
        train_split=None,
        device=DEVICE,
        verbose=0)
}

PARAMS = {

    'CNN': {
    'local_classifier__lr': [LR*0.1, LR,  LR*10],
    'local_classifier__max_epochs' : [MAX_EPOCHS,  MAX_EPOCHS*2, MAX_EPOCHS*4]
    }
}