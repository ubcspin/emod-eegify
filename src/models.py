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


class lstm_classifier(nn.Module):
    def __init__(self, num_features=12, num_hidden=32, dropout=0.5, n_labels=5):
        super(lstm_classifier, self).__init__()
        
        self.input_size = num_features
        self.hidden_size = num_hidden
        self.n_classes = n_labels
        
        self.lstm_1 = nn.LSTM(
            input_size =  self.input_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first=True
        )
        
        
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, self.n_classes))

    
    def forward(self,x):
        x, (h_t, c_t) = self.lstm_1(x)

        # x -> (N, seq_len, hidden_size)
        # h -> (1, N, hidden_size)
        x = self.classify(h_t.squeeze(0))
        return x


# autoencoder model
# input: (N, 64)
# latent features: Z
# encoder: (N,64) -> (N,32) -> (N, Z)
# decoder: (N,Z) -> (N,16) -> (N, 64)

class autoencoder(nn.Module):
    def __init__(self, num_features=12):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_features))
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x


class lstm_classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        'Initialization'
        self.x = features # (N, window_size, encoding)
        self.labels = labels # (N, 1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float()
        y = torch.from_numpy(np.array(self.labels[index])).long()
        return x, y

class autoencoder_dataset(torch.utils.data.Dataset):
    def __init__(self, features):
        'Initialization'
        self.x = features # (N, window_size, 64)
        #self.x = self.x.reshape(self.x.shape[0], 64, 1, -1) # (N, 64, 1, window_size) -> (N, C, H, W)
        self.x = self.x.reshape(-1, 64)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float()
        y = x

        return x, y




PARAMS = {
    'k_fold': 5,  # number of cross validation folds
    'num_classes': LABEL_CLASS_COUNT, # number of classes for output
    'dropout': 0.8, # dropout regularization to reduce over fitting
    'lstm_dropout': 0.5, # dropout regularization to reduce over fitting
    'optimizer': 'AdamW', # optimizer for training the classifier
    'classifier_learning_rate': 1e-3, # AdamW learning rate
    'classifier_train_epochs': 30, # length of training epochs
    'batch_size': 128, # training batch size
    'weight_decay': 1e-4, # L2 regularization for AdamW optimizer
    'loss': 'cross_entropy', # loss function
    'encoder_train_epochs': 15, 
    'encoder_features': 12, # number of features in latent space
    'encoder_split': 0.4 # % of training data to use for training the encoder
}
