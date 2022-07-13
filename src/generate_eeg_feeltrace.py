from utils import create_dataset
import os

if __name__ == '__main__':
    create_dataset(os.path.join('..', 'participant'), os.path.join('..', 'eeg_feeltrace'), num_workers=4)