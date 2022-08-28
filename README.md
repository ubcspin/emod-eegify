# emod-eegify

Code for analyzing EEG collected during the EEG study. Protocol discussion: _Choose or Fuse: Enriching Data Views with Multi-label Emotion Dynamics. Cang et al. (ACII 2022)_


## File descriptions
- ```read_data.py```: utils to load CSV files and optionally save to a pickle file
- ```clean_data.py```: utils to clean up raw files, i.e., fix sampling to 1kHz, apply notch filter to eeg, apply bandpass to eeg, etc.
- ```calculate_features.py```: utils to calculate features from EEG data (2D Differential Entropy)
- ```config.py```: constant declarations
- ```train.py```: main training script 
- ```utils.py```: general purpose methods (pickle and load pickled files)


These are the set of modules used for classification of EEG signals using emotion labels

## Guide
1. unzip trial_data_split-anon.zip from the server into the a folder such as data/trial_data_split-anon (should contain .csv files)
2. Run the following functions from the terminal
    - ```read_data.py```: to generate pickle files (*subject_data.pk*)
    - ```clean_data.py```: to adjust the sampling rate to 1000 Hz, apply filters to eeg and generate merged pickle files (*merged_data.pk*)
    - ```calculate_features.py```: calculate the eeg features and save the features by participant, i.e *p06_featurized_data.pk*
    - ```calculate_labels.py```: calculate the set of labels and save them by participant, i.e *p06_labels.pk*

From the src directory run ```python3 generate_eeg_feeltrace.py``` or ```python generate_eeg_feeltrace.py``` on non unix machines. This will create the csv files for each subject and will take some time since the dataset is large (~5GB)
- The csv files containing the eeg and feeltrace signals are located in 'eeg_feeltrace' by default
