# Emotion-Modelling EEGify

These are the set of modules used for classification of EEG signals using emotion labels

# Guide
- unzip trial_data_split-anon.zip from the server into the a folder called participant folder (participant/p1 participant/p2 etc should be visible)
- From the src directory run ```python3 generate_eeg_feeltrace.py``` or ```python generate_eeg_feeltrace.py``` on non unix machines. This will create the csv files for each subject and will take some time since the dataset is large (~5GB)
- The csv files containing the eeg and feeltrace signals are located in 'eeg_feeltrace' by default

# Notes
    - The feeltrace signals are resampled at 1kHz to match the sampling frequency of the eeg signal
    - To load 'scenes.mat' in python, run the ```convert_string_to_char_fix.sh``` to load the scene data correctly in python