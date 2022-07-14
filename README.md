# Emotion-Modelling EEGify

These are the set of modules used for classification of EEG signals using emotion labels

# Guide
- unzip trial_data_split-anon.zip from the server into the a folder called participant folder (participant/p1 participant/p2 etc should be visible)
- Run the ```convert_string_to_char_fix.sh``` to load the scene data correctly in python
- From the src directory run ```python3 generate_feeltrace.py``` or ```python generate_feeltrace.py``` on non unix machines. This will create the csv files for each subject and will take some time since the dataset is large (~5GB)
- The csv files containing the feeltrace signals are located in 'feeltrace' by default

# Notes
    - The feeltrace signals are sampled at ~30Hz but other signals, for example, EEG are sampled at 1000Hz. Further processing is required to account for this