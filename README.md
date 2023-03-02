# emod-eegify

Code for analyzing EEG collected during the EEG study. Protocol discussion: _Choose or Fuse: Enriching Data Views with Multi-label Emotion Dynamics. Cang et al. (ACII 2022)_



The dataset consists of comma separated value (.csv) files organized by participant. For each participant there are four csv files: brain activity, keypress, calibrated words and joystick data. The structure looks something like the following:
```
├── feel
│   ├── p10
│   │   ├── calibrated_words.csv
│   │   ├── eeg.csv
│   │   ├── fsr.csv
│   │   └── joystick.csv
│   ├── p12
│   │   ├── calibrated_words.csv
│   │   ├── eeg.csv
│   │   ├── fsr.csv
│   │   └── joystick.csv

```

## Getting Started
1. unzip feel.zip and the subsequent zip files within: File structure should look like feel/p* where * is an integers(should contain .csv files)
2. run setup.py from the src directory to prepare the dataset for the model (ensure you have the correct path for the feel folder)
3. run the train_hc.py from the src directory to run hyperparameter search
4. run ```hyperparameter_results.py``` to create a csv of the hyperparameters in the validation folder
5. run ```val_hc.py``` to validate the model on unseen data
6. run ```validation_results.py``` to generate the validation results in the validation folder


## File descriptions
- ```read_data.py```: utils to load CSV files and optionally save to a pickle file
- ```clean_data.py```: utils to clean up raw files, i.e., fix sampling to 1kHz, apply notch filter to eeg, apply bandpass to eeg, etc.
- ```calculate_features.py```: utils to calculate features from EEG data (2D Differential Entropy)
- ```calculate_labels.py```: utils to calculate labels
- ```calculate_validation.py```: util to calculate train/validation split (train is later further split into train\test)
- ```config.py```: constant declarations
- ```train_hc.py```: main training script 
- ```utils.py```: general purpose methods (pickle and load pickled files)


These are the set of modules used for classification of EEG signals using emotion labels

## Guide
1. unzip fell.zip into the a folder such feel (should contain .csv files)
2. Run the following functions from the terminal in the src directory
    - ```read_data.py```: to generate pickle files (*subject_data.pk*)
    - ```clean_data.py```: to adjust the sampling rate to 1000 Hz, apply filters to eeg and generate merged pickle files (*merged_data.pk*)
    - ```calculate_features.py```: calculate the eeg features and save the features by participant, i.e *p06_featurized_data.pk*
    - ```calculate_labels.py```: calculate the set of labels and save them by participant, i.e *p06_labels.pk*
    - ```calculate_validation.py```: calculate the validation\train split, i.e *p06_val_featurized_data.pk*
    - ```train_hc.py```: read through every *_featurized_data.pk* and *_labels.pk* and train an independent model for each participant and save the results in *_results.pk*
