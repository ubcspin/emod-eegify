# emod-eegify

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
├── src
```

## Getting Started
1. unzip feel.zip and the subsequent zip files within: File structure should look like feel/p* where * is an integers(should contain .csv files)
2. run setup.py from the *src* directory to prepare the dataset for the model (ensure you have the correct path for the feel folder)
3. run the train_hc.py from the *src* directory to run hyperparameter search
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