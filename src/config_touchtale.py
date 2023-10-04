TIME_INDEX = 'timedelta'
# TIME_INTERVAL = '500ms'

MIN_WINDOW_SIZE = 2000

TIME_INTERVAL = str(MIN_WINDOW_SIZE) + 'ms'

MAX_CONTINUOUS_ANNOTATION = 225

FS = 50 # sampling frequency in Hz
SAMPLE_PERIOD = '1ms'

WINDOW_TYPE = 'hamming'
# WINDOW_SIZE = 500 # ms
WINDOW_SIZE = MIN_WINDOW_SIZE # ms

COLLAPSED_PARTICIPANTS = True

LABEL_CLASS_COUNT = 4

OTHER_FEATURES_ONLY = 0
TOUCH_FEATURES_ONLY = 1
ALL_FEATURES_INCLUDED = 2

MODE = OTHER_FEATURES_ONLY


import os
if COLLAPSED_PARTICIPANTS:
  SUBJECT_IDS = ['p_all']
else:
  SUBJECT_IDS = [x[:3] if 'p0' in x and 'cleaned' in x else '' for x in os.listdir('COMBINED_DATA_TOUCHTALE/')]
  SUBJECT_IDS = list(filter(lambda a: a != '', SUBJECT_IDS))
  SUBJECT_IDS = sorted(SUBJECT_IDS)
print(SUBJECT_IDS)

N_FEATURES = 45
if MODE == TOUCH_FEATURES_ONLY:
  N_FEATURES = 6
elif MODE == OTHER_FEATURES_ONLY:
  N_FEATURES = 40
elif MODE == ALL_FEATURES_INCLUDED:
  N_FEATURES = 45

N_SAMPLES = 8
 
DEBUG = True

EXP_PARAMS = {'LABEL_TYPE': ['pos', 'acc', 'angle'],
              'WINDOW_SIZE': [MIN_WINDOW_SIZE, 5000], # ms
            #   'WINDOW_SIZE': [500, 5000], # ms
              'WINDOW_OVERLAP': [0., 0.25, 0.5] # percentage
              }

CV_SPLITS = 5