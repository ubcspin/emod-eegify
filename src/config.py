TIME_INDEX = 'timedelta'
TIME_INTERVAL = '500ms'

MAX_FEELTRACE = 225

FS = 1000
SAMPLE_PERIOD = '1ms'

WINDOW_TYPE = 'hamming'
WINDOW_SIZE = 500 # ms

LABEL_CLASS_COUNT = 3
 
DEBUG = True

EXP_PARAMS = {'LABEL_TYPE': ['pos', 'acc', 'angle'],
              'WINDOW_SIZE': [500, 5000], # ms
              'WINDOW_OVERLAP': [0., 0.25, 0.5] # percentage
              }

CV_SPLITS = 5