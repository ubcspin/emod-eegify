import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import os
import csv
import time
import sklearn

from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil import parser
from scipy.signal import savgol_filter
from ydata_profiling import ProfileReport
from arabica import arabica_freq, cappuccino, coffee_break
from nltk.tokenize import RegexpTokenizer
from scipy.signal import argrelextrema, find_peaks


class TouchtalesPipeline(): 
    def __init__(self, dataset_name, task_one_ds, task_two_ds, task_three_ds, task_four_ds, task_five_ds, transcript_ds, start_timestamp, video_start, profiling_report_dir, output_dir):
        self.dataset_name = dataset_name
        self.task_one_ds = task_one_ds              # Path to data source for task 1 data
        self.task_two_ds = task_two_ds              # Path to data source for task 2 data
        self.task_three_ds = task_three_ds          # Path to data source for task 3 data
        self.task_four_ds = task_four_ds            # Path to data source for task 4 data
        self.task_five_ds = task_five_ds            # Path to data source for task 5 data
        self.transcript_ds = transcript_ds          # Path to data source for transcript
        self.start_timestamp = start_timestamp      # Start timestamp for data collection
        self.video_start = video_start              # Start timestamp for video
        self.merged_data = pd.DataFrame()
        self.profiling_report_dir = profiling_report_dir    # Directory for output profiling report
        self.output_dir = output_dir                # Directory for output csv
        self.calibration_range = [450, -50]
        self.feeltrace_range = [700, 400]

        path = Path(self.output_dir)
        # print(path.absolute())
        Path(path.parent.absolute()).mkdir(parents=True, exist_ok=True)

    # Main function for cleaning and merging data streams
    def clean_data(self): 
        self.get_task_one()
        self.get_task_two()
        self.get_task_five()
        self.get_task_four()
        self.get_task_three()
        
        self.convert_transcript()
        self.synchronize_transcript()
        self.merge_data()
        # profile = ProfileReport(self.merged_data, title="Touchtales Profiling Report")
        # profile.to_file(self.profiling_report_dir)
        self.merged_data.to_csv(self.output_dir)

        # self.merged_data.set_index('timestamp').plot()
        # plt.xlabel("Timestamp")
        # plt.ylabel("Biosignal Value")
        # plt.title("Biosignal Data Over Time") 
        # plt.show()
        
        return
    
    # Return a cleaned dataframe for task one data (biosignal)
    def get_task_one(self):
        with open(self.task_one_ds, "r") as file: 
            data = file.read().rstrip()
        
        data = json.loads(data)
        self.task_one_df = pd.DataFrame(data)
        self.task_one_df['timestamp'] = pd.to_datetime(self.task_one_df["timestamp"], unit='ms')
        self.clean_task_one()
        return
    
    # Clean task one data, return cleaned dataframe as a class variable, line up to start timestamp
    def clean_task_one(self):
        # Smoothing task 1 biosignals
        self.task_one_df["gsr_smoothed"] = savgol_filter(self.task_one_df["gsr"], 100, 1)
        self.task_one_df["bpm_smoothed"] = savgol_filter(self.task_one_df["bpm"], 100, 1)
        pds = pd.Series(~(self.task_one_df['timestamp'] < self.start_timestamp))
        self.task_one_df = self.task_one_df[~(self.task_one_df['timestamp'] < self.start_timestamp)]
        self.task_one_df = self.task_one_df.drop(columns=['resp', 'feeltrace', 'comment'])
        return


    def plot_uncleaned_task_one(self):
        print(len(self.task_one_df["bpm"]), len(self.task_one_df["bpm_smoothed"]))
        plt.plot(self.task_one_df["bpm"])
        plt.plot(self.task_one_df["bpm_smoothed"])
        plt.show()
        return


    # Get cleaned task two data
    def get_task_two(self):
        with open(self.task_two_ds, "r") as file: 
            data = file.read().rstrip()
        
        data = json.loads(data)
        self.task_two_df = pd.DataFrame(data[:-1])[["label", "y"]]
        axis = data[-1]["axis"] # First item is higher, second item is lower
        row_high_axis = {'label': axis[0], 'y': self.calibration_range[0]}
        row_low_axis = {'label': axis[1], 'y': self.calibration_range[1]}
        self.task_two_df = pd.concat([self.task_two_df, pd.DataFrame([row_high_axis])], ignore_index=True)
        self.task_two_df = pd.concat([self.task_two_df, pd.DataFrame([row_low_axis])], ignore_index=True)
        self.task_two_df = self.task_two_df.sort_values(by = ["y"], ascending=True)
        return
    
    # Get cleaned task three data (interview)
    def get_task_three(self):
        with open(self.task_three_ds, 'r') as file:
            data = file.read().rstrip()

        data = json.loads(data)
        self.task_three_df = pd.DataFrame(data)
        self.task_three_df['timestamp'] = pd.to_timedelta(self.task_three_df['timestamp'], unit='s', errors='coerce')

        self.clean_task_three()
        return
    
    # Clean task three data, return cleaned data as a class variable, line up to video timestamp
    def clean_task_three(self):
        self.task_three_df = self.task_three_df[~(self.task_three_df['timestamp'] < self.video_start)]
        self.task_three_df['timestamp'] = self.task_three_df.apply(lambda x: x['timestamp'] + self.start_timestamp, axis=1)
        return
    
    # Get cleaned task four data (feeltrace)
    def get_task_four(self):
        with open(self.task_four_ds, 'r') as file:
            data = file.read().rstrip()

        data = json.loads(data)
        self.task_four_df = pd.DataFrame(data)
        self.task_four_df = self.task_four_df.drop(columns=['touch', 'bpm', 'gsr', 'comment', 'resp'])
        self.task_four_df['timestamp'] = pd.to_datetime(self.task_four_df["timestamp"], unit='ms', errors='coerce')
        self.clean_task_four()

        self.feeltrace_range = [self.task_four_df["feeltrace_cleaned"].max(), self.task_four_df["feeltrace_cleaned"].min()]
        self.add_calibration_to_feeltrace(self.task_two_df, "calibration_1")
        self.add_calibration_to_feeltrace(self.task_five_df, "calibration_2")
        return
    
    def add_calibration_to_feeltrace(self, task_df, col_name):
        feeltrace_span = self.feeltrace_range[0] - self.feeltrace_range[1]
        calibration_span = self.calibration_range[0] - self.calibration_range[1]
        task_df["rescaled_df"] = self.feeltrace_range[1] + ((task_df["y"] - self.calibration_range[1])/calibration_span)*feeltrace_span
        # print(task_df)
        self.task_four_df[col_name] = self.task_four_df.apply(lambda z: task_df["label"][min(enumerate(list(task_df["y"])), key=lambda x: abs(x[1]-z["feeltrace_cleaned"]))[0]], axis=1)
        return
    
    # Clean tasxk four data (feeltrace)
    def clean_task_four(self):
        df = self.task_four_df
        delta = 0.03
        span = 10
        df['y_ewma_fb'] = self.ewma_fb(df['feeltrace'], span)
        df['y_remove_outliers'] = self.remove_outliers(df['feeltrace'].tolist(), df['y_ewma_fb'].tolist(), delta)
        df['feeltrace_cleaned'] = df['y_remove_outliers'].interpolate()
           
        self.task_four_df = df.drop(columns=["y_ewma_fb", "y_remove_outliers"])
        min_time = self.task_four_df["timestamp"].min()
        self.task_four_df = self.task_four_df[~(self.task_four_df['timestamp'] < min_time + self.video_start)]
        min_time = self.task_four_df["timestamp"].min()
        time_delta = min_time - self.start_timestamp
        self.task_four_df['timestamp'] = self.task_four_df.apply(lambda x: x['timestamp'] - time_delta, axis=1)
        
        return 

    # Apply forwards, backwards exponential weighted moving average (EWMA) to a column in a dataframe (df_column)
    def ewma_fb(self, df_column, span):
        # Forwards EWMA.
        fwd = pd.Series.ewm(df_column, span=span).mean()
        # Backwards EWMA.
        bwd = pd.Series.ewm(df_column[::-1],span=10).mean()
        # Add and take the mean of the forwards and backwards EWMA.
        stacked_ewma = np.vstack(( fwd, bwd[::-1] ))
        fb_ewma = np.mean(stacked_ewma, axis=0)
        return fb_ewma
        
    # Remove data from noisy column that is > delta from fbewma. 
    def remove_outliers(self, noisy, fbewma, delta):
        np_noisy = np.array(noisy)
        np_fbewma = np.array(fbewma)
        cond_delta = (np.abs(np_noisy-np_fbewma) > delta)
        np_remove_outliers = np.where(cond_delta, np.nan, np_noisy)
        return np_remove_outliers
            
    # Get cleaned task five data
    def get_task_five(self):
        with open(self.task_five_ds, "r") as file: 
            data = file.read().rstrip()
        
        data = json.loads(data)
        self.task_five_df = pd.DataFrame(data[:-1])[["label", "y"]]
        axis = data[-1]["axis"] # First item is higher, second item is lower
        row_high_axis = {'label': axis[0], 'y': self.calibration_range[0]}
        row_low_axis = {'label': axis[1], 'y': self.calibration_range[1]}
        self.task_five_df = pd.concat([self.task_five_df, pd.DataFrame([row_high_axis])], ignore_index=True)
        self.task_five_df = pd.concat([self.task_five_df, pd.DataFrame([row_low_axis])], ignore_index=True)
        self.task_five_df = self.task_five_df.sort_values(by = ["y"], ascending=True)
        
        return

    # Convert transcript to a readable csv
    def convert_transcript(self):
        input_vtt = self.transcript_ds
        opened_file = open(input_vtt, encoding='utf8')
        content = opened_file.read()
        segments = content.split('\n\n') # split on double line
        # wrangle segments
        m = re.compile(r"\<.*?\>") # strip/remove unwanted tags
        new_segments = [self.clean_transcript_line(s, m) for s in segments if len(s)!=0][1:]

        trimmed_segments = []
        for segment in new_segments:
            split_segment = segment.split()
            time_code = split_segment[0]
            text = ' '.join(segment.split()[1:])
            trimmed_segment = (time_code, str(text[:12]), str(text[13:24]), str(text[25:]))
            trimmed_segments.append(trimmed_segment)
        
        # Add trimmed segments to csv
        with open(str(input_vtt)[:-3]+'csv', 'w', encoding='utf8', newline='') as f:
            for line in trimmed_segments:
                thewriter = csv.writer(f)
                thewriter.writerow(line)

        self.transcript_ds = self.transcript_ds[:-3] + 'csv'
        return

    # Clean a single line of the transcript file
    def clean_transcript_line(self, content, m):
        new_content = m.sub('',content)
        # new_content = o.sub('',new_content)
        new_content = new_content.replace('align:start position:0%','')
        new_content = new_content.replace('-->','')
        return new_content
    
    # trim time codes for g suite plain text formatting conversion to seconds w/ formula '=value(str*24*3600)'
    def clean_time(time):
        time = time.split(':')
        if time[0]=='00':
            return time[1]+':'+time[2]
        if not time[0]=='00':
            return time[0]+':'+time[1]+':'+time[2]

    # Synchronize transcript CSV with preexisting biosignal and feeltrace data
    def synchronize_transcript(self):
        colnames = ["time_id", "transcript_timestamp_start", "transcript_timestamp_finish", "transcript"]
        self.transcript_df = pd.read_csv(self.transcript_ds, names = colnames, header = None)
        # self.transcript_df["duration"] = self.transcript_df.apply(lambda x: str(x["Duration"][0:-3] + "." + x["Duration"][-2:]), axis=1)
        self.transcript_df["transcript_timestamp_start"] = pd.to_timedelta(self.transcript_df['transcript_timestamp_start'])
        self.transcript_df["transcript_timestamp_finish"] = pd.to_timedelta(self.transcript_df['transcript_timestamp_finish'])
        self.transcript_df["transcript_duration"] = (self.transcript_df['transcript_timestamp_finish']- self.transcript_df['transcript_timestamp_start']).fillna(pd.Timedelta(0))
        self.transcript_df = self.transcript_df[~(self.transcript_df['transcript_timestamp_start'] < self.video_start)]
        self.transcript_df['timestamp'] = self.transcript_df.apply(lambda x: x['transcript_timestamp_start'] + self.start_timestamp, axis=1)
        self.transcript_df['transcript_timestamp_finish'] = self.transcript_df.apply(lambda x: x['transcript_timestamp_finish'] + self.start_timestamp, axis=1)
        # self.transcript_df["transcript_timestamp_start"] = self.transcript_df['timestamp']

        self.transcript_df = self.transcript_df.drop(columns=["transcript_timestamp_start", "time_id"])
        return
    
    def merge_data(self): 
        self.merged_data = pd.merge_asof(self.task_one_df, self.task_four_df, on="timestamp", tolerance=pd.Timedelta('1s'))
        self.merged_data = pd.merge_asof(self.merged_data, self.task_three_df, on="timestamp", tolerance=pd.Timedelta('1s'))
        self.merged_data = pd.merge_asof(self.merged_data, self.transcript_df, on="timestamp", tolerance=pd.Timedelta('1s'))
        self.merged_data = self.merged_data[self.merged_data['feeltrace'].notna()]
        self.merged_data["timestamp"] = self.merged_data["timestamp"].apply(lambda x: datetime.timestamp(x))
        # self.merged_data = self.merged_data['transcript_tokenized']
        print(self.merged_data.columns)
    
        return
    

DATASETS_PATH = "/Users/poyuchen/Desktop/UBC/Engineering-Physics/Fifth-Year/Summer/SPIN/TouchTales-Data-Analysis-main/raw_data/"

class TouchtalesDataAnalysis:
    def __init__(self, sess_id=0):
        # PRESET VARIABLES FOR SAMPLE DATASETS IN ONEDRIVE
        # =================================================================
        
        dataset_paths = [x[0] for x in os.walk(DATASETS_PATH)]
        dataset_paths = dataset_paths[1:]

        self.datasets = {}
        for dataset_path in dataset_paths:
            if 'other' in dataset_path:
                continue
            self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
            dataset_name = self.file_paths[0].split('/')[2]
            start_info = self.get_start_info(dataset_path)
            info = {
                'dataset_name': dataset_name,
                'date': dataset_name[-4:],
                'task_one_ds': self.get_name_containing_substring('TaskOne'),
                'task_two_ds': self.get_name_containing_substring('TaskTwo'),
                'task_three_ds': self.get_name_containing_substring('TaskThree'),
                'task_four_ds': self.get_name_containing_substring('TaskFour'),
                'task_five_ds': self.get_name_containing_substring('TaskFive'),
                'transcript_ds': self.get_name_containing_substring('.vtt'),
                'start_timestamp': start_info[1].strip(),
                'video_start': str(int(start_info[0].strip())),
                'output_dir': '../cleaned/' + dataset_name + '/cleaned_data.csv',
                'analyzed_data': '../cleaned/' + dataset_name + '/analyzed_data.csv'
            }
            self.datasets[dataset_path] = info

        self.datasets_sorted = OrderedDict(sorted(self.datasets.items(), key=lambda x: x[1]['date']))

        self.dataset_dict = self.datasets_sorted[list(self.datasets_sorted.keys())[sess_id]]
        print(json.dumps(self.dataset_dict, indent=4))

        self.dataset_name = self.dataset_dict['dataset_name']
        self.task_one_ds = self.dataset_dict['task_one_ds']              # Path to data source for task 1 data
        self.task_two_ds = self.dataset_dict['task_two_ds']              # Path to data source for task 2 data
        self.task_three_ds = self.dataset_dict['task_three_ds']          # Path to data source for task 3 data
        self.task_four_ds = self.dataset_dict['task_four_ds']           # Path to data source for task 4 data
        self.task_five_ds = self.dataset_dict['task_five_ds']            # Path to data source for task 5 data
        self.transcript_ds = self.dataset_dict['transcript_ds']          # Path to data source for transcript
        self.start_timestamp = datetime.strptime(self.dataset_dict['start_timestamp'], '%Y-%m-%d %H:%M:%S.%f')      # Start timestamp for data collection 
        self.video_start = timedelta(seconds=int(self.dataset_dict['video_start']))             # Start timestamp for video (in s)
        self.output_dir = self.dataset_dict['output_dir']
        self.analyzed_data = self.dataset_dict['analyzed_data']
        self.profiling_report_dir = ""

    def get_name_containing_substring(self, substring):
        for f in self.file_paths:
            if substring in f:
                return f

    def get_start_info(self, data_path):
        with open(os.path.join(data_path, 'time.txt')) as f:
            lines = f.read().splitlines()
            
        return lines[0].split(',')

tda = TouchtalesDataAnalysis()

if __name__ == "__main__":
    sess_id = 0
    tda = TouchtalesDataAnalysis(sess_id=sess_id)
    pipeline = TouchtalesPipeline(tda.dataset_name, tda.task_one_ds, tda.task_two_ds, tda.task_three_ds, tda.task_four_ds, 
                                tda.task_five_ds, tda.transcript_ds, tda.start_timestamp, tda.video_start, tda.profiling_report_dir, tda.output_dir)
    pipeline.clean_data()

    _, _, touch, timestamp, flag, gsr, bpm = pipeline.task_one_df.T.to_numpy()

    start = timestamp[0].to_pydatetime()
    timestamp_aligned = [(x.to_pydatetime() - start).total_seconds() for x in timestamp.tolist()]

    print(np.asarray(touch).shape, timestamp.shape, flag.shape, gsr.shape, bpm.shape)
    print(touch, timestamp_aligned, flag, gsr, bpm)


    

    # print(pipeline.task_one_df.head())
    
    # print(pipeline.task_one_df.describe())
    # print(pipeline.task_two_df.describe())
    # print(pipeline.task_three_df.describe())
    # print(pipeline.task_four_df.describe())
    # print(pipeline.task_five_df.describe())