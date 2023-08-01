import subprocess
import time
import sys
import logging

## THIS REQUIRES A TON OF RAM (> 16 GB)
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    
    print("Reading data...")
    p1 = subprocess.Popen(['python3 dataset_prep_touchtales/read_data.py'], shell=True)
    p1.wait()
    time.sleep(3)
    print("Cleaning data...")
    p2 = subprocess.Popen(['python3 dataset_prep_touchtales/clean_data.py'], shell=True)
    p2.wait()
    time.sleep(3)
    print("Calculating features...")
    p3 = subprocess.Popen(['python3 dataset_prep_touchtales/calculate_features.py'], shell=True)
    p3.wait()
    time.sleep(3)
    print("Calculating labels...")
    p4 = subprocess.Popen(['python3 dataset_prep_touchtales/calculate_labels.py'], shell=True)
    p4.wait()
    time.sleep(3)
    print("Calculating validation...")
    p5 = subprocess.Popen(['python3 dataset_prep_touchtales/calculate_validation.py'], shell=True)
    p5.wait()
    print("Done!")
