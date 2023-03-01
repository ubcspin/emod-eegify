import pathlib
import sys
import subprocess

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))


if __name__ == "__main__":
    print("Reading data...")
    p1 = subprocess.Popen(['python read_data.py'])
    p1.wait()
    print("Cleaning data...")
    p2 = subprocess.Popen(['python clean_data.py'])
    p2.wait()
    print("Calculating features...")
    p3 = subprocess.Popen(['python calculate_features.py'])
    p3.wait()
    print("Calculating labels...")
    p4 = subprocess.Popen(['python calculate_labels.py'])
    p4.wait()
    print("Calculating validation...")
    p5 = subprocess.Popen(['python calculate_validation.py'])
    p5.wait()
    print("Done!")
    sys.path.remove(str(_parentdir))