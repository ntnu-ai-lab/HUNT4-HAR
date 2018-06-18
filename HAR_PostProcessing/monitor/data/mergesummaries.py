import glob
import pandas as pd
import os


path = "/Users/kerstin/Desktop/hunt/nrk/"
summaries = "summaries/*"
frames = []
for fname in glob.glob(path+summaries):
    subject_file = os.path.join(fname)
    data = pd.read_csv(subject_file, parse_dates=[0])
    frames.append(data)
appended_data = pd.concat(frames, axis=0)
appended_data.to_csv(path + "summary.csv", index=False)