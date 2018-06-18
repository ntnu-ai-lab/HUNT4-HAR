import pandas as pd
import os
import glob
import re


#path = "output/h4-data/hunt4-output-csv/*_summary.csv"
path = "/Volumes/LaCie/extracted/*_summary.csv"
# 1 = keep minutes; 60 = use hours
divide = 1
frames = []
for fname in glob.glob(path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    subject_file = os.path.join(root_dir, fname)
    data = pd.read_csv(subject_file, parse_dates=[0])
    frames.append(data)

appended_data = pd.concat(frames, axis=0)  ## see documentation for more info
appended_data.to_csv("output/summary-all-classes.csv", index=False)

appended_data['lying'] = appended_data['lying'].apply(lambda x: x / divide)
appended_data['sitting'] = appended_data['sitting'].apply(lambda x: x / divide)
appended_data['standing'] = appended_data['standing'].apply(lambda x: x / divide)
appended_data['walking'] = appended_data['walking'].apply(lambda x: x / divide)
appended_data['running'] = appended_data['running'].apply(lambda x: x / divide)
appended_data['cycling'] = appended_data['cycling'].apply(lambda x: x / divide)

appended_data.to_csv("output/h4-sum-minutes.csv", index=False)

# write to excel
#writer = pd.ExcelWriter('output/summary-all-classes-minutes.xlsx', index=False)
#appended_data.to_excel(writer,'all classes')
#writer.save()