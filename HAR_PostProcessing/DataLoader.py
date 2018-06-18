import pandas as pd
import os
import glob
import re
import Dictinaries
from datetime import timedelta



sensortime = pd.read_csv("./data/sensortime.csv", parse_dates=[1,2])
sensortime = sensortime.set_index(['H4ID'])


def create_summary(filename):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    subject_file = os.path.join(root_dir, filename)

    subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()

    # incl. acc avg
    #data = pd.read_csv(subject_file, parse_dates=[0], header=None, names=['timestamp', 'x_thigh', 'y_thigh', 'z_thigh', 'x_back', 'y_back', 'z_back', 'label'])
    #labelled_timestamp = data[['timestamp','label']]

    labelled_timestamp = pd.read_csv(subject_file, parse_dates=[0], header=None, names=['timestamp', 'label'])

    #labelled_timestamp['date'] = labelled_timestamp['timestamp'].dt.date

    labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.date
    print(subjectid, labelled_timestamp.head(1),labelled_timestamp.tail(1))

    labelled_timestamp = labelled_timestamp.replace({'label': Dictinaries.merge_classes})

    totals_row = pd.pivot_table(labelled_timestamp, index=["date","label"], aggfunc='count').unstack()
    totals_row = pd.DataFrame(totals_row.to_records())

    #renaming columns - keeping all classes
    totals_row.rename(columns={'(\'timestamp\', 1)': 'walking'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 2)': 'running'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 3)': 'shuffling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 4)': 'stairs (ascending)'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 5)': 'stairs (descending)'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 6)': 'standing'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 7)': 'sitting'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 8)': 'lying'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 9)': 'transition'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 10)': 'bending'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 11)': 'picking'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 12)': 'undefined'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 13)': 'cycling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 14)': 'cycling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 15)': 'heel drop'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 16)': 'vigorous activity'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 17)': 'non-vigorous activity'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 18)': 'Car'}, inplace=True)

    totals_row["H4ID"] = subjectid

    # Steinkjer classes
    #totals_row["class"] = Dictinaries.steinkjier_classes[subjectid]

    totals_row['weekday'] = totals_row.loc[:, 'date'].astype('datetime64[ns]').dt.dayofweek

    totals_row = totals_row.replace({'weekday':Dictinaries.weekdays})
    #Steinkjer
    #totals_row = totals_row[1:6]
    #totals_row = totals_row[['date', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'H4ID', 'weekday','class']]

    if 'running' not in totals_row:
        totals_row['running'] = 0

    if 'cycling' not in totals_row:
        totals_row['cycling'] = 0

    # totals_row = totals_row[0:7]
    totals_row = totals_row[['date', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'H4ID', 'weekday']]
    totals_row = totals_row.fillna(0)

    # selecting only wearable time
    totals_row = totals_row.set_index(['date'])

    # change: take start date >> add 1 day as start >> end = start + 6 days
    if (subjectid in sensortime.index):
        start = sensortime.ix[int(subjectid), 'start'].date()
        totals_row = totals_row[sensortime.ix[int(subjectid), 'start'].date() + timedelta(days=1):sensortime.ix[int(subjectid), 'end'].date() + timedelta(days=-1)]
    totals_row.to_csv("output/" + subjectid + "_summary.csv")


#path = "data/*_predictions.csv"
path = "data/70*s.csv"

for fname in glob.glob(path):
    create_summary(fname)