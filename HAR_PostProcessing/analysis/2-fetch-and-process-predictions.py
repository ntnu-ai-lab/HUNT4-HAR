import har_server_api as api
import pandas as pd
import h5py

import Dictinaries
from datetime import timedelta, datetime
from pandas import HDFStore
import requests



def create_summary(subjectid, labelled_timestamp):
    labelled_timestamp['timestamp'] = pd.to_datetime(labelled_timestamp['timestamp'], errors='coerce')
    labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.date
    labelled_timestamp = labelled_timestamp.replace({'label': Dictinaries.merge_classes})
    startrecording = labelled_timestamp.iloc[0]['timestamp']

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

    totals_row['weekday'] = totals_row.loc[:, 'date'].astype('datetime64[ns]').dt.dayofweek

    totals_row = totals_row.replace({'weekday':Dictinaries.weekdays})


    if 'lying' not in totals_row:
        totals_row['lying'] = 0
    if 'sitting' not in totals_row:
        totals_row['sitting'] = 0
    if 'standing' not in totals_row:
        totals_row['standing'] = 0
    if 'walking' not in totals_row:
        totals_row['walking'] = 0
    if 'running' not in totals_row:
        totals_row['running'] = 0
    if 'cycling' not in totals_row:
        totals_row['cycling'] = 0

    if 'running' not in totals_row:
        totals_row['running'] = 0

    if 'cycling' not in totals_row:
        totals_row['cycling'] = 0

    # totals_row = totals_row[0:7]
    totals_row = totals_row[['date', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'H4ID', 'weekday']]
    totals_row = totals_row.fillna(0)

    # selecting only wearable time
    #totals_row = totals_row.set_index(['date'])

    # pick start date and select the next - end is start +6 - plot 6 days

    #start = datetime.strptime(startrecording, '%Y-%m-%d').date() + timedelta(days=1)
    #end = start + timedelta(days=+5)
    #totals_row = totals_row[start: end]

        # divide by 20 to get minutes
    totals_row[["lying", "sitting", "standing", "walking", "running", "cycling"]] = totals_row[["lying", "sitting", "standing", "walking", "running", "cycling"]].divide(12, axis="index")
    return totals_row

host = 'http://localhost'
port = 10200
hf = HDFStore('/Volumes/LaCie/dataset/timestamped_predictions.hdf')
limit = 27007

subjectlist = pd.read_csv(api.subject_names(host, port, limit = limit, successful_only = True))
subjectlist = subjectlist['name'].tolist()

frames = []
for subject in subjectlist:
    print(subject)
    data = pd.read_csv( api.timestamped_predictions( host, port, subject ), names = ["timestamp", "label", "probability"])
    hf.put('s' + subject.__str__(), data, format='table', data_columns=True)
    #summary_df = create_summary(subject,data)
    #frames.append(summary_df)

#appended_data = pd.concat(frames, axis=0)
#appended_data.to_csv("../output/AAAAAA-summary-all-classes.csv", index=False)

print(hf.keys())

hf.close()