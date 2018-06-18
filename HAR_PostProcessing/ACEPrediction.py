import matplotlib
import shutil

matplotlib.use('Agg')
import glob
import re, os, sys
import pandas as pd
from datetime import datetime

from acrechain import complete_end_to_end_prediction
from joblib import Parallel, delayed
import time

# detailed activities
#from HUNT4ParticipantPlot import create_barplot, create_summary


# aggregated activities
from HUNT4ParticipantPlotAggActivities import create_barplot, create_summary


def create_predictions(filename, path, freq=50):
    subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()
    input = path
    backfile = ''
    thighfile = ''
    print('in', input)

    for name in glob.glob(input + '*_B.cwa'):
        print(name)
        backfile = name
        print('backfile', backfile)

    for name in glob.glob(input + '*_T.cwa'):
        thighfile = name
        print('thighfile', thighfile)

    print(backfile, thighfile)

    if backfile:
        if 'T' in thighfile:
            if 'B' in backfile:
                output = input + subjectid + "_timestamped_predictions.csv"
                complete_end_to_end_prediction(backfile, thighfile, output, sampling_frequency=freq)
                # output >> sent to make summaries & pretty
                startrecording = re.search('%s(.*)%s' % ('_', '_'), backfile).group(1)
                summary_data_filename = create_summary(output, subjectid, path, startrecording)
                create_barplot(summary_data_filename, subjectid, path, startrecording)
                # remove folder ../tmp
                #shutil.rmtree(path + "tmp")

# get csv with folder names
folderlist = sys.argv[1]
logfile = sys.argv[2]
#

folders = pd.read_csv(folderlist)

#
for i, row in folders.iterrows():
    start = datetime.now()
    foldername = folders.loc[i, "foldername"]
    path = "huntdata/" + foldername + "/"
    subjectpath = path + "*"
    print('foldername', foldername)
    create_predictions(foldername, foldername, 100)

    end = datetime.now()
    with open(logfile, 'w') as file:
        file.write(foldername + ',' + str(start) + ',' + str(end))


#create_predictions(path, path, 50)


############# old version #############
# for fname in glob.glob(subjectpath):
#     if not os.path.exists(path + "summaries"):
#         os.makedirs(path + "summaries")
#     if not os.path.exists(path + "tmp"):
#         os.makedirs(path + "tmp")
#     create_predictions(fname, path, 50)
#
# # create one summary file for this date
# summaries = path + "summaries/*"
# frames = []
# for fname in glob.glob(summaries):
#     subject_file = os.path.join(fname)
#     data = pd.read_csv(subject_file, parse_dates=[0])
#     frames.append(data)
# appended_data = pd.concat(frames, axis=0)
# appended_data.to_csv(path + "summary-" + date_processed + ".csv", index=False)
# shutil.rmtree(path + "summaries")
############# old version #############

# parallel on 20 cores:
#Parallel(n_jobs=1)(delayed(create_predictions)(fname) for fname in glob.glob(path))
