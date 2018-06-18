# This script takes a set of summaries and creates HUNT plots for all of them
import pandas as pd

import os, re
import Dictinaries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

from HUNT4ParticipantPlotAggActivities import create_barplot

#filename = "../output/sum-sample-file.csv"
filename = "../output/h4-sum-minutes.csv"
root_dir = os.path.dirname(os.path.abspath(__file__))
subject_file = os.path.join(root_dir, filename)

data = pd.read_csv(subject_file, parse_dates=[0])

h4ids = data.H4ID
h4ids = h4ids.drop_duplicates()


for ind, subjectid in h4ids.iteritems():
    data_onesubject = data[data.H4ID == subjectid]
    startrecording = data_onesubject.iloc[0].date.__str__()
    create_barplot(data_onesubject, subjectid, './plots', startrecording[:10])
