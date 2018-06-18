import pandas as pd
import matplotlib.pyplot as plt
import os, glob, re
import numpy as np


#filename = "output/1014_summary.csv"
def create_pieplot(filename):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    subject_file = os.path.join(root_dir, filename)

    subjectid = map(int,re.findall('\d+', filename)).pop().__str__()
    data = pd.read_csv(subject_file)
    data = data.drop('H4ID', axis=1)
    data = data.drop('weekday', axis=1)
    data = data.drop('class', axis=1)
    # convert seconds to minutes
    data = data.set_index('date')
    data = data.divide(60).transpose()
    data['Total'] = data.sum(axis=1)

    colors = ['skyblue', 'lightcyan', 'lightyellow', 'forestgreen', 'red', 'darkorange']

    ### Version 1 ###
    #data['Total'].transpose().plot(kind='pie', figsize=(8, 8), colors=colors, title='Subject #' + subjectid)

    ### Version 2 ###
    #explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    data['Total'].transpose().plot(kind='pie', figsize=(9, 7), colors=colors, #explode=explode,
                                   title='Subject #' + subjectid)
    plt.axis('equal')
    plt.savefig('plots/'+subjectid+'-pie-summary-rm.png')
    plt.close()

path = "output/*_summary.csv"

for fname in glob.glob(path):
    create_pieplot(fname)