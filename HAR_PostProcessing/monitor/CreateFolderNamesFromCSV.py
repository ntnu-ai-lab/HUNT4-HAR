import pandas as pd
import os
from datetime import timedelta, datetime


folderlist = 'data/bin-1.csv'
logfile = 'bin-1.log'

### include ###
folders = pd.read_csv(folderlist)
print(folders)

#
for i, row in folders.iterrows():
    start = datetime.now()
    foldername = folders.loc[i, "foldername"]
    print('analyze: ', foldername)

    end = datetime.now()
    with open(logfile, 'w') as file:
        file.write(foldername + ',' + str(start) + ',' + str(end))

#foldername = sys.argv[1]
