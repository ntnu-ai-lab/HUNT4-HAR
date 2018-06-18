import glob
import pandas as pd
import numpy as np
import os

# list of file names (H4IDs) that have been processsed and moved to the archive
archived = pd.read_csv('data/archive_ids', names = ['H4ID'])
# list of files (H4IDs plus zip extension)
incoming = pd.read_csv('data/incoming_ids', names = ['H4ID'])
# logged IDs (list from HUNT)
logged = pd.read_csv("data/logactivity-2017-01-12.csv", delimiter=';', dayfirst=True, parse_dates=[2])

#print(logged.head())

logged = logged.drop_duplicates('H4Id')
#print(logged.info())


archived['H4ID'] = archived['H4ID'].astype(int)

# Find IDs that are in both incoming and archived
incoming_and_archived = archived[archived.H4ID.isin(incoming.H4ID)]
# Find IDs that are in incoming, but not archived (i.e. analyzed yet) yet
incoming_not_archived = incoming[~incoming.H4ID.isin(archived['H4ID'])]
logged_notarchived = logged[~logged.H4Id.isin(archived['H4ID'])]

idsincoming_notlogged = incoming[~incoming.H4ID.isin(logged['H4Id'])]

incoming_not_archived = incoming_not_archived.sort_values(by = 'H4ID')
logged_notarchived = logged_notarchived.sort_values(by = 'H4Id')

print("############## IDs which are both in archived and incoming", incoming_and_archived.count())
print("############## IDs in incoming, not archived/analyzed", incoming_not_archived.count())
print("############## IDs logged at HUNT, not archived", logged_notarchived.count())

logged_notarchived.H4Id.to_csv(('data/logged_notarchived.csv'), index=False)
incoming_not_archived.H4ID.to_csv(('data/incoming_not_archived.csv'), index=False)
idsincoming_notlogged.H4ID.to_csv(('data/incoming-notlogged.csv'), index=False)