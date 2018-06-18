import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


file = "data/logactivity-2017-01-12.csv"
data = pd.read_csv(file, delimiter=';', dayfirst=True, parse_dates=[2])
data = data.sort_values(by = 'DateExported')

catdata = data
catdata.SensorId = catdata.SensorId.astype('category')

# getting the top most used sensors
topused = catdata['SensorId'].value_counts()

#print(data['SensorId'].value_counts())

# vizualize those ids that are in the top used df
vizcontent = data.loc[data['SensorId'].isin(topused.index)]

# don't want to do this
# vizcontent.SensorId = vizcontent.SensorId.astype('int64')

#######################################

data2 = pd.pivot_table(vizcontent, index=['SensorId','DateExported'], aggfunc='sum').dropna().unstack()

print(data2.head())
group_name = data.groupby(['SensorId', 'DateExported'])
print(group_name.size().head(20))

#plotting
#ax = vizcontent.plot(x='DateExported', style='.')
#fig = ax.get_figure()

#fig.savefig('vizcontent.png')


#######################################
turnaround = data

turnaround['Days'] = (turnaround.groupby('SensorId', group_keys=False).apply(lambda g: g['DateExported'].diff().replace(0, np.nan).ffill()))
turnaround = turnaround.dropna()

tadays = turnaround[['Days']]

#plotting
tadays = tadays['Days'].apply(lambda x: x /pd.Timedelta(days=1))

print(tadays.describe())

# get the data
ax = tadays.hist()

# Set the x-axis label
ax.set_xlabel("Number of Days")

# Set the y-axis label
ax.set_ylabel("Occurrence")

# sensor turnover --> add label

fig = ax.get_figure()


fig.savefig('histogram.png')

