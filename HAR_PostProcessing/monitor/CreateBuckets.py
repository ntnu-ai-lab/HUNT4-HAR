import pandas as pd
import os

datafolders = os.listdir('huntdata')
file = "helpers/SensorLog-2017-10-09.csv"
data = pd.read_csv(file, delimiter=';', dayfirst=True, parse_dates=[2])
data = data.drop_duplicates(subset='H4Id', keep="first")

bins = 10

data["bin"] = ""
data["foldername"] = ""

bincount = 1
for i, row in data.iterrows():
    data.loc[i, "bin"] = bincount
    sub = data.loc[i, "H4Id"].astype(str)
    data.loc[i, "foldername"] = next((s for s in datafolders if sub in s), None)
    if (bincount < bins):
        bincount = bincount + 1
    else:
        bincount = 1


idnotfound = data[(data["foldername"].isnull())]
idnotfound[['H4Id','SensorId','DateExported']].to_csv('helpers/H4IdNotFound.csv', index=False)

data = data.dropna()

count = 1
while (count <= bins):
    bin = data[(data["bin"] == count)]
    bin[['foldername']].to_csv('helpers/bin-'+count.__str__()+'.csv', index=False, header=False)
    count = count + 1
