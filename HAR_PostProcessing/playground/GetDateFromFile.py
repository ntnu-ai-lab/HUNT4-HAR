import glob

import pandas as pd
import os, re
import matplotlib.pyplot as plt
from datetime import datetime

backfile = "huntdata/2017-09-02/7050/7050-1223_2017-06-20_B.cwa"


startrecording = re.search('%s(.*)%s' % ('_', '_'), backfile).group(1)
datetime_object = datetime.strptime(startrecording, '%Y-%m-%d')

print(startrecording)
print(datetime_object)


thighfile = ''

if not thighfile:
        print("thighfile is empty")

input = '../data/'
cwafile = '*_B.cwa'
for name in glob.glob(input + cwafile):
    backfile = name
    print("backfile is", backfile)

print("*****************************************************************************")
filename = "/huntdata/2017-09-07-4217527"
subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()
print(subjectid)