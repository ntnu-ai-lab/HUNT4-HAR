# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pandas as pd
import Dictinaries



data_raw = pd.read_csv("../output/1176_summary.csv", parse_dates=[0])

data = data_raw[['date','lying']]
data = data.set_index('date')
data = data.divide(60 * 60)


x_lab_data = data
x_lab_data['weekday'] = data.index.weekday
x_lab_data['datestr'] = data.index.strftime('%d.%m.%Y')
x_lab_data['final'] =  x_lab_data.weekday.map(Dictinaries.weekdays_norsk)

x_lab_data['final'] = x_lab_data['datestr'].map(str) + " " + x_lab_data['final']
print(x_lab_data['final'])