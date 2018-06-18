import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.formats.printing import pprint_thing

seventy_fifty = "/PATH/TO/7050_timesync_time_B_T.csv"
seventy_hundred = "/PATH/TO/7100_timesync_time_B_T.csv"
hz_zero_one = "/PATH/TO/Hz01_timesync_time_B_T.csv"
hz_zero_two = "/PATH/TO/Hz02_timesync_time_B_T.csv"
hz_zero_three = "/PATH/TO/Hz03_timesync_time_B_T.csv"
hz_zero_four = "/PATH/TO/Hz04_timesync_time_B_T.csv"
hz_zero_five = "/PATH/TO/Hz05_timesync_time_B_T.csv"
hz_zero_six = "/PATH/TO/Hz06_timesync_time_B_T.csv"
hz_zero_seven = "/PATH/TO/Hz07_timesync_time_B_T.csv"
hz_zero_eight = "/PATH/TO/Hz08_timesync_time_B_T.csv"


def read_raw_csv_files(csv_file):
    return pd.read_csv(csv_file)

def separate_back_and_thigh_data(dataframe):
    back = dataframe.iloc[:,1:4]
    thigh = dataframe.iloc[:,4:7]
    return back, thigh

def calculate_means(dataframe):
    means = []
    for i in range(3):
        means.append(dataframe.iloc[:,i].mean())
    return means

def calculate_stds(dataframe):
    stds = []
    for i in range(3):
        stds.append(dataframe.iloc[:, i].std())
    return stds

def find_csv_stats(csv_file):
    dataframe = read_raw_csv_files(hz_zero_eight)
    back, thigh = separate_back_and_thigh_data(dataframe)
    back_means = calculate_means(back)
    thigh_means = calculate_stds(back)


