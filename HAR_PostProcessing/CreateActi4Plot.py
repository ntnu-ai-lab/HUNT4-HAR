import pandas as pd
import Utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))

subjectid = "108"

subject_file = os.path.join(root_dir, "data/" + subjectid + "_timestamped_predictions.csv")
data = pd.read_csv(subject_file, parse_dates=[0], header=None,
                   names=['date', 'label'])
labelled_timestamp = data[['date', 'label']]

labelled_timestamp['weekday'] = labelled_timestamp.loc[:, 'date'].dt.dayofweek

# totals_total = pd.pivot_table(labelled_timestamp, index=["weekday", "label"], aggfunc='count')
print(labelled_timestamp.head())


no_to_label_dict = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
    9: "transition",
    10: "bending",
    11: "picking",
    12: "undefined",
    13: "cycling (sit)",
    14: "cycling (stand)",
    15: "heel drop",
    16: "vigorous activity",
    17: "non-vigorous activity",
    18: "Car"
}

#labelled_timestamp = labelled_timestamp.replace({'label': no_to_label_dict})
#
monday = labelled_timestamp[labelled_timestamp.weekday == 0]
tuesday = labelled_timestamp[labelled_timestamp.weekday == 1]
wednesday = labelled_timestamp[labelled_timestamp.weekday == 2]
thursday = labelled_timestamp[labelled_timestamp.weekday == 3]
friday = labelled_timestamp[labelled_timestamp.weekday == 4]
saturday = labelled_timestamp[labelled_timestamp.weekday == 5]
sunday = labelled_timestamp[labelled_timestamp.weekday == 6]

days = [monday,tuesday,wednesday,thursday,friday,saturday,sunday]
daysnames = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
for d in days:
    day = d
    print (labelled_timestamp.ix[0,'date'])
    daylabel = daysnames[labelled_timestamp.ix[0,'weekday']+1]

    #day = day[65000:65500]
    day = day.reset_index()
    del day['index']

    #print day.head()

##### plotting #######

# scatter:
# plt.plot(wednesday['date'], wednesday['label'], "o")

# line:

    y_values = [2, 4, 5, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    text_values = ["running", "stairs (ascending)", "stairs (descending)", "walking", "shuffling", "standing", "sitting",
               "lying", "transition", "bending", "picking", "undefined", "cycling (sit)", "cycling (stand)"]
    plt.yticks(y_values, text_values)
    plt.title(subjectid + " " + daylabel)
    print(day['date'])
    print(day['label'])
    plt.plot(day['date'], day['label'])

    plt.savefig("sample-" + subjectid + "-" + daylabel + ".png")
    print("sample-" + subjectid + "-" + daylabel + ".png")
    #plt.show()

