import pandas as pd
import Utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))

subject_file = os.path.join(root_dir, "data/" + "0815" + "_timestamped_predictions.csv")
data = pd.read_csv(subject_file, parse_dates=[0], header=None,
                   names=['timestamp', 'x_thigh', 'y_thigh', 'z_thigh', 'x_back', 'y_back', 'z_back', 'label'])
labelled_timestamp = data[['timestamp', 'label']]

labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.dayofweek

# totals_total = pd.pivot_table(labelled_timestamp, index=["weekday", "label"], aggfunc='count')

# white:    8                       # sedentary --> 1
# orange: 6, 7                    # inactive --> 2
# yellow: 2 , 13 , 14             # vigorous --> 4
# green: 1 , 4, 5, 10             # moderate --> 3
# blue:                           # sleep

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

no_to_intensity_dict = {
    1: "3",
    2: "4",
    4: "3",
    5: "3",
    6: "2",
    7: "2",
    8: "1",
    10: "2",
    13: "4",
    14: "4",
}

no_to_acti4_dict = {
    1: "3",
    2: "4",
    4: "3",
    5: "3",
    6: "2",
    7: "2",
    8: "1",
    10: "2",
    13: "4",
    14: "4",
}

# line:

day = wednesday
daylabel = "Wednesday"

#day = day[65000:65500]
day = day.reset_index()
del day['index']

print(day.head())

y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
text_values = ["walking", "running", "shuffling", "stairs (ascending)", "stairs (descending)", "standing", "sitting",
               "lying", "transition", "bending", "picking", "undefined", "cycling (sit)", "cycling (stand)"]
plt.yticks(y_values, text_values)
plt.title(daylabel)
#line plot
#plt.plot(day['date'], day['label'])

# scatter:
plt.plot(day['date'], day['label'], ".")

# plt.savefig("plots/Steinkjer_adolescents-" + subjectid)
plt.show()


