
import pandas as pd
import os, re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

filename = "../output/validparticipants.csv"
root_dir = os.path.dirname(os.path.abspath(__file__))
subject_file = os.path.join(root_dir, filename)

validparticipants = pd.read_csv(subject_file)
number_part_str = validparticipants.H4ID.drop_duplicates().__len__().__str__()

##################################################################
# summaries

validparticipants['inactive'] = validparticipants['lying'] + validparticipants['sitting'] + validparticipants['standing']

validparticipants['moderate'] = validparticipants['walking']
validparticipants['active'] = validparticipants['running']  + validparticipants['cycling']

print(validparticipants.describe())


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(validparticipants['inactive'] , validparticipants['moderate'] , validparticipants['active'],c = 'b',)

fig = ax.get_figure()

ax.set_title('Activity times (' + number_part_str + ' participants)', fontsize=12, fontweight='bold')

fig.savefig('plots/newplot.png')
