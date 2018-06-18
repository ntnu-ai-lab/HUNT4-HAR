import matplotlib
import pandas as pd

import os
import re
import glob
import Dictinaries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta, datetime
import itertools


def create_summary(filename, subjectid, path, startrecording):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    subject_file = os.path.join(root_dir, filename)
    print('subject_file', subject_file)

    labelled_timestamp = pd.read_csv(subject_file, parse_dates=[0], header=None, names=['timestamp', 'label'])
    print('labelled_timestamp', labelled_timestamp)
    labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.date
    labelled_timestamp = labelled_timestamp.replace({'label': Dictinaries.merge_classes})
    totals_row = pd.pivot_table(labelled_timestamp, index=["date","label"], aggfunc='count').unstack()
    totals_row = pd.DataFrame(totals_row.to_records())

    #renaming columns - keeping all classes
    totals_row.rename(columns={'(\'timestamp\', 1)': 'walking'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 2)': 'running'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 3)': 'shuffling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 4)': 'stairs (ascending)'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 5)': 'stairs (descending)'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 6)': 'standing'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 7)': 'sitting'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 8)': 'lying'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 9)': 'transition'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 10)': 'bending'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 11)': 'picking'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 12)': 'undefined'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 13)': 'cycling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 14)': 'cycling'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 15)': 'heel drop'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 16)': 'vigorous activity'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 17)': 'non-vigorous activity'}, inplace=True)
    totals_row.rename(columns={'(\'timestamp\', 18)': 'Car'}, inplace=True)

    # data comes in 3 sec windows, multiply by 3 to get data in minutes
    totals_row["H4ID"] = subjectid

    totals_row['weekday'] = totals_row.loc[:, 'date'].astype('datetime64[ns]').dt.dayofweek

    totals_row = totals_row.replace({'weekday':Dictinaries.weekdays})

    if 'running' not in totals_row:
        totals_row['running'] = 0

    if 'cycling' not in totals_row:
        totals_row['cycling'] = 0

    # totals_row = totals_row[0:7]
    totals_row = totals_row[['date', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'H4ID', 'weekday']]
    totals_row = totals_row.fillna(0)

    # selecting only wearable time
    totals_row = totals_row.set_index(['date'])

    # pick start date and select the next - end is start +6 - plot 6 days

    start = datetime.strptime(startrecording, '%Y-%m-%d').date() + timedelta(days=1)
    end = start + timedelta(days=+5)
    totals_row = totals_row[start: end]

    # divide by 20 to get minutes
    totals_row[["lying", "sitting", "standing", "walking", "running", "cycling"]] = totals_row[["lying", "sitting", "standing", "walking", "running", "cycling"]].divide(20, axis="index")
    filename = path + subjectid + "_summary.csv"
    totals_row.to_csv(filename)
    return filename

#test
#def create_barplot(data, subjectid, path, startrecording):

#deploy
def create_barplot(filename, subjectid, path, startrecording):
    root_dir = os.path.dirname(os.path.abspath(__file__))               #deploy
    subject_file = os.path.join(root_dir, filename)                     #deploy

    # data = pd.read_csv(subject_file)
    data = pd.read_csv(subject_file, parse_dates=[0])                   #deploy
    data = data.drop('H4ID', axis=1)
    data = data.drop('weekday', axis=1)
    # data = data.drop('class', axis=1)
    # convert seconds to minutes
    data = data.set_index('date')
    data = data.divide(60)
    data = data[["cycling", "running", "walking", "standing", "sitting", "lying"]]

    # avg per activity
    activty_avg = data.mean() * 60

    # avg per weekday / weekend
    weekday_act_avg = data.reset_index()
    weekday_act_avg['weekday'] = data.index.weekday
    weekday_act_avg['wkd'] = weekday_act_avg.weekday.map(Dictinaries.uke_helge_norsk)
    weekday_act_avg = weekday_act_avg.assign(
        active=weekday_act_avg.cycling + weekday_act_avg.running + weekday_act_avg.walking)
    weekday_act_avg = weekday_act_avg.assign(inactive=weekday_act_avg.sitting)
    weekday_act_avg = weekday_act_avg[['wkd', 'active', 'inactive']]
    desired_decimals = 2
    weekday_act_avg['active'] = weekday_act_avg['active'].apply(lambda x: round(x, desired_decimals))

    weekday_act_avg = weekday_act_avg.groupby('wkd').mean() * 60
    weekday_act_avg = weekday_act_avg.reset_index()
    #   weekday_act_avg = weekday_act_avg[['wkd','active']]

    # check if both 'på ukedager' and 'på helgedager' exits - if not create it with 0
    uke = 'på ukedager' in weekday_act_avg.wkd.values
    helg = 'på helgedager' in weekday_act_avg.wkd.values

    if not uke:
        weekday_act_avg = weekday_act_avg.append(pd.DataFrame([{'wkd': 'på ukedager', 'active': 0, 'inactive': 0}]))
    if not helg:
        weekday_act_avg = weekday_act_avg.append(pd.DataFrame([{'wkd': 'på helgedager', 'active': 0, 'inactive': 0}]))

    # add sitting time table
    weekday_sit_avg = weekday_act_avg

    # weekday_act_avg.set_index('wkd', inplace=True)
    weekday_act_avg['active'] = weekday_act_avg['active'].apply(lambda x: round(x, 1))
    weekday_act_avg['avg'] = weekday_act_avg.active.map(str) + " min"
    weekday_act_avg = weekday_act_avg[['wkd', 'avg']]

    weekday_sit_avg['inactive'] = weekday_sit_avg['inactive'].apply(lambda x: round(x, 1))
    weekday_sit_avg['avg'] = weekday_sit_avg.inactive.map(str) + " min"
    weekday_sit_avg = weekday_sit_avg[['wkd', 'avg']]

    #############################  Plotting  #############################
    colors = itertools.cycle(['forestgreen', 'forestgreen', 'forestgreen', 'lightyellow', 'lightcyan', 'skyblue'])
    activities = ['cycling', 'running', 'walking', 'standing', 'sitting', 'lying']
    bar_width = 0.3
    n_dates = len(data.index)

    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((3, 7), (0, 0), rowspan=2, colspan=4)

    # creating date, weekday labels
    x_lab_data = data.reset_index()
    x_lab_data['weekday'] = data.index.weekday
    x_lab_data['datestr'] = data.index.strftime('%d.%m.%y ')
    x_lab_data['final'] = x_lab_data.weekday.map(Dictinaries.weekdays_norsk)
    x_lab_data['final'] = x_lab_data['datestr'].map(str) + "\n " + x_lab_data['final']
    # x_labels = ['']
    x_labels = x_lab_data.final.values

    Bplot_Xindex = list(range(n_dates))
    bottom = [0] * n_dates

    # Bar Chart
    for activity in activities:
        ax1.bar(Bplot_Xindex, data[activity].tolist(), bar_width, color=next(colors), bottom=bottom)

        bottom = [x + y for x, y in zip(bottom, data[activity].tolist())]

    ax1.set_xlabel("")
    ax1.set_ylabel("Timer")
    ax1.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    ax1.set_title('Fysisk aktivitetsprofil', fontsize=12, fontweight='bold')
    ax1.set_xticks(Bplot_Xindex)
    ax1.set_xticklabels(x_labels, rotation=0)

    # creating a legend
    # bike = mpatches.Patch(color='darkorange', label='Sykle')
    active = mpatches.Patch(color='forestgreen', label='Fysisk aktivitet')
    # walk = mpatches.Patch(color='forestgreen', label='Gå')
    stand = mpatches.Patch(color='lightyellow', label='Stå')
    sit = mpatches.Patch(color='lightcyan', label='Sitte')
    ly = mpatches.Patch(color='skyblue', label='Ligge')
    plt.legend(handles=[active, stand, sit, ly], loc='lower left', bbox_to_anchor=(1, 0.65))

    ############## Making the table for Gjennomsnitt fysiks aktivitet tid per dag,
    ############## first column

    ax2 = plt.subplot2grid((3, 7), (2, 0), colspan=1)
    lightgray = (0.90, 0.90, 0.90)

    col_labels = ['Tid med\nfysisk aktivitet*']
    table_vals = [['på ukedager'],
                  ['på helgedager']]

    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_frame_on(False)
    avg_values_table = plt.table(cellText=table_vals, cellLoc='left',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(0.3, 0.3, 0.9, 0.8))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(0.9, 2)

    ############### second column
    ax3 = plt.subplot2grid((3, 7), (2, 1), colspan=1)

    col_labels = ['Gjennomsnitt\nper dag']
    table_vals = [[weekday_act_avg.values[1, 1]],
                  [weekday_act_avg.values[0, 1]]]

    avg_uke = weekday_act_avg.loc[weekday_act_avg['wkd'] == 'på ukedager'].avg.item()
    avg_helg = weekday_act_avg.loc[weekday_act_avg['wkd'] == 'på helgedager'].avg.item()
    table_vals = [[avg_uke], [avg_helg]]

    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_frame_on(False)
    avg_values_table = plt.table(cellText=table_vals, cellLoc='right',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(0, 0.3, 0.9, 0.8))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(0.9, 3)

    ############ Making the table for Gjennomsnitt sitte tid per dag,
    ############ frist column
    ax4 = plt.subplot2grid((3, 7), (2, 2), colspan=1)
    col_labels = ['Tid i sittende']
    table_vals = [['på ukedager'],
                  ['på helgedager']]

    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.set_frame_on(False)
    avg_values_table = plt.table(cellText=table_vals, cellLoc='left',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(0.3, 0.3, 0.9, 0.8))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(0.9, 3)

    ############## second column
    col_labels = ['Gjennomsnitt\nper dag']
    ax5 = plt.subplot2grid((3, 7), (2, 3), colspan=1)

    avg_uke = weekday_sit_avg.loc[weekday_sit_avg['wkd'] == 'på ukedager'].avg.item()
    avg_helg = weekday_sit_avg.loc[weekday_sit_avg['wkd'] == 'på helgedager'].avg.item()
    table_vals = [[avg_uke], [avg_helg]]

    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.set_frame_on(False)
    avg_values_table = plt.table(cellText=table_vals, cellLoc='right',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(0, 0.3, 0.9, 0.8))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        # cell.set_height(1)
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(0.9, 3)

    ########### Making the table for average time of different activities,
    ########### first column

    ax6 = plt.subplot2grid((3, 7), (1, 4), colspan=1, rowspan=2)
    col_labels = ['Aktivitet']
    table_vals = [data.cycling.mean() + data.running.mean() + data.walking.mean(), data.standing.mean(),
                  data.sitting.mean(), data.lying.mean()]
    temp = [round(x * 60, 1) for x in table_vals]
    temp_str = [str(item) + ' min' for item in temp]
    table_vals = [['Fysisk aktivitet'],
                  ['Stå'], ['Sitte'], ['Ligge']]

    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_visible(False)
    ax6.set_frame_on(False)

    avg_values_table = plt.table(cellText=table_vals, cellLoc='left',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(-0.1, 0.61, 0.9, 0.6))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(1, 2)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)

    ############# Second column
    ax7 = plt.subplot2grid((3, 7), (1, 5), colspan=1, rowspan=2)
    col_labels = ['Gjennomsnitt\nper dag']
    table_vals = [data.cycling.mean() + data.running.mean() + data.walking.mean(), data.standing.mean(),
                  data.sitting.mean(), data.lying.mean()]
    temp = [round(x * 60, 1) for x in table_vals]
    temp_str = [str(item) + ' min' for item in temp]
    table_vals = [[temp_str[0]], [temp_str[1]], [temp_str[2]],
                  [temp_str[3]]]

    ax7.xaxis.set_visible(False)
    ax7.yaxis.set_visible(False)
    ax7.set_frame_on(False)

    avg_values_table = plt.table(cellText=table_vals, cellLoc='right',
                                 colLabels=col_labels, colColours=[lightgray] * 16,
                                 bbox=(-0.4, 0.61, 0.7, 0.6))

    table_props = avg_values_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(9)
        cell.set_linewidth(0)

    avg_values_table.scale(1, 2)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)

    plt.figtext(0.09, 0.04, '* Tid med fysisk aktivitet = summen av å gå, løpe/jogge og sykle')
    fig.savefig(path + '/' + subjectid.__str__() + '.png', bbox_inches='tight', dpi=150)
    plt.close()
