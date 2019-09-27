import pandas as pd

import Dictinaries
import Utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, re, glob

root_dir = os.path.dirname(os.path.abspath(__file__))

def print_weekly_view(filename):

    maxSampleDay = 14400 # 14400 for 50Hz - assumes only missing data the first and the last day

    subject_file = os.path.join(root_dir, filename)
    subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()

    labelled_timestamp = pd.read_csv(subject_file, parse_dates=[0], header=None, names=['timestamp', 'label'])
    labelled_timestamp['date'] = labelled_timestamp.loc[:, 'timestamp'].dt.date

    labelled_timestamp = labelled_timestamp.replace({'label': Dictinaries.merge_classes})

    days = labelled_timestamp['date'].drop_duplicates()

    #classes
    # 1:walking
    # 2:running
    # 6:standing
    # 7:sitting
    # 8:lying
    # 9:transition
    # 13:cycling


    no_to_color_dict = {
        1: "forestgreen",
        2: "red",
        6: "lightyellow",
        7: "lightcyan",
        8: "skyblue",
        13: "darkorange",
        99: "white"
      }

    labelled_timestamp = labelled_timestamp.replace({'label':no_to_color_dict})

    first_day = labelled_timestamp.loc[labelled_timestamp['date'] == days.iloc[0]]
    first_day = first_day[['timestamp', 'label']]
    first_day = first_day.set_index('timestamp')

    # add no-wear time to the first day
    missingdatapoints = maxSampleDay - first_day.count()
    timestamp_startnoweartime = first_day.first_valid_index() - pd.Timedelta(seconds=(missingdatapoints.__int__())*6)
    # create new data frame
    missingdata = pd.DataFrame({'timestamp': pd.date_range(start=timestamp_startnoweartime, freq='6s', periods=missingdatapoints.__int__())})
    missingdata['label'] = 'white'
    missingdata = missingdata.set_index('timestamp')
    first_day = pd.concat([missingdata, first_day])
    no_of_days = days.count()

    # Make a figure and axes with dimensions as desired.
    # start with one
    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle('Subject #' + subjectid, fontsize="x-large")

    ax = fig.add_subplot(111)
    ####### First DAY ##########
    cmap = mpl.colors.ListedColormap(first_day.reset_index().label.tolist())

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = first_day.reset_index().index.tolist()
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb0 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    # to use 'extend', you must
                                    # specify two extra boundaries:
                                    #boundaries= bounds,
                                    #ticks=bounds,  # optional
                                    #spacing='proportional',
                                    orientation='horizontal')

    a = ['', '2:24am', '4:48am', '7:12am', '9:36am', '12pm', '2:24pm', '4:48pm', '7:12pm', '9:36pm']
    ax.set_xticklabels(a)
    cb0.set_label('Date: '+ days.iloc[0].__str__() +' (orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

    # now later you get a new subplot; change the geometry of the existing
    for c in range(no_of_days-1):
        current = c+1
        one_day = labelled_timestamp.loc[labelled_timestamp['date'] == days.iloc[current]]
        one_day = one_day[['timestamp', 'label']]
        one_day = one_day.set_index('timestamp')

        if (one_day.count().__int__() < maxSampleDay):
            missingdatapoints = maxSampleDay - one_day.count().__int__()
            # create new data frame
            missingdataend = pd.DataFrame({'timestamp': pd.date_range(start=one_day.last_valid_index(), freq='6s',
                                                                   periods=missingdatapoints.__int__())})
            missingdataend['label'] = 'white'
            missingdataend = missingdataend.set_index('timestamp')
            one_day = pd.concat([one_day, missingdataend])

        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 1, 1, i + 1)

        # add the new
        ax = fig.add_subplot(n + 1, 1, n + 1)
        ####### Next DAYs ##########
        cmap = mpl.colors.ListedColormap(one_day.reset_index().label.tolist())
        bounds = one_day.reset_index().index.tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb0 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        ax.set_xticklabels(a)
        cb0.set_label('Date: '+ days.iloc[current].__str__() +' (orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig("plots/Daily-BarChart-" + subjectid)
    #plt.savefig("/Volumes/LaCie/day-bar-plots/H4ID-" + subjectid)
    #plt.savefig("plots/Steinkjer_adolescents-" + subjectid + '-new-sync')
    plt.close(fig)


#create single
#print_weekly_view("data/1103_timestamped_predictions.csv")

#create all
path = "data/10*.csv"
#path = "/Volumes/LaCie/ts-predictions/*.csv"

for fname in glob.glob(path):
    print_weekly_view(fname)
