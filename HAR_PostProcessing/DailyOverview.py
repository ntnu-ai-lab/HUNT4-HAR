import pandas as pd
import Dictinaries
import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
import os, re, glob, sys
root_dir = os.path.dirname(os.path.abspath(__file__))

term_in = sys.argv
def print_weekly_view(filename, model_name, loop):

    maxSampleDay = 14400 # 14400 for 50Hz at seq_length = 50 - assumes only missing data the first and the last day

    subject_file = os.path.join(root_dir, filename)
    subjectid = model_name.split("Stateful_",1)[1]
    subjectid = subjectid.split("_normalized",1)[0]
    #subjectid = "7050"#list(map(int, re.findall('\d+', filename))).pop().__str__()

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
        -1: "black",
        1: "forestgreen",
        2: "red",
        6: "lightyellow",
        7: "lightcyan",
        8: "skyblue",
        9: "purple",
        13: "darkorange",
        19: "lightyellow",
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
    cb0.set_label('Date: '+ days.iloc[0].__str__() +' (black: model uncertain, purple: transition, orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

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
        cb0.set_label('Date: '+ days.iloc[current].__str__() +' (black: model uncertain, purple: transition, orange: cycling, red: running, green: walking, yellow: standing, light blue:sitting, blue: lying)')

    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    # If we are running a looping prediction, we create a folder to store all plots in
    if loop:
        ts = model_name.split(os.sep)[0]
        model_name = model_name.split(os.sep)[1]

    # We are manually setting where to store here, so the user will have to change this as it is computer specific.
    # This is done because we write code on one machine, but run it on another and for some reason the paths do not
    # follow the same structure...
        if not os.path.isdir("/lhome/haakom/HUNT_Project/Haakon_Recurrent_ANN/plots/"+ts):
            os.mkdir("/lhome/haakom/HUNT_Project/Haakon_Recurrent_ANN/plots/" + ts)
        plt.savefig("/lhome/haakom/HUNT_Project/Haakon_Recurrent_ANN/plots/"+ts+"/Daily-BarChart-" +model_name +"-"+subjectid)

    # Otherwise we just store them in plots
    else:
        plt.savefig(
            "/lhome/haakom/HUNT_Project/Haakon_Recurrent_ANN/plots/Daily-BarChart-" + model_name + "-" + subjectid)

        #plt.savefig("plots/Steinkjer_adolescents-" + subjectid + '-new-sync')


#create single
#print_weekly_view("data/1103_timestamped_predictions.csv")

#create all
def do_plot(path, loop):
    #path = "/home/guest/PycharmProjects/HUNT_Haakom/Haakon_Recurrent_ANN/Predictions/08-03-2018_16:43:23_Twin_Pure-LSTM_32Cells_250_RESAMPLE-OOL_Stateful_7050_normalized_predictions_uncertainty:0,147.csv"
    model_name = path.split('Predictions/',1)[1][:-4]



    for fname in glob.glob(path):
        print_weekly_view(fname, model_name, loop)
if "run" in term_in:
    print(term_in)
    do_plot(term_in[2])