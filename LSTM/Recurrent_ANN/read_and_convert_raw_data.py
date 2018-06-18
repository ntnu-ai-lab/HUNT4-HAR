import sys

"""############################################## PATH MANIPULATION #########################################"""
# TODO: Fix path
if "guest" not in sys.path[0]:
    sys.path.insert(1, '/PATH/TO/HUNT_Project')
else:
    sys.path.insert(1, '/PATH/TO/PycharmProjects')

"""############################################## PATH MANIPULATION #########################################"""

from acrechain.conversion import timesync_from_cwa
import glob
import pandas as pd
import os

def read_raw_data(data_folder_path, samplerate = 50, store_loc = None):
    """
    Reads paths subject folders and calls for conversion of every subject to single subject csv files
    :param data_folder_path: Path to subjects data folder
    :param samplerate: The samplingrate we want to use
    :param store_loc: The location we want to store our output data (if None, data is stored in the input folder)
    :return: A list of paths to the csv subject files
    """
    # If a storage folder is specified, make sure it exists
    if not store_loc == None:
        if not os.path.isdir(store_loc):
            os.mkdir(store_loc)

    subject_csv_paths = []  # List to store the final paths where the subject csvs are stored

    # Read all subject folders
    paths_to_subjects = glob.glob(data_folder_path+"/*")

    # Extract subject names from folders
    subject_names = [sub.replace(data_folder_path+"/", "") for sub in paths_to_subjects]

    # Go through all subjects and call for the generation of a subject csv for every subject
    for i in range(len(paths_to_subjects)):
        # We don't include other files than the actual subjects
        if not ".xlsx" in paths_to_subjects[i]:
            subject_csv_paths.append(convert_raw_to_csv(paths_to_subjects[i], subject_names[i], samplerate, store_loc))

    return subject_csv_paths




def convert_raw_to_csv(subject_path, subject_name, samplerate, store_loc):
    """
    Takes in the path to a subjects, extracts the back and thigh paths.
    Converts the cwa files into csv format and stores them.
    Reads in the csv files and generates a single timestamped csv file for the subject and stores it.
    :param subject_path: path to the subject folder
    :param subject_name: name of the subject
    :param samplerate: the samplerate we want to use
    :param store_loc: The location we want to store our output data
    :return: the path to where the final csv file is stored
    """

    # If no storage location is specified, store data in source folder
    if store_loc == None:
        store_loc = os.path.dirname(subject_path)

    # If storage location is specified, make sure it exists. Otherwise create it
    else:
        store_loc = store_loc+"/"+subject_path.split("raw/",1 )[1].split("/",1)[0]
        if not os.path.isdir(store_loc):
            os.mkdir(store_loc)
    subject_name = str(samplerate)+subject_name


    # Read all files in the subject folder
    subject_files_paths = glob.glob(subject_path+"/*")

    # Filter out the sampling rates we don't want
    subject_files_paths = [sub for sub in subject_files_paths if "_"+str(samplerate)+"_" in sub]

    # Find the back and thigh paths
    for sub in subject_files_paths:
        if "_B" in sub:
            back_cwa = sub
        elif "_T" in sub:
            thigh_cwa = sub

    # Call for the conversion from cwa to csv
    back_csv, thigh_csv, timestamps_csv = timesync_from_cwa(back_cwa, thigh_cwa)

    # Read the newly created csv files into pandas dataframes
    back = pd.read_csv(back_csv)
    thigh = pd.read_csv(thigh_csv)
    timestamps = pd.read_csv(timestamps_csv)

    # Concat the dataframes to create one subject dataframe
    subject_df = pd.concat([timestamps, back, thigh], axis=1)

    # Store the single subject csv in output folder
    subject_df.to_csv(store_loc+"/"+subject_name+"_timesync_time_B_T.csv", header=False, index=False)

    return store_loc+"/"+subject_name+"_timesync_time_B_T.csv"

