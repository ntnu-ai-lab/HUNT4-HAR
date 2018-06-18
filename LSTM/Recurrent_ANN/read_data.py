import pandas as pd
import numpy as np
import glob
import math
from collections import Counter
from TRAINING_NORMALIZATION import get_mean_and_std_lists
import TRAINING_VARIABLES_NN as TV

# Fetch the relabeling dict to do relabeling in relabel_superfluous_labels()
new_labels = TV.getNewLabels()

NUM_TRAINING_FEATURES = 6   # Features in training examples
NUM_LABELS = 1  # Size of labels (1d vector)

def build_training_dataset(validation_subject, path_to_data, sequence_length,
                           num_classes, print_stats = False, generate_one_hot=True,
                           use_most_common_label=True, normalize_data=True, normalization_value_set=1,
                           use_abs_values= False):
    """
    Builds dataset from a list of paths to data points. Splits into training and validation data.
    :param validation_subject: Which subject is used for validation
    :param path_to_data: Path to the data folder for dataset
    :param sequence_length: The sequence length we want to use for our dataset (RNNs use sequential data)
    :param num_classes: The number of classes in our dataset
    :return: Numpy arrays: training examples, training labes, validation examples, validation labels
    """

    training_examples_csv, training_labels_csv = select_csv_files(path_to_data)
    if validation_subject == None:
        training_examples, training_labels = generate_examples_and_labels(training_examples_csv, training_labels_csv,
                                                                          sequence_length, num_classes,
                                                                          print_stats=print_stats,
                                                                          generate_one_hot=generate_one_hot,
                                                                          use_most_common_label=use_most_common_label)
        validation_examples = None
        validation_labels = None
    else:
        training_examples_csv, training_labels_csv, validation_examples_csv, validation_labels_csv = \
            split_into_training_validation(training_examples_csv, training_labels_csv, validation_subject)

        training_examples, training_labels = generate_examples_and_labels(training_examples_csv, training_labels_csv,
                                                                          sequence_length, num_classes,
                                                                          print_stats=print_stats,
                                                                          generate_one_hot=generate_one_hot,
                                                                          use_most_common_label=use_most_common_label)

        validation_examples, validation_labels = generate_examples_and_labels(validation_examples_csv,
                                                                              validation_labels_csv, sequence_length,
                                                                              num_classes, print_stats=print_stats,
                                                                              generate_one_hot=generate_one_hot,
                                                                              use_most_common_label=use_most_common_label)
    # Normalize dataset
    if normalize_data:
        if validation_subject == None:

            print("Normalizing data...")
            training_examples = normalize_dataset(normalization_value_set, training_examples)
        else:

            print("Normalizing data...")
            training_examples = normalize_dataset(normalization_value_set, training_examples)
            validation_examples = normalize_dataset(normalization_value_set, validation_examples)

    # TODO: Should this really happen after normalization?
    if use_abs_values:
        if validation_subject == None:

            print("Using absolute valued data...")
            training_examples = convert_to_absolute_values(training_examples)
        else:

            print("Using absolute valued data...")
            training_examples = convert_to_absolute_values(training_examples)
            validation_examples = convert_to_absolute_values(validation_examples)




    return training_examples, training_labels, validation_examples, validation_labels

def convert_to_absolute_values(examples):
    """
    Converts all sensor values to their magnitude values instead, discarding the sign
    :param examples: Input examples
    :return: abs_value examples
    """
    for i in range(examples.shape[2]):
        examples[:, :, i] = np.abs(examples[:, :, i])
    return examples

def normalize_dataset(normalization_set, examples):
    """
    Normalizes a dataset, using the specified normalization values
    :param normalization_set: The set of normalization values to use
    :param examples: Input dataset to be normalized
    :return: Normalized dataset
    """
    mean_list, std_list = get_mean_and_std_lists(normalization_set)
#    print examples.shape
    for i in range(len(mean_list)):
        examples[:,:, i] -= mean_list[i]
        examples[:,:, i] /= std_list[i]

    return examples

def select_csv_files(path_to_data, in_lab = False):
    """
    Separates csvs in a folder to examples and labels
    :param path_to_data: path to training data folder
    :return: lists: one for training example csvs and one for training label csvs
    """
    # Read all paths in specified folder
    example_folders_paths = glob.glob(path_to_data + "/*")
    csv_file_paths = []

    # Select only files with .csv ending
    for example_folder in example_folders_paths:
        csv_file_paths.append(glob.glob(example_folder + "/*.csv"))

    # Split into example and label files
    training_examples_csv = []
    training_labels_csv = []
    for csv_file_path in csv_file_paths:
        if in_lab:
            for csv_file in csv_file_path:
                if 'labels' in csv_file:
                    training_labels_csv.append(csv_file)
                else:
                    training_examples_csv.append(csv_file)
        else:
            for csv_file in csv_file_path:
                if 'Axivity' in csv_file:
                    training_examples_csv.append(csv_file)
                elif "GoPro" in csv_file:
                    training_labels_csv.append(csv_file)

    return training_examples_csv, training_labels_csv

def generate_examples_and_labels(examples_csv, labels_csv, sequence_length, num_classes, print_stats = False, use_most_common_label = True, generate_labels = True, generate_one_hot=True):
    """
    Generates examples and labels from the inputed paths to csvs.
    :param examples_csv: Path to example csvs
    :param labels_csv: Path to label csvs
    :param sequence_length: The length we want our example sequence to be 100 for one second with 100Hz data
    :param num_classes: The number of different classes in our dataset
    :return: numpy arrays: examples and one-hot labels
    """
    # Generates a list of dataframes from the given paths
    training_examples_list = fetch_training_data(examples_csv, print_stats)
    if generate_labels:
        training_labels_list = fetch_training_labels(labels_csv)

    # Converts list of dataframes to list of numpy arrays
    training_examples = convert_to_list_of_numpy_array(training_examples_list)
    if generate_labels:
        training_labels = convert_to_list_of_numpy_array(training_labels_list)



    # Combines list of numpy arrays into a single numpy array
    training_examples = combine_numpy_list_to_single_array(training_examples, sequence_length, NUM_TRAINING_FEATURES)

    if generate_labels:
        if print_stats:
            print("Generating labels")
        training_labels = combine_numpy_list_to_single_array(training_labels, sequence_length, NUM_LABELS)
        # Relabel sequences with the most commonly occurring label in that sequence
        if use_most_common_label:
            if print_stats:
                print("Using most common label")
            training_labels = use_most_common_label_in_sequence(training_labels)

        # Convert from int labels to one-hot vectors
        if generate_one_hot:
            if print_stats:
                print("Generating one-hot vectors")
            training_labels = generate_one_hot_vectors(training_labels, num_classes, use_most_common_label)

        return training_examples, training_labels
    else:
        return training_examples

def split_into_training_validation(dataset_examples, dataset_labels, validation_subject):
    """
    Splits our dataset into training and validation datasets for Leave One Subject Out (LOSO) training and validation.
    :param dataset_examples: All example paths
    :param dataset_labels: All label paths
    :param validation_subject: The subject used for this LOSO iteration
    :return: Lists of paths: training examples, training labels, validation examples, validation labels
    """
    # Generate lists for each set of examples and labels
    training_examples_csvs = []
    training_labels_csvs = []
    validation_examples_csvs = []
    validation_labels_csvs = []
    print(validation_subject)
    for example in dataset_examples:
        if validation_subject in example:
            validation_examples_csvs.append(example)
        else:
            training_examples_csvs.append(example)

    for label in dataset_labels:
        if validation_subject in label:
            validation_labels_csvs.append(label)
        else:
            training_labels_csvs.append(label)
    return training_examples_csvs, training_labels_csvs, validation_examples_csvs, validation_labels_csvs

def fetch_training_data(examples_csv, print_stats):
    """
    Goes over the list of example paths and reads the csv in that path into pandas dataframes. Iterates over only the
    BACK examples (every second example) and hardcodes the THIGH example.
    :param examples_csv: A list of example paths
    :return: List, containing pandas dataframes
    """
    training_examples_list = []
    # Iterates over every second element in the training_examples_csv list
    for i in [x for x in range(len(examples_csv) - 1) if x % 2 == 0]:
        # Select thigh and back sensors
        if "THIGH" in examples_csv[i]:
            thigh = examples_csv[i]
            back = examples_csv[i+1]
        else:
            thigh = examples_csv[i+1]
            back = examples_csv[i]
        if print_stats:
            print(thigh + " " + back)
        # Read both thigh and back datafiles
        training_examples_list.append(read_csv_file([thigh, back], examples=True))
    return training_examples_list

def fetch_training_labels(labels_csv):
    """
       Goes over the list of label paths and reads the csv in that path into pandas dataframes.
       :param labels_csv: A list of label paths
       :return: List, containing pandas dataframes
       """
    # Iterates over all the elements training_lables_csv list
    return [read_csv_file(training_label, False) for training_label in labels_csv]

def read_csv_file(path_to_file, examples):
    """
    Reads a csv path into pandas dataframe
    :param path_to_file: path to the csv file
    :param examples: boolean, True = examples, False = Labels
    :return: pandas dataframe
    """
    if examples:
        #print path_to_file
        training_THIGH = pd.read_csv(path_to_file[0])
        training_BACK = pd.read_csv(path_to_file[1])
        training_example = pd.concat([training_BACK, training_THIGH], axis=1)
    else:
        #print path_to_file
        training_example = pd.read_csv(path_to_file)
        #for col in training_example:
        #    print training_example[col].unique()
    return training_example

def convert_to_list_of_numpy_array(dataframes_list):
    """
    Converts from pandas dataframes to numpy arrays
    :param dataframes_list: List of pandas dataframes
    :return: List of numpy arrays
    """
    data_list = []
    # Convert from dataframe to numpy array and append to list
    for i in range(len(dataframes_list)):
        data_list.append(dataframes_list[i].values)

    return data_list

def combine_numpy_list_to_single_array(data_list, sequence_length, num_features):
    """
    Goes over a list of numpy arrays and combines them into a single numpy array along the depth axis
    :param data_list: List of numpy arrays
    :param sequence_length: lenght of the sequences we want
    :param num_features: The number of features we want in our arrays, 6 for training and 1 for labels
    :return: Single numpy array
    """
    #print len(data_list)
    #print data_list[0].shape
    # Go over training_examples list and convert it into a single numpy array
    data_array = np.concatenate(data_list)

    # Remove examples to make number of examples fit within sequence length
    data_array = data_array[0:(int(math.floor(data_array.shape[0] / sequence_length)) * sequence_length)]

    # Reshape examples to shape = [examples, sequence_length, 6] for the RNN
    data_array = data_array.reshape(
        [int(data_array.shape[0] / sequence_length), sequence_length, num_features])

    return data_array

def use_most_common_label_in_sequence(training_labels):
    """
    Goes over all the all the label sequences and finds the most common label in each sequence
    :param training_labels: list of label sequences
    :return: list of most common label in each sequence
    """
    new_training_labels = []
    for i in range(len(training_labels)):
        label = find_most_common_label(training_labels[i])
        new_training_labels.append(label)

    return np.asarray(new_training_labels)

def find_most_common_label(l):
    """
    Finds the most commonly occurring label in a sequence
    :param l: sequence
    :return: most common label
    """
    l = np.concatenate(l)
    word_counts = Counter(l)
    most_common_label = word_counts.most_common(1)
    return most_common_label[0][0]

def generate_one_hot_vectors(new_training_labels, num_classes, use_most_common_label):
    """
    Generates one-hot labels from int labels
    :param new_training_labels: list of int labels
    :param num_classes: number of classes in our datset
    :return: array of one-hot label vectors
    """
    #print("prev max: " + str(np.max(new_training_labels)))
    new_training_labels = relabel_superfluous_labels(new_training_labels)
    #print("new max: " +str(np.max(new_training_labels)))
    if use_most_common_label:
        training_labels = []
        for i in range(len(new_training_labels)):
            zeros = np.zeros((num_classes), dtype=np.int)
            #print int(new_training_labels[i])
            zeros[int(new_training_labels[i]) - 1] = 1
            training_labels.append(zeros)
        training_labels = np.asarray(training_labels)
    else:
        training_labels=np.zeros(shape=[new_training_labels.shape[0], new_training_labels.shape[1], num_classes], dtype=np.int)
        for i in range(len(new_training_labels)):
            for j in range(new_training_labels.shape[1]):
                training_labels[i, j, [int(new_training_labels[i,j])-1]] = 1
                #print int(new_training_labels[i,j])
                #print zeros[i, j, :]


    return training_labels

def build_prediction_data(path_to_data, sequence_length, use_abs_values = False, normalize_data = True, normalization_value_set=1, num_features = 6):
    """
    Takes the path to a prediction subject, reads it and returns a numpy array of the
    prediction inputs x
    :param path_to_data: path to prediction subject file
    :param normalize_data: whether or not we should normalize our data
    :param normalization_value_set: which normalization values to use
    :return: prediction inputs
    """
    # TODO, now assumes just one prediction file, in the future we should make it more genereal
    subject_dataframe = read_csv_file(path_to_data, examples=False)
    # Convert to list of arrays
    subject_numpy_array = convert_to_list_of_numpy_array([subject_dataframe])
    # Convert to single array
    subject_numpy_array = np.reshape(np.asarray(subject_numpy_array),
                                     newshape=[np.asarray(subject_numpy_array).shape[1],
                                               np.asarray(subject_numpy_array).shape[2]])

    time_stamps, sensor_measurements = generate_prediction_sequences(subject_numpy_array, sequence_length, num_features)

    if normalize_data:
        sensor_measurements = normalize_dataset(normalization_value_set, sensor_measurements)

    if use_abs_values:
        sensor_measurements = convert_to_absolute_values(sensor_measurements)

    return sensor_measurements, time_stamps

def generate_prediction_sequences(data_numpy_array, sequence_length, num_features):
    """
    Generate sequences of lenght "sequence_length" and returns an array of sequences and an array
    of timestamps, where the final timestamp for a given sequence is used.
    :param data_numpy_array: Input data array [timestamps, sensor_data(6)]
    :param sequence_length: The length we want for the generated our sequences
    :param num_features: The number of features used in sensor data, ususally 6
    :return: timestamps and sequences
    """
    timestamps = data_numpy_array[:, 0]
    sensor_measurements = data_numpy_array[:, 1:7]
    test_num = 0

    timestamps = use_last_timestamp_in_sequence(timestamps, sequence_length)
    sensor_measurements = combine_numpy_list_to_single_array([sensor_measurements], sequence_length, num_features)


    return timestamps, sensor_measurements

def use_last_timestamp_in_sequence(timestamps,sequence_length):
    """
    Generates a numpy array of the last timestamps for a set of sequences
    :param timestamps: The original timestamps
    :param sequence_length: The length of our sequences
    :return: An array of timestamps
    """
    new_training_labels = []
    for i in range(len(timestamps)):
        if i%sequence_length == 0:
            label = timestamps[i]
            new_training_labels.append(label)

    return np.asarray(new_training_labels)

def relabel_superfluous_labels(labels):
    """
    Relabels the original labels
    for example, all different versions of "lying(****)" is converted to "lying"
    :param labels:  The original labels
    :return:    The relabeled dataset
    """
    for k in new_labels:
        np.place(labels, labels == k, [new_labels[k]])
    return labels


