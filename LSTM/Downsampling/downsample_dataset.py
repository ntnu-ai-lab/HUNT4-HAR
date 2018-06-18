import pandas as pd
import numpy as np
import Downsampling.downsampling_functions as df
from Recurrent_ANN import read_data as rd
import os

path_to_data = "/PATH/TO/dataset/"
if "dataset" in path_to_data:
    in_lab = True

store_data_path_builder = "/PATH/TO/Downsampled-data/"
downsampling_functions = [df.decimate_data, df.resample_poly_data, df.average_two_and_two,
                          df.sample_every_nth, df.resample_data]
downsampling_function_names = ["DECIMATE", "RESAMPLE_POLY", "TWO_AND_TWO_AVG",
                              "EVERY_OTHER", "RESAMPLE"]


def do_examples(examples):
    subject_list= []
    subject_shape = []
    for example in examples:
        new_name = example.strip(path_to_data)
        subject_name = new_name[0:3]
        if not os.path.isdir(store_data_path+"/"+subject_name):
            os.makedirs(store_data_path+"/"+subject_name)
        new_name =new_name[4:]
        #print subject_name
        #print new_name
        #print store_data_path+"/"+subject_name+"/"+new_name
        if ".csv.csv" in new_name:
            #print new_name
            new_name = new_name.replace(".csv.csv", ".csv")
        example_dataframe = df.do_downsampling(pd.read_csv(example), downsampling_function)
        if subject_name not in subject_list:
            print(subject_name)
            subject_list.append(subject_name)
            subject_shape.append(example_dataframe.shape[0])
        example_dataframe.to_csv(store_data_path+"/"+subject_name+"/"+new_name, index=False, header=False)

    return subject_list, subject_shape

def do_labels(labels,subject_list, subject_shape):
    print(subject_list)
    iterator = 0
    for label in labels:
        new_name = label.strip(path_to_data)
        subject_name = new_name[0:3]
        if not os.path.isdir(store_data_path+"/"+subject_name):
            os.makedirs(store_data_path+"/"+subject_name)
        new_name =new_name[4:]
        #print subject_name
        #print new_name
        #print store_data_path+"/"+subject_name+"/"+new_name
        if ".csv.csv" in new_name:
            #print new_name
            new_name = new_name.replace(".csv.csv", ".csv")
        label_dataframe = df.do_downsampling(pd.read_csv(label), df.sample_every_nth)
        print("Labels: " + subject_name + " Examples: " + str(subject_list[iterator]))
        if subject_name == subject_list[iterator]:
            if label_dataframe.shape[0] != subject_shape[iterator]:

                label_dataframe = label_dataframe[0:subject_shape[iterator]]


        label_dataframe.to_csv(store_data_path+"/"+subject_name+"/"+new_name, index=False, header=False)
        iterator +=1

examples, labels = rd.select_csv_files(path_to_data, in_lab=in_lab)
print(examples)
print(labels)
iterator = 0
for downsampling_function in downsampling_functions:
    store_data_path = store_data_path_builder+downsampling_function_names[iterator]+ "/"
    if not os.path.isdir(store_data_path):
        os.makedirs(store_data_path)
    print(downsampling_function_names[iterator])
    subject_list, subject_shape = do_examples(examples)
    do_labels(labels, subject_list, subject_shape)


    iterator +=1


