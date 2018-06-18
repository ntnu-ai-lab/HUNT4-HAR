import predict as predict
import time
import sys
import glob
import read_and_convert_raw_data as rcrd
import os

# The sample rate we want to use
samplerate = "50"
# The path to our raw data folder
raw_data_folder = "/PATH/TO/DATA/Prediction_Dataset/raw/HzXX"
# The path we want to store our converted data in
store_converted_data_path = "/lhome/haakom/HUNT_Project/DATA/Prediction_Dataset/exported-csv"

# Path to our pretrained weights
weights_to_pretrained_model = "/PATH/TO/Recurrent_ANN/Best_model_Weights/05-03-2018_10:13:36_Twin_Pure-LSTM_32Cells_250_RESAMPLE-OOL"

# Call for conversion from .cwa to .csv
rcrd.read_raw_data(raw_data_folder, samplerate=50, store_loc=store_converted_data_path)

# Build path to .csv files for reading files
prediction_csv_data_folder = store_converted_data_path + "/" +raw_data_folder.split("raw/",1)[1]

# Extract all prediction subjects from folder
all_subjects = glob.glob(prediction_csv_data_folder+"/*.csv")

# Only use subjects with the correct samplerate
prediction_subjects = []
for subject in all_subjects:
    if samplerate in subject.split(os.sep)[-1]:
        prediction_subjects.append(subject)

# Define parameters for the network
seq_length = 250    # Length of a prediction sequence
n_layers = 1    # Number of layers in network
n_units = 32    # Number of units in each layer
bidirectional = True    # If we are building a bidirectional network

# Handle terminal inputs
term_in = sys.argv
if len(term_in) == 4:
    seq_length = term_in[1]
    n_layers = term_in[2]
    n_units = term_in[3]

elif len(term_in) == 3:
    n_layers = term_in[1]
    n_units = term_in[2]

elif len(term_in) == 2:
    n_units = term_in[1]

# Print network stats
print("Sequence length: %i" % seq_length)
print("Number of layers: %i" % n_layers)
print("Number of units per layer: %i" % n_units)

start = time.time()

# Use pretrained weights to to prediction using stateful LSTM
predict.do_prediction(num_layers=n_layers, num_units=n_units, bidirectional=bidirectional,
                                    predict_subjects=prediction_subjects,
                                    weights_location=weights_to_pretrained_model)  # Use model to predict

# Print time stats
print("Total time: %s" % (time.time() - start))