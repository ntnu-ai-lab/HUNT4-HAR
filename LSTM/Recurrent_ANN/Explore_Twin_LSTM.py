from keras.layers import Input, Dropout, Activation, Concatenate, Bidirectional, Dense, BatchNormalization, Average
from keras.models import Model
import read_data as rd
import numpy as np
import sys
from Custom_Callbacks import Overfitting_callback, LoadBestWeigtsReduceLR
from keras.callbacks import EarlyStopping, ModelCheckpoint
import STOP_TRAINING_REASON as tr
import TRAINING_VARIABLES_NN as TV


"""


-------------------------DOES NOT SAVE MODEL-------------------------------------

THIS FILE IS ONLY USED TO TEST DIFFERENT DNN ARCHITECTURES ON THE TRAINING DATA.
IT VALIDATES ON ALL THE TRAINING SUBJECT THROUG A LEAVE ONE SUBJECT OUT PROTOCOL
AND REPORTS THE FINAL ACCURACIES FOR EVERY SUBJECT AS WELL AS THE AVERAGE ACCURACY

-------------------------DOES NOT SAVE MODEL-------------------------------------


"""
# Static variables
BATCH_SIZE = 4096   # Batch size for training
NUM_ACTIVITIES = 19 # Number of classes for our model
NUM_FEATURES = 6    # Number of features in input thigh-xyz, back-xyz
NUM_EPOCHS = 512    # Maximum number of epochs per subject
predict_sequences = False   # We are not predicting sequences
SEQUENCE_LENGTH = 250   # Sequence length for RNN sequence input
OF_patience_factor = 2  # Patience of overfitting callback
ES_patience_factor = 6  # Patience of Early stopping callback
OUT_OF_LAB_SUBJECTS = ["006", "008", "009", "010", "011",
                "012", "013", "014", "015", "016",
                "017", "018", "019", "020", "021",
                "022"]  # List of all the subjects in the out-of-lab dataset
IN_LAB_SUBJECTS = ["S01", "S03", "S05", "S06", "S07",
                   "S08", "S09", "S10", "S11", "S12",
                   "S13", "S14", "S15", "S16"]  # List of all the subjects in the in-lab dataset

# Combine OOL and IL dataset subjects for validation on all subjects
ALL_SUBJECTS = OUT_OF_LAB_SUBJECTS + IN_LAB_SUBJECTS

# Load training files
dataset_path, dataset_name = TV.getDataset()

evaluation_list = []  # List for storing evalutations for each subject

# Go over all subjects in a leave-one-subject-out manner and train a model
for subject in ALL_SUBJECTS:
    tr.overfitting = False
    print("Building network...")

    # Build model on CPU or GPU based on system path
    if "guest" not in sys.path[1]:
        from keras.layers import CuDNNLSTM

        # Build LSTM RNN GPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES // 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES // 2])
        nn1 = Bidirectional(CuDNNLSTM(units=32, return_sequences=True), merge_mode='sum')(nn_in1)
        nn2= Bidirectional(CuDNNLSTM(units=32, return_sequences=True), merge_mode='sum')(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = Dropout(0.9)(nn)
        nn = Bidirectional(CuDNNLSTM(units=NUM_ACTIVITIES, return_sequences=False), merge_mode='sum')(nn)
        nn = Activation(activation="softmax")(nn)

    else:
        from keras.layers import LSTM

        # Build LSTM RNN CPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES // 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES // 2])
        nn1 = Bidirectional(LSTM(units=32, return_sequences=False))(nn_in1)
        nn2 = Bidirectional(LSTM(units=32, return_sequences=False))(nn_in2)
        nn = Concatenate(axis=1)([nn1, nn2])
        nn = Dropout(0.9)(nn)
        nn = Dense(NUM_ACTIVITIES)(nn)
        nn = Activation(activation="softmax")(nn)

    # Compile model and prepare graph
    model = Model(inputs=[nn_in1, nn_in2], outputs=nn)
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    # Print model summary in terminal
    model.summary()

    # Automatic selection of dataset location depending on which machine the code is running on
    print("Reading data...")
    train_x, train_y, val_x, val_y = rd.build_training_dataset(subject, dataset_path,
                                                                   SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                                   use_most_common_label=not predict_sequences,
                                                                   print_stats=False,
                                                                   normalize_data=True,
                                                                   normalization_value_set=dataset_name)





    # Separate thigh and back training data into separate "channels" for training
    train_x1= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])
    train_x2= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])

    # Separate thigh and back training data into separate "channels" for validation
    val_x1 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])
    val_x2 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])

    # Fill the channels for training data
    for example in range(train_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            train_x1[example, i, :] = train_x[example, i, 0:3]
            train_x2[example, i, :] = train_x[example, i, 3:6]

    # Fill the channels for validation data
    for example in range(val_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            val_x1[example, i, :] = val_x[example, i, 0:3]
            val_x2[example, i, :] = val_x[example, i, 3:6]

    # Train model
    train_history = model.fit([train_x1, train_x2], train_y,
                                  epochs=NUM_EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  validation_data=([val_x1, val_x2], val_y),
                                  callbacks=[ModelCheckpoint(filepath="/tmp/weights.hdf5",
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         mode='min'),
                                         LoadBestWeigtsReduceLR(patience=4,
                                                                weights_path="/tmp/weights.hdf5",
                                                                verbose=1),
                                         EarlyStopping(verbose=1, patience=ES_patience_factor)]) # Training

    # Check if reason for stopping was overfitting or Early stopping and append appropriate training score
    if tr.overfitting:
        try:
            print("Appending: " + str(train_history.history['val_acc'][-(OF_patience_factor+1)]) + " to eval list")
            evaluation_list.append(train_history.history['val_acc'][-(OF_patience_factor+1)])
        except IndexError:
            print("Appending: " + str(train_history.history['val_acc'][-(OF_patience_factor)]) + " to eval list")
            evaluation_list.append(train_history.history['val_acc'][-(OF_patience_factor)])
    else:
        print("Appending: " + str(train_history.history['val_acc'][-(ES_patience_factor + 1)]) + " to eval list")
        evaluation_list.append(train_history.history['val_acc'][-(ES_patience_factor + 1)])



# Print final training statistics
print("Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list))))
print("Evaluation List: " + str(evaluation_list))
