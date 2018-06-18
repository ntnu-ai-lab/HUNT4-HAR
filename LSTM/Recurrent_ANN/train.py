def train_a_model(sequence_length=250, num_layers=1, num_units=32,
                  bidirectional=True, train_dataset="Downsampled-data/RESAMPLE/OOL"):
    import read_data as rd
    import numpy as np
    import sys
    from Custom_Callbacks import Overfitting_callback, LoadBestWeigtsReduceLR
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import STOP_TRAINING_REASON as tr
    import TRAINING_VARIABLES_NN as TV
    import os
    import time
    import MODEL_BUILDING as MB
    import make_confusion_matrix as mcm




    # Load training files
    dataset_path, dataset_name = TV.getDataset(train_dataset)

    print("Dataset: " + dataset_name)

    # Set static variables
    BATCH_SIZE = 2048    # Batch size for training
    NUM_EPOCHS = 1024   # Number of epochs for training
    NUM_ACTIVITIES = 19 # Number of activities we are predicting (default = 19)
    NUM_FEATURES = 6    # Number of features in input, 3 axes for both thigh and back = 6
    predict_sequences = False   # We are only doing one prediction per sequence (default: False)
    SEQUENCE_LENGTH = sequence_length   # Set sequence length (250 examples / 50Hz data = 5sec sequences)
    NUM_UNITS = num_units  # Number of LSTM cells in the first LSTM layer
    NUM_LAYERS = num_layers # Number of LSTM layers before prediction layer
    keep_state = False # We are not training with stateful LSTMs (default = False)
    val_subject = "022"    # The subject used for validation
    tr.overfitting = False  # Set stop_training_reason

    # Reset the number of epochs if we train with stateful LSTM
    if keep_state:
        NUM_EPOCHS = 22

    # Check if store path exists, if not, generate it
    store_path = sys.path[0]+"/Trained_Models"
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    store_weights_path = sys.path[0]+"/Trained_Models_Weights"
    if not os.path.isdir(store_weights_path):
        os.mkdir(store_weights_path)

    model, net_name = MB.build_network(keep_state=False, batch_size=BATCH_SIZE,
                             sequence_length=SEQUENCE_LENGTH, num_features=NUM_FEATURES,
                             num_layers=NUM_LAYERS, num_units=NUM_UNITS,
                             num_activities=NUM_ACTIVITIES,
                             bidirectional=bidirectional)
    # Load dataset
    print("Reading data...")
    train_x, train_y, val_x, val_y = rd.build_training_dataset(val_subject, dataset_path,
                                                                       SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                                       use_most_common_label=not predict_sequences,
                                                                       print_stats=False,
                                                                       normalize_data=True,
                                                                       normalization_value_set=dataset_name,
                                                                       use_abs_values=False)

    # If we train with stateful LSTM, we must make the input fit into the batch size
    if keep_state:
        train_x = train_x[0:((train_x.shape[0]//BATCH_SIZE)*BATCH_SIZE)]
        train_y = train_y[0:((train_y.shape[0]//BATCH_SIZE)*BATCH_SIZE)]

        val_x = val_x[0:((val_x.shape[0] // BATCH_SIZE) * BATCH_SIZE)]
        val_y = val_y[0:((val_y.shape[0] // BATCH_SIZE) * BATCH_SIZE)]



    # Generate two input channels for thigh and back for training
    train_x1= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES//2])
    train_x2= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES//2])

    # Generate two input channels for thigh and back for validation
    val_x1 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])
    val_x2 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES // 2])

    # Fill input channels with training data
    for example in range(train_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            train_x1[example, i, :] = train_x[example, i, 0:3]
            train_x2[example, i, :] = train_x[example, i, 3:6]

    # Fill input channels with validation data
    for example in range(val_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            val_x1[example, i, :] = val_x[example, i, 0:3]
            val_x2[example, i, :] = val_x[example, i, 3:6]



    # Train model on training data
    model.fit([train_x1, train_x2], train_y,
                              epochs=NUM_EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_data=([val_x1, val_x2], val_y),
                              shuffle=not keep_state,
                              callbacks=[ModelCheckpoint(filepath="/tmp/weights.hdf5",
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         mode='min'),
                                         LoadBestWeigtsReduceLR(patience=4,
                                                                weights_path="/tmp/weights.hdf5",
                                                                verbose=1),
                                         EarlyStopping(verbose=1, patience=10)])

    # Generate timestamp fro model name
    ts = time.gmtime()
    ts = time.strftime("%d-%m-%Y_%H:%M:%S", ts)

    # Generate model name for storage
    if not keep_state:
        model_name = ts+"_Pure-LSTM_" +net_name + "-Cells_" + str(SEQUENCE_LENGTH) + "_" + dataset_name
    else:
        model_name = ts + "_Pure-LSTM_" + net_name +"-stateful" +"-Cells_" + str(SEQUENCE_LENGTH) + "_" + dataset_name

    # Select the best preforming weights
    model.load_weights("/tmp/weights.hdf5")

    # Make a confusion matrix for the newly trained model
    mcm.make_confusion_matrix(model, train_x1, train_x2, train_y, model_name)

    # Save only model weights
    print("Saving model weights to: " + store_weights_path+ "/" + model_name)
    model.save_weights(filepath=store_weights_path + "/" + model_name)

    # Save full model
    print("Saving full model to: " + store_path+ "/" + model_name)
    model.save(filepath=store_path + "/" + model_name, include_optimizer=True)

    return  store_weights_path+ "/" + model_name

