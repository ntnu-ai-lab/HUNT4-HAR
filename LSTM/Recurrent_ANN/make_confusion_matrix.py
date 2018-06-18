def build_model_and_make_confusion_matrix():
    import pandas as pd
    import read_data as rd
    import time
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import MODEL_BUILDING as MB
    import matplotlib as mpl
    mpl.use("PDF")
    import matplotlib.pyplot as plt


    # Set static variables
    BATCH_SIZE = 512
    NUM_ACTIVITIES = 19
    NUM_FEATURES = 6
    NUM_EPOCHS = 22
    predict_sequences = False
    NUM_UNITS = 32
    NUM_LAYERS = 1
    use_most_recent_model_weights = False
    normalize = True

    # Path to the model weights
    filepath = 'Trained_Models_Weights'

    # Sort all models in the path
    files = sorted([f for f in os.listdir(filepath)])

    # Select model to use based on if we want the most recent model or a manually selected one
    if use_most_recent_model_weights:
        model_name = files[-1]
    else:
        model_name = "05-03-2018_10:13:36_Twin_Pure-LSTM_32Cells_250_RESAMPLE-OOL"

    # Build paths and dates, used for storage path generation and normalizing dataset selection
    model_path = filepath + "/" + model_name
    model_date = model_path.split("Weights/", 1)[1][0:19]

    # Automatically determine the sequence length used when training the model
    if "_50_" in model_name:
        sequence_length = 50
    elif "_100_" in model_name:
        sequence_length = 100
    elif "_150_" in model_name:
        sequence_length = 150
    elif "_250_" in model_name:
        sequence_length = 250
    elif "_500_" in model_name:
        sequence_length = 500
    elif "_1000_" in model_name:
        sequence_length = 1000



    # Extract the name of the dataset used to train model
    print(model_path)
    trained_on_dataset = model_path.split('Cells_' + str(sequence_length) + '_', 1)[1]


    model,_ = MB.build_network(keep_state=False, batch_size=BATCH_SIZE,
                             sequence_length=sequence_length, num_features=NUM_FEATURES,
                             num_units=NUM_UNITS, num_layers=NUM_LAYERS,
                             num_activities=NUM_ACTIVITIES)

    # Load model weights from trained model
    print(model_path)
    print("Loading model: " + model_name)
    model.load_weights(model_path)
    print("Success!")

    t0 = time.time()

    dataset_path = "Downsampled-data/RESAMPLE/OOL"


    # Load sensor and timestamp data
    print("Loading sensor data...")
    train_x, train_y, val_x, val_y = rd.build_training_dataset(None, dataset_path,
                                                               sequence_length, NUM_ACTIVITIES,
                                                                use_most_common_label=not predict_sequences,
                                                                print_stats=False,
                                                                normalize_data=True,
                                                                normalization_value_set="RESAMPLE-OOL",
                                                                use_abs_values=False,
                                                               generate_one_hot=False)




    # Create two data "channels"
    pred_x1 = np.zeros(shape=[train_x.shape[0], sequence_length, 3])
    pred_x2 = np.zeros(shape=[train_x.shape[0], sequence_length, 3])

    # Fill data channels with our real sensor data from subject
    for example in range(train_x.shape[0]):
        for i in range(sequence_length):
            pred_x1[example, i, :] = train_x[example, i, 0:3]
            pred_x2[example, i, :] = train_x[example, i, 3:6]

    # Feed data into model and do predictions
    t1 = time.time()
    print("Doing Predictions...")
    raw_model_predictions = model.predict([pred_x1, pred_x2], batch_size=BATCH_SIZE, verbose=1)
    t2 = time.time()

    pred_arg_maxes = []  # List for storing predictions
    num_uncertain = 0  # The number of uncertain predictions
    uncertainty_threshold = 0.4  # The threshold used to determine whether a prediction is uncertain or not

    # Go over all predictions
    for prediction in raw_model_predictions:
        # Check if model is uncertain
        if prediction[np.argmax(prediction)] < uncertainty_threshold:
            # If it is, add uncertainty marker (-1) and increment number of uncertain predictions
            pred_arg_maxes.append(-1)
            num_uncertain += 1
        # If not uncertain
        else:
            # Add highest prediction class
            pred_arg_maxes.append(np.argmax(prediction) + 1)


    y_actu = pd.Series(train_y)
    y_pred = pd.Series(pred_arg_maxes)
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

def make_confusion_matrix(model, pred_x1, pred_x2, actu_y, model_name):
    import pandas as pd
    import numpy as np
    labels = np.zeros(shape=actu_y.shape[0])

    for i in range(actu_y.shape[0]):
        labels[i] = np.argmax(actu_y[i])+1

    raw_model_predictions = model.predict([pred_x1, pred_x2], batch_size=512, verbose=1)

    pred_arg_maxes = []  # List for storing predictions
    num_uncertain = 0  # The number of uncertain predictions
    uncertainty_threshold = 0.4  # The threshold used to determine whether a prediction is uncertain or not

    # Go over all predictions
    for prediction in raw_model_predictions:
        # Check if model is uncertain
        if prediction[np.argmax(prediction)] < uncertainty_threshold:
            # If it is, add uncertainty marker (-1) and increment number of uncertain predictions
            pred_arg_maxes.append(-1)
            num_uncertain += 1
        # If not uncertain
        else:
            # Add highest prediction class
            pred_arg_maxes.append(np.argmax(prediction) + 1)

    y_actu = pd.Series(labels)
    y_pred = pd.Series(pred_arg_maxes)
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    df_confusion.to_csv("/lhome/haakom/HUNT_Project/Haakon_Recurrent_ANN/confusion_matrices/"+model_name+".csv")
    print(df_confusion)

