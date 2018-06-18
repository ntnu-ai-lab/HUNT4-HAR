import sys
term_in = sys.argv

def do_prediction(num_layers=1, num_units=32, bidirectional=True,
                  use_most_recent_weights=False, predict_subject=None,
                  weights_location=None):
    import read_data as rd
    import time
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import MODEL_BUILDING as MB


    # Set static variables
    BATCH_SIZE = 512
    NUM_ACTIVITIES = 19
    NUM_FEATURES = 6
    NUM_UNITS = num_units
    NUM_LAYERS = num_layers
    use_most_recent_model_weights = use_most_recent_weights
    normalize = True

    # Path to the best model weights
    filepath='/PATH/TO/Recurrent_ANN/Trained_Models_Weights'



    # Select model to use based on if we want the most recent model or a manually selected one
    if use_most_recent_model_weights:

        # Sort all models in the path
        files = sorted([f for f in os.listdir(filepath)])

        # Select most recent model weights
        model_name = files[-1]
    else:
        model_name = weights_location

    # Build paths and dates, used for storage path generation and normalizing dataset selection
    model_path = filepath+"/"+model_name
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

    if predict_subject == None:
        predict_subject = "7050"    # Set subject ID




    # Extract the name of the dataset used to train model
    print(model_path)
    trained_on_dataset = model_path.split('Cells_'+str(sequence_length)+'_', 1)[1]


    print("Prediction subject is: " + predict_subject)
    print("Using sequences of length: " + str(sequence_length))

    model, _ = MB.build_network(keep_state=True, batch_size=BATCH_SIZE,
                             sequence_length=sequence_length, num_features=NUM_FEATURES,
                             num_units=NUM_UNITS, num_layers=NUM_LAYERS,
                             num_activities=NUM_ACTIVITIES,
                             bidirectional=bidirectional)

    # Load model weights from trained model
    print(model_path)
    print("Loading model: " + model_name)
    model.load_weights(model_path)
    print("Success!")

    t0 = time.time()

    # Load sensor and timestamp data
    print("Loading sensor data...")
    sensor_measurements, timestamps = rd.build_prediction_data("/PATH/TO/DATA/"
                                                               "Prediction_Dataset/exported-csv/"
                                                               + predict_subject +"_timesync_time_B_T.csv",
                                                               sequence_length=sequence_length,
                                                               normalize_data=normalize,
                                                               normalization_value_set=trained_on_dataset,
                                                               use_abs_values=False)




    # Reshape sensor measurements to fit with batch size for the model
    sensor_measurements = sensor_measurements[0:((sensor_measurements.shape[0]//BATCH_SIZE)*BATCH_SIZE)]

    if sensor_measurements.shape[0] != timestamps.shape[0]:
        timestamps = timestamps[0:timestamps.shape[0]]

    # Format timestamps to conform with expected format
    for i in range(timestamps.shape[0]):
        timestamps[i] = datetime.strptime(timestamps[i], '%Y-%m-%d %H:%M:%S.%f')

    # Create two data "channels"
    pred_x1= np.zeros(shape=[sensor_measurements.shape[0], sequence_length, 3])
    pred_x2= np.zeros(shape=[sensor_measurements.shape[0], sequence_length, 3])

    # Fill data channels with our real sensor data from subject
    for example in range(sensor_measurements.shape[0]):
        for i in range(sequence_length):
            pred_x1[example, i, :] = sensor_measurements[example, i, 0:3]
            pred_x2[example, i, :] = sensor_measurements[example, i, 3:6]


    # Feed data into model and do predictions
    t1 = time.time()
    print("Doing Predictions...")
    raw_model_predictions = model.predict([pred_x1, pred_x2], batch_size=BATCH_SIZE, verbose=1)
    t2 = time.time()


    arg_maxes = []  # List for storing predictions
    num_uncertain = 0   # The number of uncertain predictions
    uncertainty_threshold = 0.4 # The threshold used to determine whether a prediction is uncertain or not

    # Go over all predictions
    for prediction in raw_model_predictions:
        # Check if model is uncertain
        if prediction[np.argmax(prediction)] < uncertainty_threshold:
            # If it is, add uncertainty marker (-1) and increment number of uncertain predictions
            arg_maxes.append(-1)
            num_uncertain += 1
        # If not uncertain
        else:
            # Add highest prediction class
            arg_maxes.append(np.argmax(prediction) + 1)

    # reshape predictions to fit with timestamp data
    arg_maxes = np.reshape(np.asarray(arg_maxes), newshape=(np.asarray(arg_maxes).shape[0], 1))

    # reshape timestamps to the same length as predicitons
    timestamps = timestamps[0:raw_model_predictions.shape[0]]
    timestamps = np.reshape(timestamps, newshape=(timestamps.shape[0], 1))

    # Combine timestaps and predictions to one array
    results = np.concatenate((timestamps, arg_maxes), axis=1)

    # Calculate model uncertainty
    uncertainty = (np.float(num_uncertain) / np.float(len(raw_model_predictions)))
    uncertainty = str(round(uncertainty, 3)).replace('.',',')

    # Convert array to pandas dataframe
    results = pd.DataFrame(results[:, 1:results.shape[1]], index=results[:,0])

    # Generate storage location path
    if normalize:
        store_loc = "/PATH/TO/Recurrent_ANN/Predictions/"+model_name+\
                    "_Stateful_"+predict_subject+\
                    "_normalized_predictions_uncertainty:"+uncertainty+".csv"
    else:
        store_loc = "/PATH/TO/Recurrent_ANN/Predictions/"+model_date+\
                    "_Stateful_"+predict_subject + \
                    "_not-normalized_predictions_uncertainty:"+uncertainty+".csv"

    print("Storing Predictions to: " +store_loc)
    # Store dataframe to storage path
    results.to_csv(store_loc,
                   index=True, header=False)

    # Print time statistics
    print("Loading time: %s" % (t1 - t0))
    print("Training time: %s" % (t2 - t1))
    print("Total prediction time: %s" % (t2 - t0))
    return store_loc

if "run" in term_in:
    if len(term_in) == 3:
        do_prediction(use_most_recent_weigths=False, predict_subject=term_in[2])
    else:
        do_prediction(use_most_recent_weigths=False)
