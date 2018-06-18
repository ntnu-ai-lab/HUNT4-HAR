import sys
term_in = sys.argv

def do_prediction(num_layers=1, num_units=32, bidirectional=True,
                  predict_subjects=None, weights_location=None,
                  stateful=True, sequence_length=250):
    """
    Builds a network, loads pretrained weights into the network and uses the network to predict
    activities over a list of predict_subjects. Finally calls for a plot of the predictions to be
    drawn.
    :param num_layers: number of layers in newtork to build
    :param num_units: number of units per layer in the network
    :param bidirectional: if we are building a bidirectional network or not
    :param predict_subjects: a list of paths to prediction subjects
    :param weights_location: path to the pretrained weights we want to load
    :param stateful: if we are building a stateful network
    :param sequence_length: the length of each sequence in the predictions
    :return: Nothing
    """

    import read_data as rd
    import time
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import MODEL_BUILDING as MB
    import glob


    # Set static variables
    BATCH_SIZE = 512    # size of batch for prediction
    NUM_ACTIVITIES = 19 # Number of activities in the final network layer
    NUM_FEATURES = 6    # Number of features in the input
    NUM_UNITS = num_units   # number of units in the network layers
    NUM_LAYERS = num_layers # number of layers in the network
    normalize = True    # If we are normalizing the data or not


    # Extract the name of the model from the path to the weights. Used for storing predictions
    model_name = weights_location.split("Weights/",1)[1]

    # Make sure that we are actually receiving a list of subjects
    if predict_subjects[0] == None:
        print("No prediction subjects provided... Exiting!")
        exit()


    # Build the network
    model, _ = MB.build_network(keep_state=stateful, batch_size=BATCH_SIZE,
                             sequence_length=sequence_length, num_features=NUM_FEATURES,
                             num_units=NUM_UNITS, num_layers=NUM_LAYERS,
                             num_activities=NUM_ACTIVITIES,
                             bidirectional=bidirectional)

    # Load model weights from trained model
    print("Loading model: " + weights_location)
    model.load_weights(weights_location)
    print("Success!")

    # Make a timestamp for folder to store all predictions in
    ts = time.gmtime()
    ts = time.strftime("%d-%m-%Y_%H:%M:%S", ts)

    # Go over all prediction subjects and do predictions for each subject
    for prediction_subject in predict_subjects:

        # Start recording time for statistics
        t0 = time.time()

        # Extract subject name from its path. Used for prediction naming
        subject_name = prediction_subject.split(os.sep)[-1][:-4]

        # Load sensor and timestamp data from subject
        print("Loading sensor data...")
        sensor_measurements, timestamps = rd.build_prediction_data(prediction_subject,
                                                                   sequence_length=sequence_length,
                                                                   normalize_data=normalize,
                                                                   normalization_value_set="RESAMPLE-OOL",
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
        if not os.path.isdir("/PATH/TO/Recurrent_ANN/Predictions/" + ts):
            print("Making new directory " + ts)
            os.mkdir("/PATH/TO/Recurrent_ANN/Predictions/" + ts)

        # Build storage path
        store_loc = "/PATH/TO/Recurrent_ANN/Predictions/"+ts+"/"+model_name+\
                    "_Stateful_"+subject_name+\
                    "_normalized_predictions_uncertainty:"+uncertainty+".csv"



        # Store dataframe to storage path
        results.to_csv(store_loc,
                       index=True, header=False)
        print("Storing Predictions to: " + store_loc)

        # Print time statistics
        print("Loading time: %s" % (t1 - t0))
        print("Training time: %s" % (t2 - t1))
        print("Total prediction time: %s" % (t2 - t0))

        """############################################## PATH MANIPULATION #########################################"""
        # TODO: Fix path
        if "guest" not in sys.path[0]:
            sys.path.insert(1, '/PATH/TO/HAR_PostProcessing')
        else:
            sys.path.insert(1, '/PATH/TO/HAR_PostProcessing')

        """############################################## PATH MANIPULATION #########################################"""

        # Do plot of predictions
        import DailyOverview as DO

        DO.do_plot(store_loc, loop=True)  # Plot prediction results



