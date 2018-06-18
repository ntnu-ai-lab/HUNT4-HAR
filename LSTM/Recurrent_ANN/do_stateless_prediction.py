def do_prediction():
    from keras.models import load_model
    import read_data as rd
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime


    # Define model initial path
    filepath='/PATH/TO/Recurrent_ANN/Trained_Models'

    # Sort models in path
    files = sorted([f for f in os.listdir(filepath)])

    # Select most recent model
    model_name = files[-1]

    # Attach model name to model path
    model_path = filepath+"/"+model_name

    # Get model training data for naming purposes later
    model_date = model_path.split("Models/",1)[1][0:19]

    # We will normalize our data
    normalize = True

    # Set sequence length based on sequence length used in the trained model
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

    # Set subject ID
    predict_subject = "7050"

    # Set batch size for prediction
    BATCH_SIZE = 512

    # Load pretrained model from path
    print("Loading model: " +model_name)
    model = load_model(model_path)
    print("Success!")

    # Find dataset the model was trained on, for normalization
    print(model_path)
    trained_on_dataset = model_path.split('Cells_'+str(sequence_length)+'_', 1)[1]


    print("Prediction subject is: " + predict_subject)
    print("Using sequences of length: " + str(sequence_length))


    # Load sensor data
    print("Loading sensor data...")
    sensor_measurements, timestamps = rd.build_prediction_data("/PATH/TO/DATA/"
                                                               "Prediction_Dataset/exported-csv/"
                                                               + predict_subject +"_timesync_time_B_T.csv",
                                                               sequence_length=sequence_length,
                                                               normalize_data=normalize,
                                                               normalization_value_set=trained_on_dataset,
                                                               use_abs_values=False)



    # Reshape datastamps shape to fit with sensor shape if we are using stateful model
    if "stateful" in model_name:
        sensor_measurements = sensor_measurements[0:((sensor_measurements.shape[0]//BATCH_SIZE)*BATCH_SIZE)]

    # Make number of timestamps fit with number of sensor measurements
    if sensor_measurements.shape[0] != timestamps.shape[0]:
        timestamps = timestamps[0:timestamps.shape[0]]

    # Reformat timestamps to fit with desired format
    for i in range(timestamps.shape[0]):
        timestamps[i] = datetime.strptime(timestamps[i], '%Y-%m-%d %H:%M:%S.%f')

    # Split into two "channels"
    pred_x1= np.zeros(shape=[sensor_measurements.shape[0], sequence_length, 3])
    pred_x2= np.zeros(shape=[sensor_measurements.shape[0], sequence_length, 3])

    # Fill the two channels with sensor data
    for example in range(sensor_measurements.shape[0]):
        for i in range(sequence_length):
            pred_x1[example, i, :] = sensor_measurements[example, i, 0:3]
            pred_x2[example, i, :] = sensor_measurements[example, i, 3:6]


    # Do prediction
    print("Doing Predictions...")
    predictions =  model.predict([pred_x1, pred_x2], batch_size=BATCH_SIZE, verbose=1)

    # print som stats for Predictions
    arg_maxes = []  # List for storing predictions
    num_uncertain = 0# The number of uncertain predictions
    uncertainty_threshold = 0.4  # The threshold used to determine whether a prediction is uncertain or not

    # Go over all predictions and calculate amount of uncertain predictions
    for prediction in predictions:
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
    uncertainty = (np.float(num_uncertain)/np.float(len(predictions)))

    # reshape timestamps to the same length as predicitons
    timestamps = timestamps[0:predictions.shape[0]]
    timestamps = np.reshape(timestamps, newshape=(timestamps.shape[0], 1))

    # Combine timestaps and predictions to one array
    results = np.concatenate((timestamps, arg_maxes), axis=1)

    # Convert array to pandas dataframe
    results = pd.DataFrame(results[:, 1:results.shape[1]], index=results[:,0])

    # Generate storage location path
    if normalize:
        store_loc = "/PATH/TO/Recurrent_ANN/Predictions/"+model_name+\
                    "_"+predict_subject+\
                    "_normalized_predictions_uncertainty:"+str(uncertainty)+".csv"
    else:
        store_loc = "/PATH/TO/Recurrent_ANN/Predictions/"+model_date+\
                    "_"+predict_subject + \
                    "_not-normalized_predictions_uncertainty:"+str(uncertainty)+".csv"

    print("Storing Predictions to: " +store_loc)
    # Store results
    results.to_csv(store_loc,
                   index=True, header=False)
