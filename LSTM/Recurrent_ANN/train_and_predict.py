import train_model_on_full_data as tm
import do_stateful_prediction as dsfp
import tensorflow as tf
import time
import sys


units_list = [32]


for units in units_list:
    for j in range(10):
        seq_length = 250
        n_layers = 1
        n_units =units
        bidirectional = True

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

        print("Sequence length: %i" % seq_length)
        print("Number of layers: %i" % n_layers)
        print("Number of units per layer: %i" % n_units)


        start = time.time()
        # Train model on training dataset
        weights_location= tm.train_a_model(sequence_length=seq_length, num_layers=n_layers, num_units=n_units, bidirectional=bidirectional,
                                           train_dataset="Downsampled-data/RESAMPLE/OOL")#  Train model

        weights_location = weights_location.split("Weights/",1)[1]

        # Use weights from trained model to to prediction using stateful LSTM
        prediction_csv = dsfp.do_prediction(num_layers=n_layers, num_units=n_units, bidirectional=bidirectional,
                                            use_most_recent_weights=False, predict_subject="7050", weights_location=weights_location) # Use model to predict

        """############################################## PATH MANIPULATION #########################################"""
        # TODO: Fix path
        if "guest" not in sys.path[0]:
            sys.path.insert(1, '/PATH/TO/HAR_PostProcessing')
        else:
            sys.path.insert(1, '/PATH/TO/HAR_PostProcessing')

        """############################################## PATH MANIPULATION #########################################"""

        import DailyOverview as DO
        DO.do_plot(prediction_csv, loop=False)  # Plot prediction results

        print("Total time: %s" % (time.time()-start))
