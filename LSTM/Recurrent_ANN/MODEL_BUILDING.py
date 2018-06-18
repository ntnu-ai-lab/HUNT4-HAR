from keras.layers import Input, Dropout, Activation, Dense, Concatenate,Add, Bidirectional, BatchNormalization, regularizers
from keras.models import Model
from keras.layers import CuDNNLSTM, LSTM


def residual_addition(in_1, in_2, nn1, nn2, activation, batchNorm):
    """
    Does residual addition, batch normalization and activation, based on the inputs

    :param in_1: residual 1
    :param in_2: residual 2
    :param nn1: net_out 1
    :param nn2: net_out 2
    :param activation: activation to use
    :param batchNorm: if we are using batch normalization
    :return: residual added nn1, nn2
    """
    if batchNorm:
        # Add the two selected layers together batch norm and activate
        nn1 = Activation(activation)(BatchNormalization()(Add()([nn1, in_1])))
        nn2 = Activation(activation)(BatchNormalization()(Add()([nn2, in_2])))
    else:
        # Add the two selected layers together and activate
        nn1 = Activation(activation)(Add()([nn1, in_1]))
        nn2 = Activation(activation)(Add()([nn2, in_2]))
    return nn1, nn2


def bidirectional_LSTM_layer(num_units, in_1, in_2, return_seq, stateful, residual, batchNorm, GPU=True):
    """
    Creates a set of bidirectional LSTM layer channels

    :param num_units: number of units in this layer
    :param in_1: input to channel 1
    :param in_2: input to channel 2
    :param return_seq: if we are returning sequences
    :param stateful: if we are training a stateful LSTM
    :param residual: if we are doing residual addition
    :param batchNorm: if we are doing batch normalization
    :param GPU: if we are running on GPU
    :return: channel1_out, channel2_out, name of layer
    """
    if GPU:
        # Create two bidirectional LSTM layers to run on GPU
        nn1 = Bidirectional(CuDNNLSTM(units=num_units, return_sequences=return_seq,
                                      stateful=stateful), merge_mode='sum')(in_1)

        nn2 = Bidirectional(CuDNNLSTM(units=num_units, return_sequences=return_seq,
                                      stateful=stateful), merge_mode='sum')(in_2)
    else:
        # Create two bidirectional LSTM layers to run on CPU
        nn1 = Bidirectional(LSTM(units=num_units, return_sequences=return_seq,
                                 stateful=stateful), merge_mode='sum')(in_1)
        nn2 = Bidirectional(LSTM(units=num_units, return_sequences=return_seq,
                                 stateful=stateful), merge_mode='sum')(in_2)
    # Add to model name
    name_builder = "=" + str(num_units)

    # Maybe do a residual addition
    if residual:
        nn1, nn2 = residual_addition(in_1, in_2, nn1, nn2, "tanh", batchNorm=batchNorm)

    return nn1, nn2, name_builder

def regular_LSTM_layer(num_units, in_1, in_2, return_seq, stateful, residual, batchNorm, GPU=True):
    """
    Creates a set of forward LSTM layer channels

    :param num_units: number of units in this layer
    :param in_1: input to channel 1
    :param in_2: input to channel 2
    :param return_seq: if we are returning sequences
    :param stateful: if we are training a stateful LSTM
    :param residual: if we are doing residual addition
    :param batchNorm: if we are doing batch normalization
    :param GPU: if we are running on GPU
    :return: channel1_out, channel2_out, name of layer
    """
    if GPU:
        # Create two regular LSTM layers to run on GPU
        nn1 = CuDNNLSTM(units=num_units, return_sequences=return_seq,
                        stateful=stateful)(in_1)

        nn2 = CuDNNLSTM(units=num_units, return_sequences=return_seq,
                        stateful=stateful)(in_2)
    else:
        # Create two regular LSTM layers to run on CPU
        nn1 = LSTM(units=num_units, return_sequences=return_seq,
                   stateful=stateful)(in_1)
        nn2 = LSTM(units=num_units, return_sequences=return_seq,
                   stateful=stateful)(in_2)

    # Add to model name
    name_builder = "=" + str(num_units)

    # Maybe do a residual addition
    if residual:
        nn1, nn2 = residual_addition(in_1, in_2, nn1, nn2, "tanh", batchNorm=batchNorm)

    return nn1, nn2, name_builder

def final_LSTM_layer(num_units, input, stateful, bidirectional, GPU=True):
    """
    Creates a final LSTM layer for the network

    :param num_units: number of units in the layer
    :param input: input to layer
    :param stateful: if we are doing a stateful LSTm
    :param bidirectional: if we are doing bidirectianal LSTMs
    :param GPU: if we are running on GPU
    :return: layer_out, name of layer
    """

    if bidirectional:
        # Build a final bidirectional LSTM layer
        if GPU:
            out = Bidirectional(CuDNNLSTM(units=num_units,
                                          return_sequences=False,
                                          stateful=stateful), merge_mode='sum')(input)
        else:
            out = Bidirectional(LSTM(units=num_units,
                                     return_sequences=False,
                                     stateful=stateful), merge_mode='sum')(input)
    else:
        # Build a final regular LSTM layer
        if GPU:
            out = CuDNNLSTM(units=num_units,
                            return_sequences=False,
                            stateful=stateful)(input)
        else:
            out = LSTM(units=num_units,
                       return_sequences=False,
                       stateful=stateful)(input)

    # Add to model name
    name_builder="-"+str(num_units)

    return out, name_builder


def build_network(keep_state, batch_size, sequence_length, num_features,
                  num_layers, num_units, num_activities,
                  bidirectional=True, batchNorm = True, GPU = True):
    """
    Builds a recurrent LSTM network. This network is a two-channel network that takes the back and
    thigh sensor data as input to separate channels. These two channels can be either one or several
    layers deep, before they are concatenated and fed into a final LSTM layer for prediction.

    The network can also be tuned with whether we want shorcut connections between layers or not and if
    we want the network to run on GPU or CPU.

    :param keep_state: If we are remembering network state accross batches
    :param batch_size: size of input batch
    :param sequence_length: length of input sequences
    :param num_features: number of features in input
    :param num_layers: number of two-channel layers we want in the network
    :param num_units: number of units in the two-channel part of the network
    :param num_activities: number of activites in the output for the network
    :param bidirectional: if we are doing a bidirectional network
    :param batchNorm: if we are using batch normalization
    :return: built model, name of model
    """
    model_name = ""

    if keep_state:
        print("Building stateful network...")
        nn1_in = Input(batch_shape=[batch_size, sequence_length, num_features // 2])
        nn2_in = Input(batch_shape=[batch_size, sequence_length, num_features // 2])
    else:
        print("Building stateless network...")
        nn1_in = Input(shape=[sequence_length, num_features // 2])
        nn2_in = Input(shape=[sequence_length, num_features // 2])

    # Build model on CPU or GPU based on system path
    for i in range(num_layers):

        if bidirectional:
            if i == 0:
                model_name += "Bi"
                # First bidirectional layer
                nn1, nn2, name_builder = bidirectional_LSTM_layer(num_units=num_units,
                                                    in_1=nn1_in,
                                                    in_2=nn2_in,
                                                    return_seq=True,
                                                    stateful=keep_state,
                                                    residual=False,
                                                    batchNorm=batchNorm,
                                                    GPU=GPU)
                res_nn1 = nn1
                res_nn2 = nn2
            else:
                # nth bidirectional layer, this layer includes a shortcut connection similar to resnet
                nn1, nn2, name_builder = bidirectional_LSTM_layer(num_units=num_units,
                                                    in_1=nn1,
                                                    in_2=nn2,
                                                    return_seq=True,
                                                    stateful=keep_state,
                                                    residual=True,
                                                    batchNorm=batchNorm,
                                                    GPU=GPU)
            model_name+=name_builder
        else:
            if i == 0:
                # First regular layer
                nn1, nn2, name_builder = regular_LSTM_layer(num_units=num_units,
                                              in_1=nn1_in,
                                              in_2=nn2_in,
                                              return_seq=True,
                                              stateful=keep_state,
                                              residual=False,
                                              batchNorm=batchNorm,
                                              GPU=GPU)
                res_nn1 = nn1
                res_nn2 = nn2
            else:
                # nth regular layer, this layer includes a shortcut connection similar to resnet
                nn1, nn2, name_builder = regular_LSTM_layer(num_units=num_units,
                                              in_1=nn1,
                                              in_2=nn2,
                                              return_seq=True,
                                              stateful=keep_state,
                                              residual=True,
                                              batchNorm=batchNorm,
                                              GPU=GPU)
            model_name += name_builder
    # If we have more than one layer, also include a shortcut connection to the first layer
    if num_layers > 2:
        nn1, nn2 = residual_addition(res_nn1, res_nn2, nn1, nn2, "tanh", batchNorm=batchNorm)
    # Concatenate the two channels
    nn = Concatenate(axis=2)([nn1, nn2])
    # Do dropout
    nn = Dropout(0.9)(nn)
    # Final network layer
    nn, name_builder = final_LSTM_layer(num_units=num_activities,
                                        input=nn,
                                        stateful=keep_state,
                                        bidirectional=bidirectional,
                                        GPU=GPU)

    # Since tensorflow's CuDNNLSTM does not allow us to select our activation function, we instead add a final
    # activation layer at the end of our network. If this is changed in the future, the user is strongly encouraged
    # to remove this layer and select softmax as the activation in the final network layer above instead.
    nn = Activation(activation="softmax")(nn)

    # Finsish model name
    model_name += name_builder

    # Compile model and prepare graph
    model = Model(inputs=[nn1_in, nn2_in], outputs=nn)
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    # Print model summary in terminal
    model.summary()

    return model, model_name
