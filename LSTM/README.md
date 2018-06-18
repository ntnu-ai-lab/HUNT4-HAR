@author: Haakom

## Requirements:
* Python 3.5
  * [Tensorflow](https://www.tensorflow.org/)
  * [Keras](https://github.com/keras-team/keras)
  * [Pandas](https://pypi.python.org/pypi/pandas/0.18.0/#downloads)
  * [Scipy](https://github.com/scipy/scipy)
  * [Matplotlib](https://matplotlib.org/index.html)
  * [Numpy](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
  * [seaborn](https://seaborn.pydata.org/index.html) 
  * [wand](https://docs.wand-py.org/en/0.4.4/)
  
For anaconda environments see:
- anaconda_tensorflow_python035_cpu.yml
- anaconda_tensorflow_python035_gpu.yml
  
###Project structure:

Recurrent_ANN:
   - pipeline_convert_train_and_predict.py. Converts raw (.cwa) data, located in a user specified folder  to .csv files 
   and stores it in another specified folder. Then trains a LSTM recurrent network on a specified training dataset and 
   stores the network as well as a confusion matrix over the training data. Finally uses the newly trained network to do
   predictions on the converted data, creates prediction .csv files and plots for each subject
   
   - pipeline_convert_and_predict.py. Converts raw (.cwa) data, located in a user specified folder to .csv files and 
   stores them in another user specified folder. Then builds a LSTM network and loads it with pretrained weights located 
   in a user specified folder. Uses the network to do prediction over the converted data, creates prediction .csv files 
   and plots for each subject
   
   - pipeline_train_and_predict.py. Trains a LSTM network on training data and stores the network as well as a confusion
   matrix over the training data. Assumes that the prediction data is already available and does prediction over the 
   user specified prediction data. Generates prediction .csv files and plots for each subject.
   
   - pipeline_only_predict.py. Builds a LSTM network and loads weights from a user specified location. Then uses the 
   network to do predictions over user specified prediction subjects. Generates prediction .csv files and plots for each
   subject.
   
    For all pipelines:
    1. Confusion matrix, if training, is found in /Recurrent_ANN/confusion_matrices/
    2. Prediction .csvs are found in /Recurrent_ANN/Predictions/(timestamp)/
    3. Prediction plots are found in /Recurrent_ANN/plots/(timestamp)/
    4. Trained model, if training, is found in /Recurrent_ANN/Trained_Models/
    5. Model weights, if training, are found in /Recurrent_ANN/Trained_Models_Weights/
   
   - predict.py. Builds a network, loads pretrained weights into the network and uses the network to predict
    activities over a list of predict_subjects. Finally calls for a plot of the predictions to be
    drawn.
   
   - train_and_predict.py. Trains a LSTM recurrent network on the dataset specified. 
    Uses this model to do prediction on a specified subject. Takes the predictions and generates
    a plot of the activities for that subject.
    
   - Explore_Twin_LSTM.py. Used for exploration of LSTM structures. This file does not store any network models, but 
    explores the effectiveness of a network architecture by training the architecture in a leave-one-subject-out manner
    on the training dataset. Usefull for determening which architectures show promise. It also gives some statistics on the
    accuracies for each subject as well as the average accuracy on the entire training data.
 
   - train.py. Trains a specified LSTM architecture, using the entire training dataset, with one 
    subject as a validation subject for performance monitoring. Creates a confusion matrix for the model on the training 
    dataset and finally saves the weights as well as the entire model to: /Trained_Model_Weights/ and /Trained_Models/ 
    respectively.
    NOTE: It is assumed that the training data resides in /DATA/
    
   - read_data.py. The file containing all data handling. This file reads training data, splits it into training and 
    validation data based on the specifed split. It also has a set of parameters which can be set:
        - normalize_data(default=True): Normalizes the dataset per channel(acceleration axis)
        - normalization_value_set(default=1): Specifies which dataset to use fetch normalization values from.
        - validation_subject: Specifies which subject(if any) should be used as validation subject.
        - generate_one_hot(default=True): Generates one-hot vectors from the label data
        - use_most_common_label(default=True): If we are using the most common label in the sequence.
        - use_abs_values(default=False): Will convert the dataset to absolute values, possibly useful if the sensors were
        mounted upside down.
        
   - do_stateful_prediction.py. Generates a network of stateful LSTMs and loads the specified pretrained weights into the 
    network. Then does prediction on the specified prediction subject. Stateful LSTMs remeber their prediction between 
    sequences, which makes them well suited for tasks where the goal is to predict over extremely long sequences, which 
    is exactly what we are doing here!
 
   - do_stateless_prediction.py. Generates a network of stateless LSTMs and loads the specified pretrained weights into the 
    network. Then does prediction on the specified prediction subject.
    
   - MODEL_BUILDING.py. Builds a LSTM based on the specified parameters and returns it, ready for
    training or for weights to be loaded into it.
    
   - read_and_convert_raw_data.py. Reads in a path to a folder containing subjects with .cwa files. Then converts all
    the data into .csv. Also generates a single .csv file with timestamps and both sensor data, ready for prediction.
    
   - make_confusion_matrix.py. Loads a model and generates a confusion matrix for the model, using the training dataset.
    
   - TRAINING_NORMALIZATION.py. Contains the different normalization values for each training dataset.
    
   - TRAINING_VARIABLES.py. Builds the path to the training dataset and returns it. Also contains a dictionary
    of relabeling rules for the training dataset.

Data_analysis: 
  >Contains a set of helper files to analyse datasets.
  - find_normalizing_stats.py. Takes a specified dataset and finds the normalization means and standard deviations for 
  each channel(acceleration axis).
  
  - plot_data_normdists.py. Plots the normal distributions of specified datasets.
  
  - plot_downsapled_data_stats.py. Plots the normal distributions of datasets after they have been downsampled. Can be
  can be used to compare downsampled datasets to each other and to 50Hz data. 
  
  - print_labels.py. Prints the max label of a subject if it is higher than a threshold.
  
  - read_dataset.py. Helper file for reading datasets.
  
  - fetch_dataframe_stats.py. Returns the mean and var of a pandas dataframe.
  
Downsampling:
  
  - downsample_dataset.py. Takes a specified dataset and downsamples it using all the different downsampling
  functions specified in downsampling_functions.py. Then saves each downsampled version to its own folder in
  /DATA/Downsampled-data/
  
  - downsapling_functions.py. Contains differnet downsampling strategies for downsampling data. They are mostly geared
  towards downsampling from 100Hz to 50Hz, so for other frequencies, significant changes may be required.
  
  - downsampling_data_distributins.py. Plots the normal distributions of the downsampled datasets and the originals. 
  Useful for comparing downsampling strategies and figuring out which strategy produces data that approximates the goal 
  distribution the best.
  
  - read_and_plot_data.py A helper for the downsampling functions. Used for handling pandas dataframes.
  
  - rename_IL_subjects.py. Renames the In Lab subject files, to conform with the training data structure.
  
###Training Data Structure:
The training data is expected to conform to a simple naming scheme. Each subject should have its own folder, where all
the sensor data and labels are stored. The sensor data should be split into two .csv files: Back and Thigh. Finally 
there should also be a labels .csv file. The three files should follow the following naming scheme:
- DATA/(Path_To_Subjects)/(subjectID)/(subjectID)_Axivity_BACK_Back.csv
- DATA/(Path_To_Subjects)/(subjectID)/(subjectID)_Axivity_THIGH_Right.csv
- DATA/(Path_To_Subjects)/(subjectID)/(subjectID)_GoPro_LAB_All.csv (THESE ARE THE LABELS!)

###Manual paths:
Since this code was developed on a laptop, but run on a separate machine, some files contain manual paths that will have
to be changed to achieve the intended behaviour at runtime. There are also a limited amount of path manipulations in the
files. The affected files when training networks and doing predictions are:

####Path manipulations(You HAVE to change these for code to run):
- read_and_convert_raw_data.py
- predict.py
- train_and_predict.py

#####Manual path definitions(Code might run, but models, predictions and plots will be stored in strange places):
- pipeline_convert_and_predict.py
- pipeline_convert_train_and_predict.py
- pipeline_only_predict.py
- do_stateful_prediction.py
- do_stateless_prediction.py
- make_confusion_matrix.py
- predict.py
- There might also be more, but a quick inspection revealed these files.
