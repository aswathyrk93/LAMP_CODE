# LAMP_CODE

There are 4 folders in this reposiory:

DATA : 
Contains the sample data for each configuration and also the train data that will be used for re-initializing the GP model.

MODELS: 
> Contains all the models that were trained offline. 
> The names of the models are self-explanatory of the configuration and the algorithm that was used for its training.
> Includes the maximum and the minimum values that is used for normalization in case of GP+OCSVM and SVM classifiers.
> These models are loaded in test set up

Test_python:
> The folder contains the 9 test codes. Each code is for a particular algorithm:
   - CNN_TEST.py : tests the CNN block for classifying the configurations.
   - Fault_Classifier_testC1.py : Tests the SVM model trained using data of Configuration 1. Similarly there are 3 more files for each of the configurations. 
   - GP_OCSVM_TEST_C1.py: Tests the GP_OCSVM model trained using data of configuration 1. Similary there are 3 more files for each of the configurations. 
   - microgrid_reading.py: Reads all the data using modbus and passes them to the above test files for performing the fina computation of probabilities.
> The main code here is the "microgrid_reading.py" which reads the data from client and processes them through all the models in "MODELS" folder and computes the final output to display them.

Train:
> contains ipynb files for training all the algorithms: CNN (for topology estimation), GP+OCSVM (fault detection), SVM (fault classification)
> The train data needs to be accessed from outside. These files were trained using a huge file called 'RTL3.mat" which had data corresponding to all the configuration. These data is used for training all the algorithms.
> The folder contains 6 files:
   - CNN1D_train_Save: trains the CNN model and saves the model in ".h5" format.
   - Fault_classifier_trainC1:  trains the fault classifier using SVM for configuration 1 and saves the corresponding model. Similarly we have fault classifier training models for the other three configurations
   - GP_OCSVM_train: THIS FILE IS IMPORTANT. This mode needs to RETRAINED most frequently. EVERY other day. This is unsupervised block and it trains on the input data and saves all three GP models and OCSVM model



            
