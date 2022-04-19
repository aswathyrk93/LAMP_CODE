#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
from sklearn.svm import SVC
import scipy.io as sio
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def data_preparation(data2, max1, min1):
    data2 = (data2 - min1) / (max1 - min1)
    test = data2
    x_test = test#[:,1:7]
    #y_test = test#[:,10]
    return x_test#, y_test

def load_svm_model(relay):
    '''
    LOADING MODEL PARAMETERS:
    - loading the svm model 
    - loading max and min values of the training data
    for normalization.
    '''
    config = 'C3'
    dir_model = '../Models'
    os.chdir(dir_model)
    fil = config + relay
    filename = fil +'_svmmodel.sav' 
    loaded_model = pickle.load(open(filename, 'rb'))

    max_min = np.load('maxmin_svm_' + fil + '.npy')

    return loaded_model, max_min

def svm_test(x_test_orig, svm_model):
    '''
    Testing the data using the loaded svm model
    '''
    svm_predictions = svm_model.predict(x_test_orig)
    scores_oc = svm_model.decision_function(x_test_orig)
    prob = np.exp(scores_oc)/np.sum(np.exp(scores_oc),axis=1, keepdims=True) # softmax after the voting
    return prob

def loading_matfile_test(dir_path, relay, config):
    dir_sample = '../Data'
    os.chdir(dir_sample)
    mat_file = config+relay 
    data2 = sio.loadmat(relay+'.mat')[mat_file]
    test = data2[round(data2.shape[0]/2):,:]
    return data2

def loading_Sample_file(relay, config):
    mat_file = 'sample_' + config + relay
    dir_sample = '../Data'
    os.chdir(dir_sample)
    data2 = sio.loadmat(mat_file)[mat_file]
    return data2

def svm_test_main(data, relay):
    
    [svm_model, max_min_tr] = load_svm_model(relay)
# using pymodbus
    test = data_preparation(data,  max_min_tr[0,:],  max_min_tr[1,:])
#### CHANGED HERE ####    
    prob = svm_test(np.array(test).reshape(-1,1).T, svm_model)
    return prob[0] # probability has the probability scores

'''
def main():
    relay = 'RTL3'
    config = 'C1' # IMP:::::::::: need to change based on the test data config
    dir_sav = './Models'
    [svm_model, max_min_tr] = load_svm_model(relay)
    #Data loading

    config = 'C1' # IMP:::::::::: need to change based on the test data config

    ############## ONLY NEEDED IF LOADING ENTIRE MATFILE #########################
    #directory = '/content/drive/MyDrive/CONF _EXPERIMETN'
    #data = loading_matfile_test(directory, relay, config) # if loading happens through mat file
    ##############################################################################
    data = loading_Sample_file(relay, config) # if getting the sample data provided
    #### data = array ### IMP::: change to this if getting data using pymodbus
    test,y = data_preparation(data,  max_min_tr[0,:],  max_min_tr[1,:])
    prob = svm_test(test, svm_model)
    print(prob) # probability has the probability scores

if __name__ == '__main__':
    main()
'''

