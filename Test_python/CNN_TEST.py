#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation, Flatten
import warnings
warnings.filterwarnings('ignore')

# In[2]:


def convert_to_3d(data,n_time=100):
    a1 = np.int(np.floor(data.shape[0]/n_time)*n_time)
    data = data[:a1,:]
    n_feat = data.shape[1]
    X1 = data.reshape((round(data.shape[0]/n_time),n_time,n_feat))
    X11 = X1[:,:,1:7]
    Y11 = X1[:,0,11]
    return X11, Y11

def model_struct():
    n_timesteps = 100
    n_features = 6
    n_outputs = 4
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    return model

def split_sequence_window_past_only(data1, n_steps=100):
    data = np.array(data1)
    #print(data1)
    if data.shape[0]< n_steps+1 :
        return [[0],[0]]
    else:
        X, y, ind = list(), list(), list()
        #sequence_y = data[:,11]
        for i in range(data.shape[0]):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > data.shape[0]-1:
                break
            # gather input and output parts of the pattern
            seq_x = data[i:end_ix].reshape((1,(end_ix-i),data.shape[1]))#, sequence_y[end_ix]
            X.append(seq_x)
            #y.append(seq_y)
            ind.append(end_ix)
        return np.squeeze(X, axis = 1), np.array(ind)

def CNN_model_load(relay, config):
    model = model_struct()
    dir_sample = '../Models'
    os.chdir(dir_sample)
    model.load_weights('CNN_weights_' + relay +'.h5')
    #os.chdir(dir_sav)
    fil = config + relay
    max_min = np.load('maxmin_svm_' + fil + '.npy')
    return model, max_min

def loading_matfile_test(dir_path, relay, config):
    dir_sample = './Data'
    os.chdir(dir_sample)
    mat_file = config+relay 
    data2 = sio.loadmat(relay + '.mat')[mat_file]
    test = data2[round(data2.shape[0]/2):,:]
    return test

def loading_Sample_file(relay, config):
    mat_file = 'sample_' + config + relay
    dir_sample = '../Data'
    os.chdir(dir_sample)
    data2 = sio.loadmat(mat_file)[mat_file]
    return data2

def data_preparation(data2, max1, min1):
    data2 = (data2 - min1) / (max1 - min1)
    test = data2
    x_test = test#[:,1:7]
    #y_test = test[:,10]
    return x_test#, y_test

def CNN_TEST1(data, relay):
    dir_sav = './Models'
    mdic = {"data": data}
    config = 'C1'
    [model, max_min_tr] = CNN_model_load(relay, config)
    test = data_preparation(data,  max_min_tr[0,:],  max_min_tr[1,:])
    
    [t1, ind1] = split_sequence_window_past_only(test)
 
    if np.shape(t1)[0]==1:
        print('WARNING: insufficient data to process through the model')
        return [float("NaN"), float("NaN"), float("NaN"), float("NaN")]
    else:
        X_test = t1
        #print(X_test)
        Y_pred = model.predict(X_test, verbose=2)
        #print(Y_pred[-1])
        return Y_pred[-1]

'''

def main():
    dir_sav = './Models'
    relay = 'RTL3'

    config = 'C1'
    [model, max_min_tr] = CNN_model_load(relay, config)
    model.summary()

    # Data loading

    data = loading_Sample_file(relay, config)
    ############## ONLY NEEDED IF LOADING ENTIRE MATFILE #########################
    #directory = '/content/drive/MyDrive/CONF _EXPERIMETN'
    #data = loading_matfile_test(directory, relay, config) # if loading happens through mat file
    ##############################################################################
    test,y = data_preparation(data,  max_min_tr[0,:],  max_min_tr[1,:])

    [t1, ind1] = split_sequence_window_past_only(test)
    if t1.shape[0]==1:
        print('WARNING: insufficient data to process through the model')
    else:
        X_test = t1
        Y_pred = model.predict(X_test, verbose=2)
        print(Y_pred)

if __name__ == '__main__':
    main()


# In[ ]:
'''



