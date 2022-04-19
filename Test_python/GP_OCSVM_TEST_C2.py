#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import warnings
import torch
import gpytorch
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import svm
import pickle
import os

warnings.filterwarnings('ignore')

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_prior = gpytorch.priors.GammaPrior(2,2)))#gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_prior = gpytorch.priors.GammaPrior(0.5,2)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def GP_model(train_x, train_y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    return model, likelihood

def loading_Sample_file(relay, config):
    '''
    loads small sample test data
    '''
    mat_file = 'sample_' + config + relay
    dir_sample = '../Data'
    os.chdir(dir_sample)
    data2 = sio.loadmat(mat_file)[mat_file]
    return data2
    
def loading_matfile_test(relay, config):
    '''
    loads the entire test matfile
    '''
    dir_sample = '../Data'
    os.chdir(dir_sample)
    mat_file = config+relay 
    data2 = sio.loadmat(relay + '.mat')[mat_file]
    test = data2[round(data2.shape[0]/2):,:]
    j = 100000
    i = 0
    test1 = test[i:j,:]
    return test1

def load_GPOCSVM_models(relay, config):
    mat_file = config + relay 
    
    ## loading train data for initializing GP
    os.chdir('../Data')    
    [train_x, train_y] = np.load('train_GP_' + mat_file + '.npy')
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    ## loading all the GP models
    os.chdir('../Models')
    #GP1
    GP1, likelihood1 = GP_model(train_x, train_y[:,0])
    GP1_state_dict = torch.load('GP1_' + mat_file + '.pth')
    GP1.load_state_dict(GP1_state_dict)
    #GP2
    GP2, likelihood2 = GP_model(train_x, train_y[:,1])
    GP2_state_dict = torch.load('GP2_' + mat_file + '.pth')
    GP2.load_state_dict(GP2_state_dict)
    #GP3
    GP3, likelihood3 = GP_model(train_x, train_y[:,2])
    GP3_state_dict = torch.load('GP3_' + mat_file + '.pth')
    GP3.load_state_dict(GP3_state_dict)

    # Loading OCSVM model
    file_name = 'OCSVM_'+ mat_file + '.sav'
    OCSVM = pickle.load(open(file_name, 'rb'))
    [max1, min1] = np.load('maxmin_GPOCSVM_' + mat_file + '.npy')
    
    return GP1.double(), GP2.double(), GP3.double(), likelihood1, likelihood2, likelihood3, OCSVM, max1, min1

def data_preparation(test, max1, min1):
    test = (test - min1) / (max1 - min1)
    test_x = test[:,0:3]
    test_y = test[:,3:6]
    #test_lab = 2*(test[:,10]==0)-1
    return test_x, test_y


def Gp_test(test_x, model, likelihood):

    model.eval()
    likelihood.eval()
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #observed_pred = model(test_x)
        observed_pred = likelihood(model(test_x))
        f_var = observed_pred.variance

    return observed_pred, f_var

def OCSVM_test(clf, test):
    pred = clf.predict(test)
    scores_oc = clf.decision_function(test)
    p = 1/(1+np.exp(scores_oc))
    return pred, scores_oc, p

def GP_plus_OC_TEST(test_x, test_y, model1, likelihood1, model2, likelihood2, model3, likelihood3, clf):
    
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    ##################### TEST #######################
    #test GP1
    [observed_pred1, f_var1] = Gp_test(test_x.double(), model1, likelihood1)
    [error1, f_var1] = plot_test_GP(observed_pred1, f_var1, test_y[:,0].double())

    
    # test GP2
    [observed_pred2, f_var2] = Gp_test(test_x.double(), model2, likelihood2)
    [error2, f_var2] = plot_test_GP(observed_pred2, f_var2, test_y[:,1].double())

    
    # test GP3
    [observed_pred3, f_var3] = Gp_test(test_x.double(), model3, likelihood3)
    [error3, f_var3] = plot_test_GP(observed_pred3, f_var3, test_y[:,2].double())

    std_f1 = np.sqrt(f_var1)
    std_f2 = np.sqrt(f_var2)
    std_f3 = np.sqrt(f_var3)
    
    test = np.vstack((error1/std_f1,error2/std_f2,error3/std_f3)).T

    #print(train.shape, test.shape)
    
    [pred, scores_oc, p] = OCSVM_test(clf, test)

    return pred, scores_oc, p, test

def plot_test_GP(observed_pred, f_var, test_y):

    n_sample0 = 0
    n_sample1 = test_y.shape[0]
  
    with torch.no_grad():
    # Initialize plot
    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
        mae = mean_absolute_error(test_y.numpy(), observed_pred.mean.numpy())
        mse = mean_squared_error(test_y.numpy(), observed_pred.mean.numpy())
        mse_var = mean_squared_error(test_y.numpy(), observed_pred.mean.numpy())/np.var(test_y.numpy(), axis=0)
        lower = observed_pred.mean-2*f_var
        upper = observed_pred.mean+2*f_var
        error1 = observed_pred.mean.numpy()[n_sample0:n_sample1] - test_y.numpy()[n_sample0:n_sample1]
    return error1, f_var

def GP_test_main(test, relay):
    config = 'C2'   # for training
    [GP1, GP2, GP3, lh1, lh2, lh3, OCSVM, max1, min1] = load_GPOCSVM_models(relay, config)
    
    [test_x, test_y] = data_preparation(np.array(test).reshape(-1,1).T, max1, min1)
    # testing on the data
    [pred, scores_oc, p, test] = GP_plus_OC_TEST(test_x, test_y, GP1, lh1, GP2, lh2, GP3, lh3, OCSVM)
    return pred # +1 normal, -1 abnormal
'''
def main():
    config = 'C1'
    relay = 'RTL3'
    [GP1, GP2, GP3, lh1, lh2, lh3, OCSVM, max1, min1] = load_GPOCSVM_models(relay, config)
    test = loading_Sample_file(relay, 'C1')
    ############## ONLY NEEDED IF LOADING ENTIRE MATFILE #########################
    #data = loading_matfile_test(relay, config) # if loading happens through mat file
    ##############################################################################
    [test_x, test_y, test_lab] = data_preparation(test, max1, min1)
    # testing on the data
    [pred, scores_oc, p, test] = GP_plus_OC_TEST(test_x, test_y, GP1, lh1, GP2, lh2, GP3, lh3, OCSVM)
    print(pred,test_lab) # +1 normal, -1 abnormal
    
if __name__ == '__main__':
    main()
'''

