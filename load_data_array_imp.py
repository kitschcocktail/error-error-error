# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:59:59 2015

@author: heiligenstein
"""

#!/usr/bin/env python
import os
import sys

import cPickle
import numpy
import nrg

from itertools import tee, izip



# load Commerzbank Arena dataset
execfile("data_Commerzbank.py")

def load_dataset(k_root, H, addition=False):
    """ Loads and formats data from Commerzbank Arena dataset.
    
    k_root: type int. Size of the input related to the root temperature.
    H: type int. Size of the future window for root temperature. The delay
        of both signals is set to 32 by default.
    addition: type boolean. If False, the window of the root temperature is
        delayed 32 samples. If True, the root temperature window is also
        delayed but air temperature window is filled with the next 32 values
        matching the root window.
    impute: type boolean. If 'z', impute using ran_z method (scattered 0s). If
        'c', impute using ran_c method (consecutive 0s).
    pc: type int. Percentage to impute.
    len_outage: type int. Length of outage in hours.
    
    Returns:
        103-length arrays from raw root and air data formatted for input to
        forecasting model.
    """
    if type(k_root) != int:
        sys.exit('k_root must be an integer.')
    if k_root <= 0 :
        sys.exit('k_root must be an integer greater than zero')
    if type(H) != int:
        sys.exit('H must be an integer.')
    if H <= 0 :
        sys.exit('H must be an integer greater than zero')
        
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)
    
    def build_set(root_data, air_data, k_root, H,\
            addition, norm=False): 
        
        root_data = root_data[-(len(root_data) - 32):]
        
        input1 = []
        ref_temp = []
        for each in window(root_data, k_root):
            input1.append(numpy.array(each))
            ref_temp.append(numpy.array([each[-1]]))
        input1 = numpy.array(input1)
        ref_temp = numpy.array(ref_temp)
        #input1 = numpy.array([numpy.array(i - i[-1]) for i in input1])
        input1 = numpy.array([numpy.delete(i, -1) for i in input1])
        
        input2 = []
        for each in window(air_data, k_root):
            input2.append(numpy.array(each))
        input2=numpy.array(input2)
        input2 = numpy.array([numpy.array(i - i[-1]) for i in input2])
        input2 = numpy.array([numpy.delete(i, -1) for i in input2])
        
        y_target = []
        for each in window(root_data, H):
            y_target.append(numpy.array(each))
        y_target = numpy.array(y_target)
        #y_target = numpy.array([numpy.array(i - i[0]) for i in y_target])
        y_target = numpy.array([numpy.delete(i, 0) for i in y_target])
        
        input1=input1[:-H]
        ref_temp = ref_temp[:-H]
        input2=input2[:-(H + 32)]
        y_target=y_target[-(len(y_target)-k_root):]
        
        if addition:
            input2 = []
            for each in window(air_data, k_root + 32):
                input2.append(numpy.array(each))
            input2=numpy.array(input2)
            #input2 = numpy.array([numpy.array(i - i[-1]) for i in input2])
            #input2 = numpy.array([numpy.delete(i, -1) for i in input2])
            
            input2=input2[:-H]
        
        # concatenate root and external temperature
        input = numpy.concatenate((input1, ref_temp, input2),axis=1)
        
        
        if norm:
            mean_input = input.mean(axis=0)
            std_input  = input.std(axis=0)
            mean_target = y_target.mean(axis=0)
            std_target  = y_target.std(axis=0)

            input    = (input - mean_input)/std_input
            y_target = (y_target - mean_target)/std_target
            return [input, y_target, mean_input, std_input, mean_target, std_target] 
        else:
            return [input, y_target]
           
    # use only one month of data for our purposes

    root_train_s = root[0]
    air_train_s = air[0]

        
    #root_test = root[13]     ; air_test = air[13]
    
    
    input_train_s, y_train_s = build_set(root_train_s, air_train_s, k_root,
                                   H, addition, norm=False)
    #input_test, y_test      = build_set(root_test, air_test, k_root,
    #                              H, addition, norm=False)
    
    rval = [(input_train_s, y_train_s)]
            #(input_test, y_test)]
    return rval

def load_dataset_imp(k_root, H, imp_pc, c, addition=False):
    """ Loads and formats data from Commerzbank Arena dataset.
    
    k_root: type int. Size of the input related to the root temperature.
    H: type int. Size of the future window for root temperature. The delay
        of both signals is set to 32 by default.
    addition: type boolean. If False, the window of the root temperature is
        delayed 32 samples. If True, the root temperature window is also
        delayed but air temperature window is filled with the next 32 values
        matching the root window.
    impute: type boolean. If 'z', impute using ran_z method (scattered 0s). If
        'c', impute using ran_c method (consecutive 0s).
    pc: type int. Percentage to impute.
    len_outage: type int. Length of outage in hours.
    
    Returns:
        103-length arrays from raw root and air data formatted for input to
        forecasting model.
    """
    if type(k_root) != int:
        sys.exit('k_root must be an integer.')
    if k_root <= 0 :
        sys.exit('k_root must be an integer greater than zero')
    if type(H) != int:
        sys.exit('H must be an integer.')
    if H <= 0 :
        sys.exit('H must be an integer greater than zero')
        
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)
    
    def build_set(root_data, air_data, k_root, H,\
            addition, norm=False): 
        
        root_data = root_data[-(len(root_data) - 32):]
        
        input1 = []
        zeros1 = []
        ref_temp = []
        for each in window(root_data, k_root):
            input1.append(numpy.array(each))
            zeros1.append(numpy.array([i == 0 for i in each]))
            ref_temp.append(numpy.array([each[-1]]))
        input1 = numpy.array(input1)
        zeros1 = numpy.array(zeros1)
        ref_temp = numpy.array(ref_temp)
        input1 = numpy.array([numpy.array(i - i[-1]) for i in input1])
        input1 = numpy.array([numpy.delete(i, -1) for i in input1])
        zeros1 = numpy.array([numpy.delete(i, -1) for i in zeros1])
        
        input2 = []
        zeros2 = []
        for each in window(air_data, k_root):
            input2.append(numpy.array(each))
            zeros2.append(numpy.array([i == 0 for i in each]))
        input2=numpy.array(input2)
        zeros2 = numpy.array(zeros2)
        input2 = numpy.array([numpy.array(i - i[-1]) for i in input2])
        input2 = numpy.array([numpy.delete(i, -1) for i in input2])
        zeros2 = numpy.array([numpy.delete(i, -1) for i in zeros2])

        
        y_target = []
        for each in window(root_data, H):
            y_target.append(numpy.array(each))
        y_target = numpy.array(y_target)
        y_target = numpy.array([numpy.array(i - i[0]) for i in y_target])
        y_target = numpy.array([numpy.delete(i, 0) for i in y_target])
        
        input1=input1[:-H]
        zeros1=zeros1[:-H]
        ref_temp = ref_temp[:-H]
        input2=input2[:-(H + 32)]
        y_target=y_target[-(len(y_target)-k_root):]
        
        if addition:
            input2 = []
            zeros2 = []
            for each in window(air_data, k_root + 32):
                input2.append(numpy.array(each))
                zeros2.append(numpy.array([i == 0 for i in each]))
            input2=numpy.array(input2)
            zeros2 = numpy.array(zeros2)
            input2 = numpy.array([numpy.array(i - i[-1]) for i in input2])
            input2 = numpy.array([numpy.delete(i, -1) for i in input2])
            zeros2 = numpy.array([numpy.delete(i, -1) for i in zeros2])
            
            input2=input2[:-H]
            zeros2=zeros2[:-H]
        
        # concatenate root and external temperature
        input = numpy.concatenate((input1, ref_temp, input2),axis=1)
        zerosrt = numpy.array([i == 0 for i in ref_temp])
        zeros = numpy.concatenate((zeros1, zerosrt, zeros2),axis=1)
        
        
        if norm:
            mean_input = input.mean(axis=0)
            std_input  = input.std(axis=0)
            mean_target = y_target.mean(axis=0)
            std_target  = y_target.std(axis=0)

            input    = (input - mean_input)/std_input
            y_target = (y_target - mean_target)/std_target
            return [input, y_target, mean_input, std_input, mean_target, std_target] 
        else:
            return [input, zeros]
           
    # use only one month of data for our purposes

    root_train_s = nrg.ran_c(root[0], imp_pc, c)
    air_train_s = nrg.ran_c(air[0], imp_pc, c)

        
    #root_test = root[13]     ; air_test = air[13]
    
    
    input_train_s, zeros = build_set(root_train_s, air_train_s, k_root,
                                   H, addition, norm=False)
    #input_test, y_test      = build_set(root_test, air_test, k_root,
    #                              H, addition, norm=False)
    
    #rval = [(input_train_s, y_train_s),
    #        (input_test, y_test)]
    rval = input_train_s, zeros
    return rval
'''       
# move to /output_folder
os.chdir("../")
os.chdir(os.getcwd() + "/output_folder")

# load weights of the network
f = cPickle.load(open('weights_summer.pkl','rb'))
W0     = f[0][0].get_value(borrow=True)
hbias0 = f[0][1].get_value(borrow=True)
vbias0 = f[0][2].get_value(borrow=True)

W1     = f[1][0].get_value(borrow=True)
hbias1 = f[1][1].get_value(borrow=True)
vbias1 = f[1][2].get_value(borrow=True)

W2     = f[2][0].get_value(borrow=True)
hbias2 = f[2][1].get_value(borrow=True)
vbias2 = f[2][2].get_value(borrow=True)

W_out  = f[3][0].get_value(borrow=True)
b_out  = f[3][1].get_value(borrow=True)

# load normalization parameters
norm        = numpy.load('normalization.npz')
mean_input  = norm['arr_0'][0]
std_input   = norm['arr_0'][1]
mean_output = norm['arr_0'][2]
std_output  = norm['arr_0'][3]
        
''' 
os.chdir("../")
