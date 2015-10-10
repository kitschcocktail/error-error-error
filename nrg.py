# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 13:50:13 2015

@author: heiligenstein
"""

import numpy
import random



def sigmoid(z):
    return  1 / (1 + numpy.exp(-z))

def normalize(input_layer):
    return (input_layer - mean_input)/std_input
    
def c0(array):
    return list(array).count(0)

def MAE(pred_ys, true_ys):
    return sum(numpy.abs(true_ys - pred_ys)) / len(pred_ys)

def MSE(pred_ys, true_ys):
    return sum((true_ys - pred_ys)**2) / len(pred_ys)


# for randomly scattered 0s
def ran_z(input_array, pc):
    """Imputes randomly scattered 0s."""
    masked = numpy.array(input_array)
    rand = numpy.random.binomial(size=len(input_array), n=1, p=float(pc)/100)
    for i in xrange(len(rand)):
        if rand[i] == 1:
            masked[i] = 0
    return masked
        
# for consecutive 0s
def ran_c(input_array, pc, len_outage):
    """Ramdomly imputes consecutive 0s."""
    # temperature is recorded every 10 mins, so 1 hr = 6 values.
    # 0 mapped to 1 value of data array 
    if len_outage == 0:
        len_outage = 1
    else:
        len_outage = len_outage*6
    impc = numpy.array(input_array)
    o_0 = list(input_array).count(0)
    length = len(input_array)
    num_zeros = int(length * float(pc) / 100) - o_0
    num_outs = int(num_zeros/len_outage)
    remaining_zeros = (num_zeros)%(len_outage)
    for i in range(num_outs):
        rand = numpy.random.randint(0, high=(length - len_outage))
        j = len_outage
        while j != 0:
            if impc[rand%length] != 0:
                impc[rand%length] = 0
                j -= 1
            rand += 1
    if remaining_zeros != 0:
        rand2 = numpy.random.randint(0, high=(length - remaining_zeros))
        j2 = remaining_zeros
        while j2 != 0:
            if impc[rand2%length] != 0:
                impc[rand2%length] = 0
                j2 -= 1
            rand2 +=1
    return impc
    
    
def ran_nan(input_array, pc, len_outage):
    """Ramdomly imputes consecutive 0s."""
    # temperature is recorded every 10 mins, so 1 hr = 6 values.
    # 0 mapped to 1 value of data array 
    if len_outage == 0:
        len_outage = 1
    else:
        len_outage = len_outage*6
    impc = numpy.array(input_array)
    o_0 = list(input_array).count(0)
    length = len(input_array)
    num_zeros = int(length * float(pc) / 100) - o_0
    num_outs = int(num_zeros/len_outage)
    remaining_zeros = (num_zeros)%(len_outage)
    for i in range(num_outs):
        rand = numpy.random.randint(0, high=(length - len_outage))
        j = len_outage
        while j != 0:
            if impc[rand%length] != numpy.nan:
                impc[rand%length] = numpy.nan
                j -= 1
            rand += 1
    if remaining_zeros != 0:
        rand2 = numpy.random.randint(0, high=(length - remaining_zeros))
        j2 = remaining_zeros
        while j2 != 0:
            if impc[rand2%length] != numpy.nan:
                impc[rand2%length] = numpy.nan
                j2 -= 1
            rand2 +=1
    return impc


########################
#   ENERGY FORMULAS    #
########################

                                         
def energy_vh(vlayer, vbias, hlayer, hbias, W):
    ''' Function to compute the energy of joint configuration (v, h)'''
    # e = -sumVisibleBias - sumHiddenBias - sumVisibleWeightHidden
    sum_vb = numpy.dot(vlayer, vbias)
    sum_hb = numpy.dot(hlayer, hbias)
    vWh = numpy.dot(numpy.dot(vlayer, W), hlayer)
    E = -sum_vb - sum_hb - vWh
    return E
    
def energy_xyh(vlayer, vbias, hlayer, hbias, clayer, cbias, W, U):
    """Function to compute the energy of visible vector given h and y"""
    # E(x, y, h) = -sumHiddenWeightVisible - sumvbiasVisible
    #              - sumhbiasHidden - sumcbiasClass - sumHiddenUClass
    vWh = numpy.dot(numpy.dot(vlayer, W), hlayer)
    sum_vb = numpy.dot(vlayer, vbias)
    sum_hb = numpy.dot(hlayer, hbias)
    sum_cb = numpy.dot(clayer, cbias)
    hUc = numpy.dot(numpy.dot(clayer, U), hlayer)
    E = - vWh - sum_vb - sum_hb - sum_cb - hUc
    return E
    

def fe_v(vlayer, vbias, hbias, W):
    ''' Function to compute free energy of visible vector v'''
    # f_e = -sumVisibleBias - sumLog(1 + exp(sumVisibleWeight + hbias))
    sum_vb = numpy.dot(vlayer, vbias)
    vW = numpy.dot(vlayer, W)
    sumLog = sum(numpy.log(1 + numpy.exp(vW + hbias)))
    FE = -sum_vb - sumLog
    return FE
    
def fe_v1(vlayer, vbias, hlayerprobs, hbias, W): # works
    '''Free energy, expected energy minus the entropy'''
    #f(v) = -sumVisibleBias - sumProbH(sumVisibleWeight + hbias) + entropy
    sum_vb = numpy.dot(vlayer,vbias)
    xj = numpy.dot(vlayer,W) + hbias
    B = numpy.dot(hlayerprobs,xj)
    H = sum((hlayerprobs*numpy.log(hlayerprobs)) + 
        ((1 - hlayerprobs)*numpy.log(1 - hlayerprobs)))
    FE1 = - sum_vb - B + H
    return FE1    

def fe_v2(vlayer, vbias, hbias, W): # works
    '''Free energy, expected energy minus the entropy'''
    #f(v) = -sumVisibleBias - sumProbH(sumVisibleWeight + hbias) + entropy
    sum_vb = numpy.dot(vlayer,vbias)
    xj = numpy.dot(vlayer,W) + hbias
    hlayerprobs = sigmoid(numpy.dot(vlayer,W) + hbias)
    B = numpy.dot(hlayerprobs,xj)
    H = sum((hlayerprobs*numpy.log(hlayerprobs)) + 
        ((1 - hlayerprobs)*numpy.log(1 - hlayerprobs)))
    FE2 = - sum_vb - B + H
    return FE2


def fe_v3(vlayer, vbias, hbias, W): # works
    '''Elfwing formula for free energy, expected energy minus the entropy.
    Uses hlayers instead of hbias'''
    #f(v) = -sumVisibleBias - sumProbH(sumVisibleWeight + hbias) + entropy
    hlayerprobs = sigmoid(numpy.dot(vlayer,W) + hbias)
    vW = numpy.dot(vlayer, W)
    vWhp = numpy.dot(vW, hlayerprobs)
    sum_vb = numpy.dot(vlayer, vbias)
    sum_hhp = numpy.dot(hbias, hlayerprobs)
    H = sum((hlayerprobs*numpy.log(hlayerprobs)) + 
           ((1 - hlayerprobs)*numpy.log(1 - hlayerprobs)))
    FE3 = -vWhp - sum_vb - sum_hhp + H
    return FE3
    
def fe_gbv(vlayer, vbias, hbias, W):
    '''Schmah function to compute free energy for GB values'''
    # f(gbv) = - sumLog(1 + exp(sumVisibleWeight + hbias)) + 1/2sum(Visible - vbias)^2
    vb = numpy.sum((vlayer - vbias)**2) / 2
    vW = numpy.dot(vlayer, W)
    sumLog = numpy.sum(numpy.log(1 + numpy.exp(vW + hbias)))
    FE_gbv = - sumLog + vb
    return FE_gbv
    
def fe_gbv_enrique(vlayer, vbias, hbias, W):            
    wx_b = numpy.dot(vlayer, W) + hbias
    vbias_term = 0.5 * numpy.dot((vlayer - vbias), (vlayer - vbias).T)
    hidden_term = numpy.sum(numpy.log(1 + numpy.exp(wx_b)))
    return -hidden_term - vbias_term

def fe_vc(vlayer, hbias, W, clayer, cbias, U):
    ''' Function to compute free energy of visible vector v and class c'''
    # f_e = -sumClassBias - sumLog(1 + exp(sumVisibleWeight + hbias + sumClassWeight))
    sum_cb = numpy.dot(clayer, cbias)
    vWbcU = (numpy.dot(vlayer, W)) + hbias + (numpy.dot(clayer, U))
    sumLog = sum(numpy.log(1 + numpy.exp(vWbcU)))
    F_vc = -sum_cb - sumLog
    return F_vc


def exp_nrg(vlayer, vbias, hbias, W):
    A = numpy.dot(vlayer,vbias)
    xj = numpy.dot(vlayer,W) + hbias
    hlayerprobs = sigmoid(numpy.dot(vlayer,W) + hbias)
    B = numpy.dot(hlayerprobs,xj)
    EE = - A - B
    return -EE


def prob_h_given_v(vlayer, W, hbias):
    '''Function to compute conditional probability of hidden given visible'''
    # probhv = product of sigmoid(sumVisibleWeight + hbias)
    probs = sigmoid(numpy.dot(vlayer,W) + hbias)
    return reduce(lambda x, y: x*y, probs)
    

def prob_v_given_h(hlayer, W, vbias):
    '''Function to compute conditional probabilit of visible given hidden'''
    # probvh = product of 
    probs = sigmoid(numpy.dot(hlayer,numpy.transpose(W)) + vbias)
    return reduce(lambda x, y: x*y, probs)

