# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 15:04:24 2015

@author: heiligenstein
"""

import os
import numpy
import nrg

import cPickle
import load_data_array_imp as ld

print 'Loading datasets...'
datasets = ld.load_dataset(36, 36, addition=True)
train_datasets_x, train_datasets_y = datasets[0]


#test_datasets_x, test_datasets_y = datasets[1]

# move to /output_folder
os.chdir("output_folder")
print 'Loading parameters...'

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
        
        
os.chdir("..")

def threek_plug(reclass_iters, corr_pc, c_ones):
    
    #train_datasets_x_corr, zeros = ld.load_dataset_imp(36, 36, corr_pc, c_ones, addition=True)

    reclass_iters, corr_pc, c_ones = int(reclass_iters), int(corr_pc), int(c_ones)

    def MAE(pred_ys, true_ys):
        return sum(numpy.abs(true_ys - pred_ys)) / len(pred_ys)
        
    def MSE(pred_ys, true_ys):
        return sum((true_ys - pred_ys)**2) / len(pred_ys)
        
    def is_zero(array):
        return numpy.array([i == 0 for i in array]), lambda z: z.nonzero()[0]

    def zeros_indices(array):
        return numpy.where(array == 0)[0]
    
    def format_input(raw_x):
        zeros = zeros_indices(raw_x[:-1])
        root_in = raw_x[:36]
        ref_temp_r = numpy.array([raw_x[35]])
        air_in = raw_x[36:]
        #ref_temp_a = numpy.array([air_in[-1]])
        
        root_in = root_in - root_in[-1]
        root_in = numpy.delete(root_in, -1)
        air_in = air_in - air_in[-1]
        air_in = numpy.delete(air_in, -1)
        
        formatted_array = numpy.concatenate((root_in, ref_temp_r, air_in), axis=1)
        
        return formatted_array, zeros
        
    '''            
    def deformat_gen(gen, ref_temp_r, ref_temp_a):
        gen_root = gen[:35] + ref_temp_r
        gen_air = gen[36:] + ref_temp_a
        gen_out = numpy.concatenate((gen_root, ref_temp_r, gen_air, ref_temp_a))
        return gen_out'''
    
    
    def sigmoid(z):
        return  1 / (1 + numpy.exp(-z))
        
        
    def up(x_form, l3s=False):
        # hidden layer predictions
        #input_array, rt_r, rt_a = format_input(raw_array) # format input

        norm_input = (x_form - mean_input)/std_input # normalize input

        fe0 = nrg.fe_gbv(norm_input, vbias0, hbias0, W0)
        layer1probs   = sigmoid(numpy.dot(norm_input, W0) + hbias0)
        layer1states  = numpy.random.binomial(size=layer1probs.shape, 
                                              n=1, p=layer1probs)
        fe1 = nrg.fe_v(layer1states, vbias1, hbias1, W1)
        layer2probs   = sigmoid(numpy.dot(layer1states, W1) + hbias1)
        layer2states  = numpy.random.binomial(size=layer2probs.shape,
                                              n=1, p=layer2probs)
        fe2 = nrg.fe_v(layer2states, vbias2, hbias2, W2)    
        layer3probs   = sigmoid(numpy.dot(layer2states, W2) + hbias2)
        layer3states  = numpy.random.binomial(size=layer3probs.shape,
                                              n=1, p=layer3probs)
        
        tclass   = numpy.dot(layer3states, W_out) + b_out #clayerprobs
        pred_y = (tclass * std_output + mean_output)
        
        FE = fe0 + fe1 + fe2
        
        if l3s:
            out = tclass, FE, layer3states
        else:
            out = pred_y, FE          
        return out
    
    
    def generate(tclass, up_output, numCDiters):
        neg3states = up_output # to initialize the loop
        for k in range(numCDiters):
            neg2probs  = sigmoid(numpy.dot(neg3states,W2.T) + vbias2)
            neg2states = numpy.random.binomial(size=neg2probs.shape, 
                                              n=1, p=neg2probs)
            neglabprobs  = tclass # label clamped
            neg3pprobs  = sigmoid(numpy.dot(neg2states,W2) + numpy.dot(neglabprobs, W_out.T) + hbias2)
            neg3states = numpy.random.binomial(size=neg3pprobs.shape, 
                                              n=1, p=neg3pprobs)
    
        sleep2states = neg2states
        sleep1probs = sigmoid(numpy.dot(sleep2states,W1.T) + vbias1)
        sleep1states = numpy.random.binomial(size=sleep1probs.shape, 
                                              n=1, p=sleep1probs)
        sleep0probs  = sigmoid(numpy.dot(sleep1states,W0.T) + vbias0)
        gen_denorm = (sleep0probs * std_input) + mean_input # denormalize
        return gen_denorm
        

        
    def up_down(x_form, imp_zeros, reclass_iters):
        #input_array, rt_r, rt_a = format_input(raw_array)
        tclass, FE, up_pass = up(x_form, l3s=True) # up (classify)
        forecast = (tclass * std_output + mean_output) # denorm and reformat predictions
    
        threshold = 38
        CDiters = 500
    
        if FE > threshold:
            #print 'Reclassifying...'
            cl_r = numpy.zeros_like(tclass)
            FE_r = []
            up_pass_r = numpy.zeros_like(up_pass)
            for i in xrange(reclass_iters):
                mtclass, mFE, mup_pass = up(x_form, l3s=True)
                cl_r += mtclass
                FE_r.append(mFE)
                up_pass_r += mup_pass
            tcl_r = cl_r / reclass_iters # average the reclassifications
            forecast_r = (tcl_r * std_output + mean_output) # denorm and reformat predictions
            FE_r = numpy.mean(FE_r)
            up_pass_r = up_pass_r / reclass_iters
            
            gen = generate(tcl_r, up_pass_r, CDiters) # down (generate)
            
            #gen_deform = deformat_gen(gen_denorm, rt_r, rt_a) # deformat
            
            # plug in to imputed array
            imp_recon = numpy.array(x_form)
            #zeros, indices = is_zero(imp_recon)
            imp_recon[imp_zeros] = gen[imp_zeros]

            out = forecast_r, FE_r, gen, imp_recon
        else:
            out = forecast, FE, x_form, x_form
        return out
    
    # masking, no reclass, no reconstruction
    if reclass_iters == 0:
        FEs_imp = []    ; MAEs_imp_y = []    ; MSEs_imp_y = []
        counter = 0
        for x, y in zip(train_datasets_x, train_datasets_y):
            if counter%100 == 0:
                print 'input vector', counter
            x_corr = nrg.ran_c(x, corr_pc, c_ones)
            rt_r = x_corr[35]
            x_corr, imp_zeros = format_input(x_corr)
            x_form = format_input(x)[0]
            
            pred_y_imp, FE_imp = up(x_corr)
            pred_y_imp = pred_y_imp + rt_r
            #y = y - y[0]
            FEs_imp.append(FE_imp)
            MAEs_imp_y.append(MAE(pred_y_imp, y))
            MSEs_imp_y.append(MSE(pred_y_imp, y))

            counter += 1
        FE_out_file = FEs_imp
        MAE_out_file = MAEs_imp_y
        MSE_out_file = MSEs_imp_y
    
    # reclassification
    else:
        FEs_corr_r = []  ; MAEs_corr_y_r = []  ; MSEs_corr_y_r = []
        MAEs_corr_x = [] ; MSEs_corr_x = [] # corr vs. x
        MAEs_gen_x_r = [] ; MSEs_gen_x_r = [] # gen vs. x
        MAEs_imp_x_r = [] ; MSEs_imp_x_r = [] # imp vs. x      
        FEs_gen = [] ; FEs_imp = []
        MAEs_gen_y = [] ; MSEs_gen_y = []
        MAEs_imp_y = [] ; MSEs_imp_y = []
        
        counter = 0
        for x, y in zip(train_datasets_x, train_datasets_y):
            if counter%100 == 0:
                print 'input vector', counter
            x_corr = nrg.ran_c(x, corr_pc, c_ones)
            rt_r = x_corr[35]
            if rt_r == 0:
                x_corr_form = format_input(x_corr)[0]
                tclass, FE, layer3states = up(x_corr_form, l3s = True)
                x_corr[35] = generate(tclass, layer3states, reclass_iters)[35]
            x_corr, imp_zeros = format_input(x_corr)
            x_form = format_input(x)[0]

            pred_y_corr_r, FE_corr_r, gen_r, imp_r = up_down(x_corr, imp_zeros,
                                                    reclass_iters=reclass_iters)
            pred_y_corr_r = pred_y_corr_r + rt_r
            #y = y - y[0]
            FEs_corr_r.append(FE_corr_r)
            MAEs_corr_y_r.append(MAE(pred_y_corr_r, y))
            MSEs_corr_y_r.append(MSE(pred_y_corr_r, y))
                        
            MAEs_corr_x.append(MAE(x_corr, x_form))
            MSEs_corr_x.append(MSE(x_corr, x_form))
            
            MAEs_gen_x_r.append(MAE(gen_r, x_form))
            MSEs_gen_x_r.append(MSE(gen_r, x_form))
            
            MAEs_imp_x_r.append(MAE(imp_r, x_form))
            MSEs_imp_x_r.append(MSE(imp_r, x_form))


            #-----------------------------------------------------------------
            # classification with generated input            
            pred_y_gen, FE_gen = up(gen_r)
            pred_y_imp, FE_imp = up(imp_r)            
            
            #pred_y_gen = pred_y_gen + rt_r
            #pred_y_plug = pred_y_plug + rt_r
            
            FEs_gen.append(FE_gen)
            FEs_imp.append(FE_imp)            
            MAEs_gen_y.append(MAE(pred_y_gen, y))
            MSEs_gen_y.append(MSE(pred_y_gen, y))
            MAEs_imp_y.append(MAE(pred_y_imp, y))
            MSEs_imp_y.append(MSE(pred_y_imp, y))
            counter += 1
    
        FE_out_file = [FEs_corr_r, FEs_gen, FEs_imp]
        MAE_out_file = [MAEs_corr_y_r,
                        MAEs_gen_x_r,
                        MAEs_imp_x_r,
                        MAEs_corr_x,
                        MAEs_gen_y,
                        MAEs_imp_y]
        MSE_out_file = [MSEs_corr_y_r,
                        MSEs_gen_x_r,
                        MSEs_imp_x_r,
                        MSEs_corr_x,
                        MSEs_gen_y,
                        MSEs_imp_y]

    FE_save_name = 'temp_FEs'+ str(reclass_iters)+ '_' +\
                str(corr_pc)+ '_' + str(c_ones) + 'p.txt'
    
    MAE_save_name = 'temp_MAEs'+ str(reclass_iters)+ '_' +\
                str(corr_pc)+ '_' + str(c_ones) + 'p.txt'

    MSE_save_name = 'temp_MSEs'+ str(reclass_iters)+ '_' +\
                str(corr_pc)+ '_' + str(c_ones) + 'p.txt'

    os.chdir("test_results")
    numpy.savetxt(str(FE_save_name), FE_out_file, fmt='%1.2e')
    print 'Results saved in /test_results as', FE_save_name
    numpy.savetxt(str(MAE_save_name), MAE_out_file, fmt='%1.2e')
    print 'Results saved in /test_results as', MAE_save_name
    numpy.savetxt(str(MSE_save_name), MSE_out_file, fmt='%1.2e')
    print 'Results saved in /test_results as', MSE_save_name
    os.chdir("..")


if __name__ == "__main__":
    import sys
    threek_plug(sys.argv[1], sys.argv[2], sys.argv[3])