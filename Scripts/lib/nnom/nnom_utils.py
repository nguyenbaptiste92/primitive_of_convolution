'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com

    SPDX-License-Identifier: Apache-2.0

    Change Logs:
    Date           Author       Notes
    2019-02-05     Jianjia Ma   The first version


    This file provides:
    -> fake_quantisation layers which simulate the output quantisation on fixed-point NN models.
    -> weights/bias quantisation of Convolution and Dense Layer. "weight.h" file generations
    -> export "testing set" binary data file.
    -> print output ranges of each layers.

    Currently, this script does not support RNN (type) layers.
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Model

from sklearn import metrics
from .fully_connected_opt_weight_generation import *
import time
import warnings

""" 
this is the generate the test set data to a bin file
bin file can be used to validate the implementation in MCU

"""
def generate_test_bin(x, y, name='test_data_with_label.bin'):
    '''
    this method generate the
    :param x:  input x data size
    :param y:  input label (one hot label)
    :return:
    '''
    # quantize input x
    min_value = np.min(x)
    max_value = np.max(x)

    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits = 7 - int_bits
    x = np.round(x*2**dec_bits).astype(np.int8)
    # get label
    if(len(y.shape) >1):
        test_label = np.argwhere(y == 1).astype(np.int8)  # test data
        test_label = test_label[:, 1]
    else:
        test_label = y

    # get data
    dat = x.astype(dtype="byte")  # test data
    batch_size = dat.shape[0]     # total pices of data	
    dat = dat.flatten()           # flatten to get the total size.
    block_size = int(dat.size / batch_size) # this must be integer but... just to confirm

    # write (label x 128) (data_block x 128)
    label_batch = 128       # the Y-modem example uses 128 batch
    with open(name, 'wb') as f:
        start = 0
        while start <= (test_label.size - label_batch):
            test_label[start: start + label_batch].tofile(f)
            dat[block_size * start: block_size * (start + label_batch)].tofile(f)
            start += label_batch

        # the rest data
        if (start < test_label.size):
            rest_len = test_label.size - start
            new_labls = test_label[start:]
            new_labls = np.pad(new_labls, (0, label_batch - rest_len), mode='constant')
            new_labls.tofile(f)
            dat[block_size * start:].tofile(f)

    print("binary test file generated:", name)
    print("test data length:", test_label.size)
    return
    
#############################################################################################################################################################################################################
#CHANGING SHIFT
#############################################################################################################################################################################################################

def is_shift_layer(layer):
    ''' layer which can change the output encoding'''
    #FIXME: add more which will change the output shift
    if('input' in layer.name or
       'conv1d' in layer.name or
       'group_conv1d' in layer.name or
       'shift_conv1d' in layer.name or
       'conv2d' in layer.name or
       'add_conv2d' in layer.name or
       'shift_conv2d' in layer.name or
       'group_conv2d' in layer.name or
       'dense' in layer.name or
       'softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('add' in layer.name and 'zero' not in layer.name) or # the name, zero_padding contains 'add'
        'subtract' in layer.name or
        'multiply' in layer.name or
       ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')or
       ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
       ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return False

def is_shift_fixed(layer):
    ''' layer which shift to a fixed value'''
    #FIXME: add more which will change the output shift
    if('softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('activation' in layer.name and layer.get_config()['activation'] == 'softmax') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return  False
    
#############################################################################################################################################################################################################
#BATCH NORM FUSING
#############################################################################################################################################################################################################
    
def factorize_bn(layer):
     if ('batch_normalization' in layer.name):
         
         bn_gamma = layer.get_weights()[0]
         bn_beta = layer.get_weights()[1]
         bn_mean = layer.get_weights()[2]
         bn_variance = layer.get_weights()[3]
         
         epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
         batchnorm_weights = bn_gamma / np.sqrt(bn_variance + epsilon)
         batchnorm_bias = bn_beta - (bn_gamma * (bn_mean/ np.sqrt(bn_variance + epsilon)))
         layer.add_weight(name=layer.name+"/kernel",trainable=False,shape=batchnorm_weights.shape, dtype=tf.float32)
         layer.add_weight(name=layer.name+"/bias",trainable=False,shape=batchnorm_weights.shape, dtype=tf.float32)
         layer.set_weights([bn_gamma, bn_beta,bn_mean,bn_variance,batchnorm_weights,batchnorm_bias])
    
def fuse_bn_to_addconv(layer):
    # try to fuse BN layer to add convolution
    if ('add_conv' in layer.name) and ('batch_normalization' in layer._outbound_nodes[0].outbound_layer.name):

        print("fusing batch normalization to", layer.name)
        bn_layer = layer._outbound_nodes[0].outbound_layer
        c_w = layer.get_weights()[0]
        c_b = layer.get_weights()[1]
        print('original weight max', c_w.max(), 'min', c_w.min())
        print('original bias max', c_b.max(), 'min', c_b.min())
        bn_gamma = bn_layer.get_weights()[0]
        bn_beta = bn_layer.get_weights()[1]
        bn_mean = bn_layer.get_weights()[2]
        bn_variance = bn_layer.get_weights()[3]

        epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
        batchnorm_weights = bn_gamma / np.sqrt(bn_variance + epsilon)
        if ('conv2d' in layer.name):
            depth_dim = c_w.shape[3]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]
        # conv1d
        else:
            depth_dim = c_w.shape[2]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]

        print('fused bias max', c_b.max(), 'min', c_b.min())
        print('bn_factor max', batchnorm_weights.max(), 'min',batchnorm_weights.min())
        # write the weights back to the layer
        # after that, the model will be destroyed.. need a better way to pass the new weight
        layer.add_weight(name=layer.name+"/bn_factor",trainable=False,shape=batchnorm_weights.shape, dtype=tf.float32)
        layer.set_weights([c_w, c_b,batchnorm_weights])

def fuse_bn_to_conv(layer):
    # try to fuse BN layer to convolutional
    if ('conv' in layer.name) and \
            ('batch_normalization' in layer._outbound_nodes[0].outbound_layer.name):

        print("fusing batch normalization to", layer.name)
        bn_layer = layer._outbound_nodes[0].outbound_layer
        c_w = layer.get_weights()[0]
        c_b = layer.get_weights()[1]
        print('original weight max', c_w.max(), 'min', c_w.min())
        print('original bias max', c_b.max(), 'min', c_b.min())
        bn_gamma = bn_layer.get_weights()[0]
        bn_beta = bn_layer.get_weights()[1]
        bn_mean = bn_layer.get_weights()[2]
        bn_variance = bn_layer.get_weights()[3]

        if ('conv2d' in layer.name):
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for l in range(c_w.shape[3]):
                for k in range(c_w.shape[2]):
                    for j in range(c_w.shape[1]):
                        for i in range(c_w.shape[0]):
                            if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                                c_w[i][j][k][l] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)
                            else:
                                c_w[i][j][k][l] *= bn_gamma[l] / np.sqrt(bn_variance[l] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[2]
            else:
                depth_dim = c_w.shape[3]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]
        # conv1d
        else:
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for k in range(c_w.shape[2]):
                for j in range(c_w.shape[1]):
                    for i in range(c_w.shape[0]):
                        if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                            c_w[i][j][k] *= bn_gamma[j] / np.sqrt(bn_variance[j] + epsilon)
                        else:
                            c_w[i][j][k] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[1]
            else:
                depth_dim = c_w.shape[2]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]

        print('fused weight max', c_w.max(), 'min', c_w.min())
        print('fused bias max', c_b.max(), 'min', c_b.min())
        # write the weights back to the layer
        # after that, the model will be destroyed.. need a better way to pass the new weight
        if len(layer.get_weights())==2:
            layer.set_weights([c_w, c_b])
        elif len(layer.get_weights())==3:
            layer.set_weights([c_w, c_b,layer.get_weights()[2]])
            
#############################################################################################################################################################################################################
#QUANTISIZE WEIGHT AND SAVE THEM IN FILE
#############################################################################################################################################################################################################

def generate_weights(model, name='weights.h', format='hwc', shift_list=None,batch_norm=True):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = open(name, 'w')
    f.write('#include "nnom.h"\n\n')
    f.close()
    
    for curr_idx, layer in  enumerate(model.layers):
        if (not layer.weights):
            continue

        # before merging bn layer, check if the bn is "legally" after Conv
        if('batch_normalization' in layer.name) and \
            ('conv' not in layer.inbound_nodes[0].inbound_layers.name):
            raise  Exception('Currently only support batch_normalization after conv', layer.name,
                            layer._inbound_nodes[0].inbound_layers[0].name)

        # try to fuse BN layer to convolutional
        if ('conv' in layer.name) and ('add_conv' not in layer.name):
            if len(layer.outbound_nodes)>0:
                if ('batch_normalization' in layer.outbound_nodes[0].outbound_layer.name):
                    fuse_bn_to_conv(layer)
        
        if batch_norm:
            if ('batch_normalization' in layer.name and 'add_conv2d'in last_layer.name):
                factorize_bn(layer)
        else:
            if ('add_conv' in layer.name):
                if len(layer.outbound_nodes)>0:
                    if ('batch_normalization' in layer.outbound_nodes[0].outbound_layer.name):
                        fuse_bn_to_addconv(layer)
            
        last_layer=layer
        
        # generate weights and bias now
        weight_dec_shift = 0
        print('weights for layer', layer.name)
        for var in layer.weights:
            quantisize = True
            var_name = str(var.name)
            if("kernel" in var_name and 'batch_normalization' not in layer.name):
                var_values = layer.get_weights()[0] # weight
                print("  weight:", var_name)
            elif("bias" in var_name and 'batch_normalization' not in layer.name):
                var_values = layer.get_weights()[1] # bias
                print("  bias: ",var_name)
            elif ("kernel" in var_name and 'batch_normalization' in layer.name):
                var_values = layer.get_weights()[4] # weight
            elif ("bias" in var_name and 'batch_normalization' in layer.name):
                var_values = layer.get_weights()[5] # bias
            elif("bn_factor" in var.name):
                var_values = layer.get_weights()[2] # batch_norm factor for add convolution
                print("  bn_factor: ",var_name)
            elif ("shift" in var_name and "active_shift_conv" in layer.name):
                var_values = layer.get_weights()[2]
                quantisize = False
            else:
                continue
                
            if quantisize:
                print("  original shape: ", var_values.shape)
                min_value = np.min(var_values)
                max_value = np.max(var_values)
                int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    
                dec_bits = 7 - int_bits
                print("  dec bit", dec_bits)
                bSameAsKernel = False
                if(is_shift_layer(layer)):
                    bSameAsKernel = False
                    inp = layer.input.name.replace(':','/').split('/')[0]
                    input_encoding = shift_list[inp]
                    if ("kernel" in var_name):
                        weight_dec_shift = dec_bits
                    elif ("bias" in var_name):
                        shift = input_encoding+weight_dec_shift-dec_bits
                        if(shift < 0):
                            bSameAsKernel = True
                if(shift_list is None or bSameAsKernel):
                    # check if bias shift > weight shift, then reduce bias shift to weight shift	
                    if ("kernel" in var_name):
                        weight_dec_shift = dec_bits	
                    else:	
                        if(dec_bits > weight_dec_shift):	
                            dec_bits = weight_dec_shift	
                    print("  new dec bit", dec_bits)
    
                
                # convert to [-128,128) or int8
                var_values = np.round(var_values * 2 ** dec_bits)
                var_name = var_name.replace('/', '_')
                var_name = var_name.replace(':', '_')
                with open(name, 'a') as f:
                    f.write('#define ' + var_name.upper() + ' {')
                # CHW format
                if ('chw' in format):
                    if "dense" in var_name and "kernel" in var_name:
                        transposed_wts = np.transpose(var_values)
                        transposed_wts = convert_to_x4_q7_weights(
                            np.reshape(transposed_wts, (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                    # all other kernels, bias stay the same
                    else:
                        transposed_wts = var_values
                # HWC format
                else:
                    if (len(var_values.shape) == 3):  # 1D convolution layer weights
                        transposed_wts = np.transpose(var_values, (2, 0, 1))
                    elif (len(var_values.shape) == 4):  # 2D convolution layer weights
                        transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                    else:  # fully connected layer weights or biases of any layer
                        # test, use opt weight reorder
                        if "dense" in var_name and "kernel" in var_name:
                            transposed_wts = np.transpose(var_values)
                            transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                        else:
                            transposed_wts = np.transpose(var_values)
    
                print("  reshape to:",transposed_wts.shape)
    
                with open(name, 'a') as f:
                    transposed_wts.tofile(f, sep=", ", format="%d")
                    f.write('}\n\n')
                    if ("bn_factor" in var_name):
                        f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')
                    if ("bias" in var_name):
                        f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')
                    if ("kernel" in var_name ):
                        f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')
                        
            else:
                if np.linalg.matrix_rank(var_values)==2:
                    var_values = np.transpose(var_values, (1, 0))
                var_name = var_name.replace('/', '_')
                var_name = var_name.replace(':', '_')
                transposed_wts = np.round(var_values).astype(np.int8)
                print(transposed_wts)
                with open(name, 'a') as f:
                    f.write('#define ' + var_name.upper() + ' {')
                    transposed_wts.tofile(f, sep=", ", format="%d")
                    f.write('}\n\n')

#############################################################################################################################################################################################################
#GET LAYER OUTPUT RANGE
############################################################################################################################################################################################################# 

def layers_output_ranges(model, x_test, quantize_method='max_min', calibrate_size=1000,batch_norm=True):
    # limit the test data size
    np.random.shuffle(x_test)
    if(x_test.shape[0] > calibrate_size):
        x_test = x_test[:1000]
    # test, show the output ranges
    shift_list = {}
    # FIXME: only support one input
    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    last_layer = None

    for layer in L: # layer loop
        if("input" in layer.name):
            features = x_test
        else:
            # batch_normalization will need to be handled differently, since we are fusing the weight to its predecessor.
            # sigmoid and tanh are different, their shift is fixed to 7
            if(is_shift_layer(layer) or
                ('batch_normalization' in layer.name)):
                layer_model = Model(inputs=model.input, outputs=layer.output)
                features = layer_model.predict(x_test)
            else:
                # leave the features not changed, so this layer shift will be the same
                # as its inputs
                pass
        #  calculate no saturation shift
        max_val = features.max()
        min_val = features.min()
        int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bits = 7 - int_bits

        # saturation shift, using KLD method
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        if('kld' in quantize_method and not is_shift_fixed(layer) and "input" not in layer.name and "dense" not in layer.name): # test, also do not use kld in input layer
            import scipy.stats
            abs_max = max(abs(max_val), abs(min_val))
            small_var = 1e-5
            bins = np.arange(-abs_max, abs_max, abs_max/2048*2)
            q_bins = np.arange(-abs_max, abs_max, abs_max/256*2)
            flat_hist = np.histogram(features.flatten(), bins=bins)[0]
            kl_loss = []
            kl_shifts = []
            for shift in range(4):
                t = 2 ** (dec_bits + shift)     # 2-based threshold
                act = np.round(features.flatten() * t)
                act = act / t
                act = np.clip(act, -128/t, 127/t)
                act = np.histogram(act, bins=q_bins)[0]
                act_hist = np.zeros(2047)
                chunk = int(2048/256)
                for i in range(int(255)):
                    none_zero = np.count_nonzero(flat_hist[i*chunk:(i+1)*chunk])
                    if none_zero == 0:
                        continue
                    for j in range(chunk):
                        act_hist[i*chunk+j] = act[i]/none_zero if flat_hist[i*chunk+j] != 0 else 0
                flat_hist[flat_hist==0] = small_var
                act_hist[act_hist==0] = small_var
                kl = scipy.stats.entropy(flat_hist, act_hist)
                kl_loss.append(kl)
                kl_shifts.append(dec_bits + shift)
                """
                ax = plt.subplot(8, 1, shift+1)
                ax.plot(flat_hist)
                ax.plot(act_hist)
                """
            new_dec = kl_shifts[np.argmin(kl_loss)] # set the dec_bit to the KLD results
            #plt.show()
            print("KLD loss", kl_loss)
            print("KLD shift", kl_shifts)
            if(new_dec != dec_bits):
                print(layer.name,"is using KLD method, original shift",dec_bits, "KLD results", new_dec)
                dec_bits = new_dec

        print( layer.name, "max value:", max_val, "min value:", min_val,"dec bit", dec_bits)
        # record the shift
        if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
            shift_list[layer.name.split(':')[0]] = dec_bits
        else:
            shift_list[layer.name] = dec_bits
        #if ('batch_normalization' in layer.name):
        if batch_norm:
            if ('batch_normalization' in layer.name and 'add_conv2d' not in last_layer.name):
                shift_list[last_layer.name] = dec_bits  # use the bn layer shift to update the last layer.
        else:
            if ('batch_normalization' in layer.name):
                shift_list[last_layer.name] = dec_bits  # use the bn layer shift to update the last layer.
        last_layer = layer

    LM = {}
    for layer in model.layers:
        LM[layer.name] = layer
    L = [l for l in model.layers[1:]]
    L.reverse()

    def update_previous_layer_shift(layer, Q):
        if(type(layer.input) == list):
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if('input' in iname):
                    continue
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Q)
        else:
            iname = layer.input.name.split('/')[0]
            if('input' in iname):
                return
            shift_list[iname] = Qmin
            if(not is_shift_layer(LM[iname])):
                update_previous_layer_shift(LM[iname], Q)
    for layer in L:
        if(type(layer.input) == list):
            iname = layer.input[0].name.split('/')[0]
            Qmin = shift_list[iname]
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if(shift_list[iname] < Qmin):
                    Qmin = shift_list[iname]
                if(shift_list[iname] != Qmin):
                    bFlag = True
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Qmin)
            print('set shift', Qmin, 'for the input of', layer.name, ':', [inp.name.split('/')[0] for inp in layer.input])
            if(not is_shift_layer(layer) or Qmin < shift_list[layer.name]): # update current layer's shift only when we cannot change the shift
                shift_list[layer.name] = Qmin
    print("shift list", shift_list)
    return shift_list

#############################################################################################################################################################################################################
#GENERATE MODEL
#############################################################################################################################################################################################################

"""
batch_norm is for add convolution: if False: we fuse addconv and batchnorm, if True: we separate them
"""
def generate_model(model, x_test, name='weights.h', format='hwc', quantize_method='max_min',batch_norm=True):
    shift_list = layers_output_ranges(model, x_test, quantize_method=quantize_method,batch_norm=batch_norm)
    print(shift_list)
    generate_weights(model, name=name, format=format, shift_list=shift_list,batch_norm=batch_norm)
    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    
    with open(name,'a') as fp:
        fp.write('\n/* output enconding for each layer */\n')
        for layer in L:
            if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                iname = layer.name.split(':')[0]
            else:
                iname = layer.name
            fp.write('#define %s_OUTPUT_SHIFT %s\n'%(iname.upper(), shift_list[iname]))
        fp.write('\n/* bias shift and output shift for each layer */\n')
        for layer in model.layers:
            if(is_shift_layer(layer)):
                iname = layer.name.upper()
                
                #Conv and Dense
                if(len(layer.weights) == 2 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name and "add_conv2d" not in layer.name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp, kname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                            iname, inp, kname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_LSHIFT must be bigger than 0\n#endif\n'.format(iname))
                
                #Shift Conv
                if(len(layer.weights) == 3 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name and "active_shift_conv" in layer.name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp, kname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                            iname, inp, kname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_LSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    
                #Add_conv
                if(len(layer.weights) == 3 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name and "bn_factor" in layer.weights[2].name and "add_conv2d" in layer.name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    bn_name = layer.weights[2].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    
                    fp.write('#define {0}_INPUT_LSHIFT ({1}_SHIFT-{2}_OUTPUT_SHIFT)\n'.format(
                            iname, kname, inp))
                    fp.write('#if {0}_INPUT_LSHIFT > 0\n#define {0}_MODE 1\n#define {0}_INTER_SHIFT {1}_SHIFT\n#define {0}_INTER_LSHIFT ({1}_SHIFT-{2}_OUTPUT_SHIFT)\n'.format(iname,kname, inp))
                    fp.write('#elif {0}_INPUT_LSHIFT < 0\n#define {0}_MODE 2\n#define {0}_INTER_SHIFT {2}_OUTPUT_SHIFT\n#define {0}_INTER_LSHIFT ({2}_OUTPUT_SHIFT-{1}_SHIFT)\n'.format(iname,kname, inp))
                    fp.write('#else\n#define {0}_MODE 0\n#define {0}_INTER_SHIFT {1}_SHIFT\n#define {0}_INTER_LSHIFT 0\n#endif\n'.format(iname,kname))
                    fp.write('#define {0}_OUTPUT_RSHIFT   ({0}_INTER_SHIFT+{1}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname,bn_name))
                    fp.write('#define {0}_BIAS_LSHIFT   ({0}_INTER_SHIFT+{2}_SHIFT-{1}_SHIFT)\n'.format(
                            iname, bname,bn_name))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_LSHIFT must be bigger than 0\n#endif\n'.format(iname))
                
                #Add_conv    
                if(len(layer.weights) == 2 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name and "add_conv2d" in layer.name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    
                    fp.write('#define {0}_INPUT_LSHIFT ({1}_SHIFT-{2}_OUTPUT_SHIFT)\n'.format(
                            iname, kname, inp))
                    fp.write('#if {0}_INPUT_LSHIFT > 0\n#define {0}_MODE 1\n#define {0}_INTER_SHIFT {1}_SHIFT\n#define {0}_INTER_LSHIFT ({1}_SHIFT-{2}_OUTPUT_SHIFT)\n'.format(iname,kname, inp))
                    fp.write('#elif {0}_INPUT_LSHIFT < 0\n#define {0}_MODE 2\n#define {0}_INTER_SHIFT {2}_OUTPUT_SHIFT\n#define {0}_INTER_LSHIFT ({2}_OUTPUT_SHIFT-{1}_SHIFT)\n'.format(iname,kname, inp))
                    fp.write('#else\n#define {0}_MODE 0\n#define {0}_INTER_SHIFT {1}_SHIFT\n#define {0}_INTER_LSHIFT 0\n#endif\n'.format(iname,kname))
                    fp.write('#define {0}_OUTPUT_RSHIFT   ({0}_INTER_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({0}_INTER_SHIFT-{1}_SHIFT)\n'.format(
                            iname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_LSHIFT must be bigger than 0\n#endif\n'.format(iname))
                
                # add, sub
                if (('add' in layer.name or'subtract' in layer.name) and 'add_conv2d' not in layer.name):
                    # only consider the first, they have been set to same in out_put_range()
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
                if ('multiply' in layer.name ):
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT*2-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    
            else:
                iname = layer.name.upper()
                #Batchnorm after add_conv    
                if ('batch_normalization' in layer.name and len(layer.weights) == 6):
                    kname = layer.weights[4].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[5].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    print
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp, kname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                            iname, inp, kname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_LSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    

        fp.write('\n/* weights for each layer */\n')
        LI = {}
        ID = 0
        def is_skipable_layer(layer):
            # FIXME: add more that could be skiped
            if('lambda' in layer.name or
               'dropout' in layer.name or
               ('batch_normalization' in layer.name and len(layer.weights) != 6) or
                ('flatten' in layer.name and 'chw' not in format)): # flatten layer can be skipped in HWC but have to present in CHW
                return True
            return False
        for id,layer in enumerate(L):
            if(is_skipable_layer(layer)):
                inp = layer.input.name.replace(':','/').split('/')[0]
                LI[layer.name] = (LI[inp][0], layer)
            else:
                if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                    LI[layer.name.split(':')[0]] = (ID, layer)
                else:
                    LI[layer.name] = (ID, layer)
                ID += 1

            if ('input' in layer.name or not layer.weights):
                continue
            for var in layer.weights:
                var_name = str(var.name).replace('/', '_').replace(':', '_')
                if("kernel" in var_name):
                    fp.write('static const int8_t %s_weights[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const nnom_weight_t %s_w = { (const void*)%s_weights, %s_OUTPUT_RSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))
                elif("bias" in var_name):
                    fp.write('static const int8_t %s_bias[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const nnom_bias_t %s_b = { (const void*)%s_bias, %s_BIAS_LSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))
                elif("shift" in var_name):
                    fp.write('static const int8_t %s_shift[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const nnom_shift_t %s_s = { (const void*)%s_shift};\n'%(layer.name,layer.name))
            
            if ('add_conv2d' in layer.name):
                liste_var_name= [str(var.name).replace('/', '_').replace(':', '_') for var in layer.weights if "bn_factor" in var.name]
                if len(liste_var_name)==0:
                    fp.write('static const nnom_addconv_parameter_t %s_parameter = { (const void*)NULL, %s_MODE, %s_INTER_LSHIFT};\n'%(layer.name, layer.name.upper(),layer.name.upper()))
                else:
                    fp.write('static const int8_t %s_bn_factor[] = %s;\n'%(layer.name, liste_var_name[0].upper()))
                    fp.write('static const nnom_addconv_parameter_t %s_parameter = { (const void*)%s_bn_factor, %s_MODE, %s_INTER_LSHIFT};\n'%(layer.name,layer.name, layer.name.upper(),layer.name.upper()))
                    
        fp.write('\n/* nnom model */\n')
        # FIXME: now only support one input and one output
        sz = 1
        #for d in model.input.shape[1:]:
        for d in model.input['input'].shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_input_data[%d];\n'%(sz))
        sz = 1
        #for d in model.output.shape[1:]:
        for d in model.output['label'].shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_output_data[%d];\n'%(sz))
        fp.write('static nnom_model_t* nnom_model_create(void)\n{\n')
        fp.write('\tstatic nnom_model_t model;\n')
        if(ID>32):
            fp.write('\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*%d);\n'%(ID+1))
            fp.write('\tif(NULL == layer) return NULL;\n')
        else:
            fp.write('\tnnom_layer_t* layer[%d];\n'%(ID+1))
        fp.write('\n\tnew_model(&model);\n\n')
        
        
        for layer in L:
            if(is_skipable_layer(layer)):
                continue
            #FIXME: need a better solution to seperate the input 'tensor' from other layers
            if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                id,_ = LI[layer.name.split(':')[0]]
            else:
                id,_ = LI[layer.name]

            if('input' in layer.name):
                try:
                    inshape = layer.input_shape[0][1:] # new changes in tf2?
                except:
                    inshape = layer.shape[1:]
                if (len(inshape) == 1):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(%d,1,1), nnom_input_data);\n' % (id, inshape[0]))
                elif (len(inshape) == 2):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(1,%d,%d), nnom_input_data);\n' % (id, inshape[0], inshape[1]))
                else:
                    fp.write('\tlayer[%d] = Input(shape%s, nnom_input_data);\n' % (id, inshape))

            # 1D Convolution Primitives
            elif('conv1d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DepthwiseConv1D({1}, kernel(1,{2}), stride(1,{3}), dilation(1,{4}), PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, 1, cfg['kernel_size'][0], cfg['strides'][0], cfg['dilation_rate'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                        
                                # Group
                elif ('group' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(GroupConv1D({1}, {2}, kernel(1,{3}), stride(1,{4}), dilation(1,{5}), PADDING_{6}, &{7}_w, &{7}_b), layer[{8}]);\n'.format(id, cfg['filters'], cfg['groups'], cfg['kernel_size'][0], cfg['strides'][0], cfg['dilation_rate'][0], cfg['padding'].upper(),layer.name, LI[inp][0]))
                        
                # Shift
                elif ("active_shift" in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ShiftConv1D({1}, stride(1,{2}), &{3}_w, &{3}_b, &{3}_s), layer[{4}]);\n'.format(id, cfg['filters'], cfg['strides'][0], layer.name, LI[inp][0]))
                    
                else:
                    fp.write('\tlayer[{0}] = model.hook(Conv1D({1}, kernel(1,{2}), stride(1,{3}), dilation(1,{4}), PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, cfg['filters'], cfg['kernel_size'][0], cfg['strides'][0], cfg['dilation_rate'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                        
            # 2D Convolution Primitives
            elif('conv2d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                
                # Depthwise
                if ('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DW_Conv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, 1, cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                        
                # Group
                elif ('group' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(GroupConv2D({1}, {2}, kernel{3}, stride{4}, dilation{5}, PADDING_{6}, &{7}_w, &{7}_b), layer[{8}]);\n'.format(id, cfg['filters'], cfg['groups'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),layer.name, LI[inp][0]))
                    
                # Shift
                elif ("active_shift" in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ShiftConv2D({1}, &{2}_w, &{2}_b, &{2}_s), layer[{3}]);\n'.format(id, cfg['filters'], layer.name, LI[inp][0]))
                 
                # Add    
                elif('add_conv2d' in layer.name):
                    if(len(layer.weights) == 3):
                        fp.write('\tlayer[{0}] = model.hook(BnAddConv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b, &{6}_parameter), layer[{7}]);\n'.format(
                                id, cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),
                                layer.name, LI[inp][0]))
                    else:
                        fp.write('\tlayer[{0}] = model.hook(AddConv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b, &{6}_parameter), layer[{7}]);\n'.format(
                                id, cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),
                                layer.name, LI[inp][0]))
                    
                # Standard    
                else:
                    #assert cfg['groups']==1,"Only Conv2D layers with groups==1 are supported, please use GroupConv2D layers."
                    if cfg['groups']==1:
                        fp.write('\tlayer[{0}] = model.hook(Conv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(id, cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),layer.name, LI[inp][0]))
                    else:
                        fp.write('\tlayer[{0}] = model.hook(GroupConv2D({1}, {2}, kernel{3}, stride{4}, dilation{5}, PADDING_{6}, &{7}_w, &{7}_b), layer[{8}]);\n'.format(id, cfg['filters'], cfg['groups'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),layer.name, LI[inp][0]))
                            
            #Batchnormalization for addconv
            elif ('batch_normalization' in layer.name and len(layer.weights) == 6):
                inp = layer.input.name.replace(':','/').split('/')[0]
                units = int(layer.weights[0].shape[0])
                fp.write('\tlayer[{0}] = model.hook(BatchNormalization({1}, &{2}_w, &{2}_b), layer[{3}]);\n'.format(
                    id, units, layer.name, LI[inp][0]))
                    
            # activations
            elif('activation' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if(cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
                if(cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                if(cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                elif(cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('re_lu' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
            # pooling
            elif('max_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(GlobalMaxPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(MaxPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(MaxPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif('average_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    # a global avg pool before softmax can be replace by sumpool in MCU (recommend)
                    if(layer == model.layers[-2] and 'Softmax' in model.layers[-1].output.name):
                        print(layer.name, 'has been replaced by GlobalSumPool()')
                        fp.write('\tlayer[%s] = model.hook(GlobalSumPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                    else:
                        fp.write('\tlayer[%s] = model.hook(GlobalAvgPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(AvgPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(AvgPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif ('up_sampling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(UpSample(kernel%s), layer[%d]);\n'%(id, cfg['size'],  LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(UpSample(kernel(1,{1})), layer[{2}]);\n'.format(
                        id,  cfg['size'][0], LI[inp][0]))
            # zero padding
            elif ('zero_padding' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['padding'][0][0], cfg['padding'][0][1], cfg['padding'][1][0],cfg['padding'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['padding'][0], cfg['padding'][1], LI[inp][0]))
            # Cropping
            elif ('cropping' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['cropping'][0][0], cfg['cropping'][0][1], cfg['cropping'][1][0],cfg['cropping'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['cropping'][0], cfg['cropping'][1], LI[inp][0]))

            # others
            elif('flatten' in layer.name): # flatten is needed in CHW backend but not needed in HWC
                inp = layer.input.name.replace(':', '/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Flatten(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('concatenate' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                cfg = layer.get_config()
                fp.write('\tlayer[%s] = model.mergex(Concat(%s), %s%s);\n'%(
                    id, cfg['axis'], len(inps), inX))
            elif('add' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Add(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('subtract' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Sub(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('multiply' in layer.name):
                warnings.warn("Warning mutiply is under testing")
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Mult(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('dense' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                fp.write('\tlayer[{0}] = model.hook(Dense({1}, &{2}_w, &{2}_b), layer[{3}]);\n'.format(
                    id, cfg['units'], layer.name, LI[inp][0]))
            elif('softmax' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            else:
                raise Exception('unsupported layer', layer.name, layer)
            
			
        # FIXME, test later.
        if('softmax' in layer.name
           or ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')):
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        elif len(layer.output.shape) == 4:
            fp.write('\tlayer[%s] = model.hook(Output(shape%s, nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1:], id))
        elif len(layer.output.shape) == 3:
            fp.write('\tlayer[%s] = model.hook(Output(shape(1,%s,%s), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], layer.output.shape[2], id))
        elif len(layer.output.shape) == 2:
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        else:
            raise Exception('unsupported output shape of the last layer', layer.name, layer)
        fp.write('\tmodel_compile(&model, layer[0], layer[%s]);\n'%(id+1))
        if(ID>32):
            fp.write('\tfree(layer);\n')
        fp.write('\treturn &model;\n}\n')
    with open('.shift_list','w') as fp:
        fp.write(str(shift_list))

