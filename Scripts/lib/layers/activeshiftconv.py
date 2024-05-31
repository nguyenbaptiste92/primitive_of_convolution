__all__=['ActiveShift']

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import backend as K
import time

import lib.layers.active_shift_1d.active_shift_1d as active_shift_1d

import lib.layers.active_shift_2d.active_shift_2d as active_shift_2d
#from .utils import register_keras_custom_object

##################################################################################################################################################
# CONVOLUTION + ACTIVESHIFT FROM https://github.com/jyh2986/Active-Shift-TF
##################################################################################################################################################

class ActiveShiftConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters,shift_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),shift_regularizer=None,
shift_constraint=None,**kwargs):
        tf.keras.layers.Conv2D.__init__(self,filters,1,**kwargs)
        self.shift_initializer=shift_initializer
        self.shift_regularizer=shift_regularizer
        self.shift_constraint=shift_constraint
        self.PADDING = self.padding.upper()
        
    def build(self, input_shape):
        assert self.data_format == 'channels_last','The input format should be NHWC.'

        input_dim = input_shape[-1]
            
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None
            
        shift_shape = (2,input_dim)
        self.shift = self.add_weight(shape=shift_shape,
                                 initializer=self.shift_initializer,
                                 name='shift',
                                 regularizer=self.shift_regularizer,
                                 constraint=self.shift_constraint)
            
        self.built = True


    def call(self, inputs, training=True):
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        if training:
            output = active_shift_2d.active_shift2d_op(inputs, self.shift, strides=[1, 1, 1, 1], paddings=[0, 0, 0, 0])
        else:
            output = active_shift_2d.active_shift2d_op(inputs, tf.math.round(self.shift), strides=[1, 1, 1, 1], paddings=[0, 0, 0, 0])
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.nn.conv2d(output,self.kernel,strides=[1, 1, 1, 1],padding=self.PADDING,dilations=[1, 1, 1, 1])
        if self.use_bias:
            output = K.bias_add(output, self.bias,data_format=self.data_format)
        return output
        
    def get_config(self):
        config = {"shift_initializer":self.shift_initializer,"shift_regularizer":self.shift_regularizer,"shift_constraint":self.shift_constraint}
        base_config = super(ActiveShiftConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class ActiveShiftConv1D(tf.keras.layers.Conv1D):
    
    def __init__(self, filters,shift_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),shift_regularizer=None,
shift_constraint=None,**kwargs):
        tf.keras.layers.Conv1D.__init__(self,filters,1,**kwargs)
        self.shift_initializer=shift_initializer
        self.shift_regularizer=shift_regularizer
        self.shift_constraint=shift_constraint
        self.PADDING = self.padding.upper()
        
    def build(self, input_shape):
        assert self.data_format == 'channels_last','The input format should be NWC.'

        input_dim = input_shape[-1]
            
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None
            
        shift_shape = (input_dim,)
        self.shift = self.add_weight(shape=shift_shape,
                                 initializer=self.shift_initializer,
                                 name='shift',
                                 regularizer=self.shift_regularizer,
                                 constraint=self.shift_constraint)
            
        self.built = True


    def call(self, inputs, training=True):
        inputs = tf.transpose(inputs, [0, 2, 1])
        if training:
            output = active_shift_1d.active_shift1d_op(inputs, self.shift, strides=[1, 1, 1], paddings=[0, 0, 0])
        else:
            output = active_shift_1d.active_shift1d_op(inputs, tf.math.round(self.shift), strides=[1, 1, 1], paddings=[0, 0, 0])
        output = tf.transpose(output, [0, 2, 1])
        output = tf.nn.conv1d(output,self.kernel,stride=self.strides[0],padding=self.PADDING,dilations=[1, 1, 1])
        if self.use_bias:
            output = K.bias_add(output, self.bias,data_format=self.data_format)
        return output
        
    def get_config(self):
        config = {"shift_initializer":self.shift_initializer,"shift_regularizer":self.shift_regularizer,"shift_constraint":self.shift_constraint}
        base_config = super(ActiveShiftConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
