__all__=['tensorflow_add_conv2d_function','AddConv2D']

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import backend as K
import time

from .utils import register_keras_custom_object
    
##################################################################################################################################################
# CUSTOM ADD_CONV TENSORFLOW
##################################################################################################################################################
    
@tf.custom_gradient
def tf_adder(W, patches):
    def grad(dy):
        grad_W=tf.math.reduce_sum(tf.concat([tf.expand_dims(patches-W[:, i],axis=3) for i in range(W.shape[-1])],axis=3)*tf.expand_dims(dy,-1),axis=[0,1,2])
        ita=0.1
        grad_W_col = ita*math.sqrt(W.shape[0]*W.shape[1])*grad_W/tf.clip_by_value(tf.norm(grad_W, ord=2),1e-12,np.inf)
        
        grad_patches=tf.math.reduce_sum(tf.clip_by_value(tf.concat([tf.expand_dims(W[:, i]- patches,axis=3) for i in range(W.shape[-1])],axis=3),-1,1)*tf.expand_dims(dy,-1),axis=3)
        return grad_W,grad_patches
        
    output = tf.concat([-tf.math.reduce_sum(tf.abs(W[:, i]- patches), axis=3, keepdims=True) for i in range(W.shape[-1])],axis=3)
    return output,grad
    


"""
This add-convolution use the classical im2col algorithm which requires a considerable amount of memory : may overflow 
the gpu memory for large dataset such as ImageNet
"""
def add_conv2d_function(ix, w, strides=(1,1) ,padding="VALID",dilation_rate=(1,1)):
   # filter shape: [filter_height, filter_width, in_channels, out_channels]
   # flatten filters
   filter_height, filter_width, in_channels, out_channels= int(w.shape[0]), int(w.shape[1]), int(w.shape[2]), int(w.shape[3])
   ix_height, ix_width, ix_channels = int(ix.shape[1]), int(ix.shape[2]), int(ix.shape[3])
   
   padding = "SAME" if padding=="same" else "VALID" if padding=="valid" else padding
   
   flat_w = tf.reshape(w, [filter_height * filter_width * in_channels, out_channels])
   patches = tf.image.extract_patches(ix,sizes=[1, filter_height, filter_width, 1],strides=[1, strides[0], strides[1], 1],rates=[1, dilation_rate[0], dilation_rate[1], 1],padding=padding)
   
   #add_convolutions
   features=tf_adder(flat_w,patches)
   
   return features
   
"""
Layer of Add-convolutions
"""

@register_keras_custom_object    
class AddConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters,kernel_size,**kwargs):
        tf.keras.layers.Conv2D.__init__(self,filters,kernel_size,**kwargs)
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
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
            
        self.built = True


    def call(self, inputs):
    
        output = add_conv2d_function(inputs,self.kernel,strides=self.strides,padding=self.padding,dilation_rate=self.dilation_rate)
        if self.use_bias:
            output = K.bias_add(output, self.bias,data_format=self.data_format)
                
        return output
        
    def get_config(self):
        base_config = super(AddConv2D, self).get_config()
        return dict(list(base_config.items()))