"""
Code taken from: https://medium.com/@krzechowski/custom-group-convolution-for-tensorflow-2-fc74a83189ce
"""

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

from .utils import register_keras_custom_object

__all__ = ['GroupConv2D','GroupConv1D']

@register_keras_custom_object 
class GroupConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters,kernel_size,groups=2,**kwargs):
        tf.keras.layers.Conv2D.__init__(self,filters,kernel_size,**kwargs)
        self.groups = groups
        assert filters % groups == 0, "number of filters %r is not dividable by nuber of groups: %r" %(filters, groups)
        self.padding=self.padding.upper()

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]
            
        assert self.in_channels % self.groups == 0, "number of input channels %r is not dividable by nuber of groups: %r" %(self.in_channels, self.groups)
        
        self.groupConv = lambda i, k: tf.nn.conv2d(
            i, k, strides=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilation_rate, name=self.name
        )

        self.filter_shape = (
            self.kernel_size[0], self.kernel_size[1], int(self.in_channels / self.groups), self.filters
        )

        self.kernel = self.add_weight(
            name='kernel',
            shape=self.filter_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        self.built = True
    
    def call(self, inputs):
        if self.groups == 1:
            outputs = self.groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.groups, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=self.groups, value=self.kernel)
            convGroups = [self.groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]
            outputs = tf.concat(axis=3, values=convGroups)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format, name='bias_add')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
        
    def get_config(self):
        config = {"groups":self.groups}
        base_config = super(GroupConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
@register_keras_custom_object 
class GroupConv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters,kernel_size,groups=2,**kwargs):
        tf.keras.layers.Conv1D.__init__(self,filters,kernel_size,**kwargs)
        self.groups = groups
        assert filters % groups == 0, "number of filters %r is not dividable by nuber of groups: %r" %(filters, groups)
        self.padding=self.padding.upper()

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
            self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]
            
        assert self.in_channels % self.groups == 0, "number of input channels %r is not dividable by nuber of groups: %r" %(self.in_channels, self.groups)
        
        self.groupConv = lambda i, k: tf.nn.conv1d(
            i, k, stride=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilation_rate, name=self.name
        )

        self.filter_shape = (
            self.kernel_size[0], int(self.in_channels / self.groups), self.filters
        )

        self.kernel = self.add_weight(
            name='kernel',
            shape=self.filter_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        self.built = True
    
    def call(self, inputs):
        if self.groups == 1:
            outputs = self.groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=2, num_or_size_splits=self.groups, value=inputs)
            weightsGroups = tf.split(axis=2, num_or_size_splits=self.groups, value=self.kernel)
            convGroups = [self.groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]
            outputs = tf.concat(axis=2, values=convGroups)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format, name='bias_add')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
        
    def get_config(self):
        config = {"groups":self.groups}
        base_config = super(GroupConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))