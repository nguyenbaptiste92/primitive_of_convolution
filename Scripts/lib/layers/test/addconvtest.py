import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from torch.autograd import Function
import math
import tensorflow as tf
import time

import subprocess as sp
import os

from ..add_conv import *

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

##################################################################################################################################################
# ADD_CONV PYTORCH FROM https://github.com/huawei-noah/AdderNet
##################################################################################################################################################

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    
    X_col = torch.nn.functional.unfold(X, h_filter, dilation=1, padding=padding, stride=stride)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder.apply(W_col,X_col)
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out
    
class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs()
        output = output.sum(1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        return grad_W_col, grad_X_col
    
##################################################################################################################################################
# Test
##################################################################################################################################################

def test_adder(h_filter,input_channel,output_channel):

    image = np.random.rand(2,input_channel,32,32)
    kernel = np.random.rand(output_channel,input_channel,h_filter,h_filter)
    print(image.shape)
    print(kernel.shape)
    
    torch_image = tensor(image, requires_grad=True)
    torch_kernel = tensor(kernel, requires_grad=True)
    torch_x = adder2d_function(torch_image, torch_kernel, stride=1, padding=0)
    torch_x_reduce=torch.sum(torch_x)
    gradient_torch_x=torch.autograd.grad(torch_x_reduce, torch_kernel)[0]
    print(gradient_torch_x.size())
    
    tf_image = tf.convert_to_tensor(image)
    tf_image = tf.Variable(tf.transpose(tf_image, [0, 2, 3, 1]))
    tf_kernel = tf.convert_to_tensor(kernel)
    tf_kernel = tf.Variable(tf.transpose(tf_kernel, [2, 3, 1, 0]))
    tf_x = custom_add_conv2d_function(tf_image, tf_kernel, strides=1, padding="VALID")
    with tf.GradientTape() as tape:
        tf_x = custom_add_conv2d_function(tf_image, tf_kernel, strides=1, padding="VALID")
        tf_x_reduce = tf.math.reduce_sum(tf_x)
    gradient_tf_x=tape.gradient(tf_x_reduce, tf_kernel)
    print(gradient_tf_x.shape)
    
    tf_x = tf.transpose(tf_x, [0, 3, 1, 2])
    gradient_tf_x=tf.transpose(gradient_tf_x, [3, 2, 0, 1])
    if (np.all(torch_x.detach().numpy()-tf_x.numpy()<1e-5)):
        print("Forward success.")
    if (np.all(gradient_torch_x.detach().numpy()-gradient_tf_x.numpy()<1e-5)):
        print("Backward success.")

test_adder(3,5,7)