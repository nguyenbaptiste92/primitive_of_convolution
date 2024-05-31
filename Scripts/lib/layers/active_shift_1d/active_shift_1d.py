from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import os

import tensorflow as tf
from tensorflow.python.framework import ops

filename = str(Path(__file__).parent/"cc"/"active_shift_1d.so")
_active_shift1d_module = tf.load_op_library(filename)
active_shift1d_op = _active_shift1d_module.active__shift_1d
active_shift1d_grad_op = _active_shift1d_module.active__shift_1d__backprop


@ops.RegisterGradient("Active_Shift_1D")
def _active_shift1d_grad(op, grad):
  """The gradients for `active_shift1d`.
  Args:
    op: The `active_shift1d` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `active_shift1d` op.
  Returns:
    Gradients with respect to the input of `active_shift1d`."""
  
  data = op.inputs[0]
  shift = op.inputs[1]
  
  strides = op.get_attr('strides')
  paddings = op.get_attr('paddings')
  data_format = op.get_attr('data_format')
  normalize = op.get_attr('normalize')

  # compute gradient
  data_grad = active_shift1d_grad_op(data, shift, grad, strides, paddings, normalize, data_format)

  return data_grad  # List of one Tensor, since we have one input
