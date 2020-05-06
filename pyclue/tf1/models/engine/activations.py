#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that's not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == 'linear':
        return None
    elif act == 'sigmoid':
        return sigmoid
    elif act == 'softmax':
        return softmax
    elif act == 'relu':
        return relu
    elif act == 'leaky_relu':
        return leaky_relu
    elif act == 'gelu' or act == 'gelu_tanh':
        return gelu
    elif act == 'gelu_erf':
        return gelu_erf
    elif act == 'tanh':
        return tanh
    elif act == 'swish':
        return swish
    else:
        raise ValueError('Unsupported activation: %s' % act)


def gelu(x):
    """Gaussian Error Linear Unit.（tanh）
    This is a smoother version of the RELU. Original paper: https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def gelu_erf(x):
    """Gaussian Error Linear Unit.（erf）"""
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def swish(x):
    """
    Source: Searching for Activation Functions (Ramachandran et al. 2017)
    https://arxiv.org/abs/1710.05941
    """
    return tf.nn.swish(x)


def sigmoid(x):
    return tf.math.sigmoid(x)


def softmax(x):
    return tf.nn.softmax(x)


def relu(x):
    return tf.nn.relu(x)


def leaky_relu(x, alpha=0.2):
    """
    Source: Rectifier Nonlinearities Improve Neural Network Acoustic Models.
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    """
    return tf.nn.leaky_relu(x, alpha=alpha)


def tanh(x):
    return tf.math.tanh(x)
