# -*- coding: utf-8 -*-

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


# def create_attention_mask_from_input_mask(from_tensor, to_mask):
#     # used in bert, dropped in albert, we don't use this function.
#     """Create 3D attention mask from a 2D tensor mask.
#     Args:
#       from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
#       to_mask: int32 Tensor of shape [batch_size, to_seq_length].
#     Returns:
#       float Tensor of shape [batch_size, from_seq_length, to_seq_length].
#     """
#     from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
#     batch_size = from_shape[0]
#     from_seq_length = from_shape[1]
#
#     to_shape = get_shape_list(to_mask, expected_rank=2)
#     to_seq_length = to_shape[1]
#
#     to_mask = tf.cast(
#         tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
#
#     # We don't assume that `from_tensor` is a mask (although it could be). We
#     # don't actually care if we attend *from* padding tokens (only *to* padding)
#     # tokens so we create a tensor of all ones.
#     #
#     # `broadcast_ones` = [batch_size, from_seq_length, 1]
#     broadcast_ones = tf.ones(
#         shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
#
#     # Here we broadcast along two dimensions to create the mask.
#     mask = broadcast_ones * to_mask
#
#     return mask
