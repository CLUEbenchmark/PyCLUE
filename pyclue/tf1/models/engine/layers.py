#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
from pyclue.tf1.models.engine.utils import create_initializer, get_shape_list

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_einsum=True):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    size_per_head = int(from_shape[2] / num_attention_heads)

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if batch_size is None or from_seq_length is None or to_seq_length is None:
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # `query_layer` = [B, F, N, H]
    q = dense_layer_3d(from_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), query_act,
                       use_einsum, "query")

    # `key_layer` = [B, T, N, H]
    k = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), key_act,
                       use_einsum, "key")
    # `value_layer` = [B, T, N, H]
    v = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), value_act,
                       use_einsum, "value")
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.transpose(v, [0, 2, 1, 3])
    if attention_mask is not None:
        attention_mask = tf.reshape(
            attention_mask, [batch_size, 1, to_seq_length, 1])
        # 'new_embeddings = [B, N, F, H]'
    new_embeddings = dot_product_attention(q, k, v, attention_mask,
                                           attention_probs_dropout_prob)

    return tf.transpose(new_embeddings, [0, 2, 1, 3])


def attention_ffn_block(layer_input,
                        hidden_size=768,
                        attention_mask=None,
                        num_attention_heads=1,
                        attention_head_size=64,
                        attention_probs_dropout_prob=0.0,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        initializer_range=0.02,
                        hidden_dropout_prob=0.0,
                        use_einsum=True):
    """A network with attention-ffn as sub-block.

    Args:
      layer_input: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      hidden_size: (optional) int, size of hidden layer.
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      attention_head_size: int. Size of attention head.
      attention_probs_dropout_prob: float. dropout probability for attention_layer
      intermediate_size: int. Size of intermediate hidden layer.
      intermediate_act_fn: (optional) Activation function for the intermediate
        layer.
      initializer_range: float. Range of the weight initializer.
      hidden_dropout_prob: (optional) float. Dropout probability of the hidden
        layer.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test

    Returns:
      layer output
    """

    with tf.variable_scope("attention_1"):
        with tf.variable_scope("self"):
            attention_output = attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                use_einsum=use_einsum)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
            attention_output = dense_layer_3d_proj(
                attention_output,
                hidden_size,
                attention_head_size,
                create_initializer(initializer_range),
                None,
                use_einsum=use_einsum,
                name="dense")
            attention_output = dropout(attention_output, hidden_dropout_prob)
    attention_output = layer_norm(attention_output + layer_input)
    with tf.variable_scope("ffn_1"):
        with tf.variable_scope("intermediate"):
            intermediate_output = dense_layer_2d(
                attention_output,
                intermediate_size,
                create_initializer(initializer_range),
                intermediate_act_fn,
                use_einsum=use_einsum,
                num_attention_heads=num_attention_heads,
                name="dense")
            with tf.variable_scope("output"):
                ffn_output = dense_layer_2d(
                    intermediate_output,
                    hidden_size,
                    create_initializer(initializer_range),
                    None,
                    use_einsum=use_einsum,
                    num_attention_heads=num_attention_heads,
                    name="dense")
            ffn_output = dropout(ffn_output, hidden_dropout_prob)
    ffn_output = layer_norm(ffn_output + attention_output)
    return ffn_output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_hidden_groups=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      inner_group_num=1,
                      intermediate_act_fn='gelu',
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      share_parameter_across_layers=False,
                      use_einsum=True):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    See the original paper:
    https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers_test (blocks) in the Transformer.
      num_hidden_groups: int. Number of group for the hidden layers_test, parameters
        in the same group are shared.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      inner_group_num: int, number of inner repetition of attention and ffn.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers_test.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers_test or just the final
        layer.
      share_parameter_across_layers: Whether to share attention_ffn_blocks
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.
    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = hidden_size // num_attention_heads
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers_test so that the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        prev_output = dense_layer_2d(
            input_tensor, hidden_size, create_initializer(initializer_range),
            None, use_einsum=use_einsum, name="embedding_hidden_mapping_in")
    else:
        prev_output = input_tensor

    all_layer_outputs = []

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE if share_parameter_across_layers else False):
        for layer_idx in range(num_hidden_layers):
            group_idx = int(layer_idx / num_hidden_layers * num_hidden_groups) \
                if share_parameter_across_layers else 0
            name_variable_scope = 'layer_%d'
            with tf.variable_scope('group_%d' % group_idx):
                with tf.name_scope(name_variable_scope % layer_idx):
                    layer_output = prev_output
                    if share_parameter_across_layers:
                        for inner_group_idx in range(inner_group_num):
                            with tf.variable_scope("inner_group_%d" % inner_group_idx):
                                layer_output = attention_ffn_block(
                                    layer_input=layer_output,
                                    hidden_size=hidden_size,
                                    attention_mask=attention_mask,
                                    num_attention_heads=num_attention_heads,
                                    attention_head_size=attention_head_size,
                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                    intermediate_size=intermediate_size,
                                    intermediate_act_fn=intermediate_act_fn,
                                    initializer_range=initializer_range,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    use_einsum=use_einsum)
                                prev_output = layer_output
                                all_layer_outputs.append(layer_output)
                    else:
                        layer_output = attention_ffn_block(
                            layer_input=layer_output,
                            hidden_size=hidden_size,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            attention_head_size=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            intermediate_size=intermediate_size,
                            intermediate_act_fn=intermediate_act_fn,
                            initializer_range=initializer_range,
                            hidden_dropout_prob=hidden_dropout_prob,
                            use_einsum=use_einsum)
                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)
    if do_return_all_layers:
        return all_layer_outputs
    else:
        return all_layer_outputs[-1]


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    """Implements einsum via matmul and reshape ops.

    Args:
      input_tensor: float Tensor of shape [<batch_dims>, <inner_dims>].
      w: float Tensor of shape [<inner_dims>, <outer_dims>].
      num_inner_dims: int. number of dimensions to use for inner products.

    Returns:
      float Tensor of shape [<batch_dims>, <outer_dims>].
    """
    input_shape = get_shape_list(input_tensor)
    w_shape = get_shape_list(w)
    batch_dims = input_shape[: -num_inner_dims]
    inner_dims = input_shape[-num_inner_dims:]
    outer_dims = w_shape[num_inner_dims:]
    inner_dim = np.prod(inner_dims)
    outer_dim = np.prod(outer_dims)
    if num_inner_dims > 1:
        input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
    if len(w_shape) > 2:
        w = tf.reshape(w, [inner_dim, outer_dim])
    ret = tf.matmul(input_tensor, w)
    if len(outer_dims) > 1:
        ret = tf.reshape(ret, batch_dims + outer_dims)
    return ret


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   head_size,
                   initializer,
                   activation,
                   use_einsum,
                   name=None):
    """A dense layer with 3D kernel.

    Args:
      input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
      num_attention_heads: Number of attention heads.
      head_size: The size per attention head.
      initializer: Kernel initializer.
      activation: Actication function.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test.
      name: The name scope of this layer.

    Returns:
      float logits Tensor.
    """

    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]

    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[hidden_size, num_attention_heads * head_size],
            initializer=initializer)
        w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
        b = tf.get_variable(
            name="bias",
            shape=[num_attention_heads * head_size],
            initializer=tf.zeros_initializer)
        b = tf.reshape(b, [num_attention_heads, head_size])
        if use_einsum:
            ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
        else:
            ret = einsum_via_matmul(input_tensor, w, 1)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        head_size,
                        initializer,
                        activation,
                        use_einsum,
                        name=None):
    """A dense layer with 3D kernel for projection.

    Args:
      input_tensor: float Tensor of shape [batch,from_seq_length,
        num_attention_heads, size_per_head].
      hidden_size: The size of hidden layer.
      head_size: The size of head.
      initializer: Kernel initializer.
      activation: Actication function.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test.
      name: The name scope of this layer.

    Returns:
      float logits Tensor.
    """
    input_shape = get_shape_list(input_tensor)
    num_attention_heads = input_shape[2]
    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[num_attention_heads * head_size, hidden_size],
            initializer=initializer)
        w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
        b = tf.get_variable(
            name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)
        if use_einsum:
            ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
        else:
            ret = einsum_via_matmul(input_tensor, w, 2)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   use_einsum,
                   num_attention_heads=1,
                   name=None):
    """A dense layer with 2D kernel.

    Args:
      input_tensor: Float tensor with rank 3.
      output_size: The size of output dimension.
      initializer: Kernel initializer.
      activation: Activation function.
      use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers_test.
      num_attention_heads: number of attention head in attention layer.
      name: The name scope of this layer.

    Returns:
      float logits Tensor.
    """
    del num_attention_heads  # unused
    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]
    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[hidden_size, output_size],
            initializer=initializer)
        b = tf.get_variable(
            name="bias", shape=[output_size], initializer=tf.zeros_initializer)
        if use_einsum:
            ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
        else:
            ret = tf.matmul(input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def dot_product_attention(q, k, v, bias, dropout_rate=0.0):
    """Dot-product attention.

    Args:
      q: Tensor with shape [..., length_q, depth_k].
      k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
      v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
      bias: bias Tensor (see attention_bias())
      dropout_rate: a float.

    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    logits = tf.multiply(logits, 1.0 / math.sqrt(float(get_shape_list(q)[-1])))
    if bias is not None:
        # `attention_mask` = [B, T]
        from_shape = get_shape_list(q)
        if len(from_shape) == 4:
            broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)
        elif len(from_shape) == 5:
            # from_shape = [B, N, Block_num, block_size, depth]#
            broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], from_shape[3],
                                      1], tf.float32)
        else:
            raise ValueError('wrong dimension for from_shape')

        bias = tf.matmul(broadcast_ones,
                         tf.cast(bias, tf.float32), transpose_b=True)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - bias) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        logits += adder
    else:
        adder = 0.0

    attention_probs = tf.nn.softmax(logits, name="attention_probs")
    attention_probs = dropout(attention_probs, dropout_rate)
    return tf.matmul(attention_probs, v)
