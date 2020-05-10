#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
import json

import tensorflow as tf

from pyclue.tf1.models.embeddings.bert_embedding import embedding_lookup, embedding_postprocessor
from pyclue.tf1.models.engine.layers import dropout
from pyclue.tf1.models.engine.utils import create_initializer, get_shape_list


class TextCnnConfig(object):

    def __init__(self,
                 vocab_size,
                 embedding_size=256,
                 hidden_size=512,
                 filter_sizes=[2, 3, 4],
                 dilations=None,
                 hidden_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.filter_sizes = filter_sizes
        self.dilations = dilations
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = TextCnnConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as f:
            text = f.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True) + '\n'


class TextCnnModel(object):

    def __init__(self, config_file):
        self.config = TextCnnConfig.from_json_file(config_file)

    def __call__(self, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
        if not is_training:
            self.config.hidden_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size, seq_length = input_shape

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope('embeddings'):
            word_embedding_output, output_embedding_table = embedding_lookup(
                input_ids=input_ids,
                vocab_size=self.config.vocab_size,
                embedding_size=self.config.embedding_size,
                initializer_range=self.config.initializer_range,
                word_embedding_name='word_embeddings')

            # embedding_output = embedding_postprocessor(
            #     input_tensor=word_embedding_output,
            #     use_token_type=True,
            #     token_type_ids=segment_ids,
            #     token_type_vocab_size=self.config.type_vocab_size,
            #     token_type_embedding_name='token_type_embeddings',
            #     use_position_embeddings=True,
            #     position_embedding_name='position_embeddings',
            #     initializer_range=self.config.initializer_range,
            #     max_position_embeddings=self.config.max_position_embeddings,
            #     dropout_prob=self.config.hidden_dropout_prob)

            embedding_output = tf.multiply(
                word_embedding_output, tf.expand_dims(tf.cast(input_mask, tf.float32), -1))

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                filters = tf.get_variable(
                    name='filters',
                    initializer=tf.truncated_normal(
                        shape=[filter_size, self.config.embedding_size, self.config.hidden_size]),
                    dtype=tf.float32)
                bias = tf.get_variable(
                    name='bias',
                    initializer=tf.constant(
                        0., shape=[self.config.hidden_size]),
                    dtype=tf.float32)
                conv = tf.nn.conv1d(
                    embedding_output,
                    filters=filters,
                    stride=1,
                    padding='VALID',
                    dilations=self.config.dilations,
                    name='conv')
                conv = tf.nn.bias_add(conv, bias)
                conv = tf.nn.relu(conv, name='relu')
                pooled = tf.nn.max_pool1d(
                    conv,
                    ksize=[1, self.config.max_position_embeddings - filter_size + 1, 1],
                    strides=1,
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        concat_pooled_outputs = tf.concat(pooled_outputs, axis=-1)

        with tf.variable_scope('flat-dropout-fnn'):
            flatten_size = concat_pooled_outputs.shape[-1] * concat_pooled_outputs.shape[-2]
            h_flatten = tf.reshape(
                concat_pooled_outputs,
                shape=[-1, flatten_size],
                name='flatten')
            h_dropout = tf.nn.dropout(
                h_flatten,
                rate=self.config.hidden_dropout_prob)
            output_weights = tf.get_variable(
                name='output_weights',
                shape=[flatten_size, num_labels],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                dtype=tf.float32)
            output_bias = tf.get_variable(
                name='output_bias',
                shape=[num_labels],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)

        with tf.variable_scope('loss'):
            logits = tf.matmul(h_dropout, output_weights)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.softmax(logits, axis=-1)
            predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            batch_loss = tf.reduce_mean(per_example_loss)

            return batch_loss, per_example_loss, probabilities, logits, predictions
