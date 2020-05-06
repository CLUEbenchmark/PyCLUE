#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pyclue.tf1.models.albert import AlbertConfig, AlbertModel
from pyclue.tf1.models.bert import BertConfig, BertModel
from pyclue.tf1.open_sources.configs import pretrained_names, pretrained_types
from pyclue.tf1.open_sources.download import get_pretrained_model
from pyclue.tf1.models.engine.layers import dropout


class SiameseModel(object):

    def __init__(self, model_name=None, model_type=None, vocab_file=None,
                 config_file=None, init_checkpoint_file=None):
        self.model_name = model_name.lower()
        if not model_name:
            assert all([model_type, vocab_file, config_file, init_checkpoint_file]), \
                'If not given model_name provided by open_sources, ' \
                'you should specify all the details of model.'
            self.model_type = model_type.lower()
            self.vocab_file = vocab_file
            self.config_file = config_file
            self.init_checkpoint_file = init_checkpoint_file
        else:
            assert self.model_name in pretrained_names, \
                '%s not provided by open_sources.' % self.model_name
            self.model_type = pretrained_types.get(self.model_name).split('_')[0]
            self._from_pretrained()
        self._set_pretrained_model()

    def _from_pretrained(self):
        pretrained_dir = get_pretrained_model(pretrained_name=self.model_name)
        self.vocab_file = os.path.join(pretrained_dir, 'vocab.txt')
        self.config_file = os.path.join(pretrained_dir, 'config.json')
        self.init_checkpoint_file = os.path.join(pretrained_dir, 'model.ckpt')

    def _set_pretrained_model(self):
        if self.model_type == 'bert':
            self.Config = BertConfig
            self.PretrainedModel = BertModel
        elif self.model_type == 'albert':
            self.Config = AlbertConfig
            self.PretrainedModel = AlbertModel
        else:
            self.Config = None
            self.PretrainedModel = None
            raise ValueError('model_type %s not support.'
                             % self.model_type)
        self.pretrained_config = self.Config.from_json_file(self.config_file)

    def _build_pretrained(self, is_training, input_ids_1, input_mask_1, segment_ids_1,
                          input_ids_2, input_mask_2, segment_ids_2,
                          use_one_hot_embeddings=False, use_einsum=True):
        if self.model_type == 'bert':
            scope = 'bert'
        elif self.model_type == 'albert':
            scope = 'bert'

        self.pretrained_model_1 = self.PretrainedModel(
            config=self.pretrained_config,
            is_training=is_training,
            input_ids=input_ids_1,
            input_mask=input_mask_1,
            token_type_ids=segment_ids_1,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum,
            scope=scope)
        self.pretrained_model_2 = self.PretrainedModel(
            config=self.pretrained_config,
            is_training=is_training,
            input_ids=input_ids_2,
            input_mask=input_mask_2,
            token_type_ids=segment_ids_2,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum,
            scope=scope)

        pooled_output_1 = self.pretrained_model_1.get_pooled_output()
        pooled_output_2 = self.pretrained_model_2.get_pooled_output()
        diff = tf.abs(pooled_output_1 - pooled_output_2)

        # method 1
        cos_sim = tf.math.reduce_sum(
            tf.math.multiply(pooled_output_1, pooled_output_2), axis=1, keepdims=True)

        # method 2
        pooled_output = tf.concat([pooled_output_1, pooled_output_2, diff], axis=1)

        return pooled_output, pooled_output_1, pooled_output_2, cos_sim

    @staticmethod
    def _build_downstream(is_training, pooled_output, labels, num_labels):
        """Fully connected layer for text_matching."""
        hidden_size = pooled_output.shape[-1].value
        output_weights = tf.get_variable(
            'output_weights', [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            'output_bias', [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope('loss'):
            if is_training:
                # I.e., 0.1 dropout
                pooled_output = dropout(pooled_output, dropout_prob=0.1)

            logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.softmax(logits, axis=-1)
            predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            batch_loss = tf.reduce_mean(per_example_loss)

            return batch_loss, per_example_loss, probabilities, logits, predictions

    def __call__(self, is_training, input_ids_1, input_mask_1, segment_ids_1,
                 input_ids_2, input_mask_2, segment_ids_2, labels, num_labels,
                 use_one_hot_embeddings=False, use_einsum=True):
        pooled_output, pooled_output_1, pooled_output_2, cos_sim = self._build_pretrained(
            is_training, input_ids_1, input_mask_1, segment_ids_1,
            input_ids_2, input_mask_2, segment_ids_2,
            use_one_hot_embeddings, use_einsum)
        batch_loss, per_example_loss, probabilities, logits, predictions = self._build_downstream(
            is_training, pooled_output, labels, num_labels)
        return batch_loss, per_example_loss, probabilities, logits, \
               predictions, pooled_output_1, pooled_output_2, cos_sim
