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

from pyclue.tf1.nlu.from_pretrained.albert import AlbertConfig
from pyclue.tf1.nlu.from_pretrained.albert import AlbertModel
from pyclue.tf1.nlu.from_pretrained.bert import BertConfig
from pyclue.tf1.nlu.from_pretrained.bert import BertModel
from pyclue.tf1.nlu.from_pretrained.warehouse.information import pretrained_names
from pyclue.tf1.nlu.from_pretrained.warehouse.information import pretrained_type
from pyclue.tf1.nlu.from_pretrained.warehouse.prepare import get_pretrained_model
from pyclue.tf1.nlu.modeling.layers import dropout


class TraditionalSiameseModel(object):
    # TODO
    def __init__(self):
        pass


class PretrainedSiameseModel(object):

    def __init__(self, nlu_name=None, nlu_model_type=None,
                 nlu_vocab_file=None, nlu_config_file=None,
                 nlu_init_checkpoint_file=None, downstream_name=None):
        self.nlu_name = nlu_name
        if not nlu_name:
            assert all([nlu_model_type,
                        nlu_vocab_file,
                        nlu_config_file,
                        nlu_init_checkpoint_file]), \
                'if not given nlu_name the warehouse provided, ' \
                'you should specify all details of nlu model'
            self.nlu_model_type = nlu_model_type.lower()
            self.nlu_vocab_file = nlu_vocab_file
            self.nlu_config_file = nlu_config_file
            self.nlu_init_checkpoint_file = nlu_init_checkpoint_file
        else:
            assert nlu_name in pretrained_names, \
                '%s not provided by warehouse' % nlu_name
            self.nlu_model_type = pretrained_type.get(nlu_name).split('_')[0]
            self._from_pretrained_warehouse()
        self._check_pretrained_model()

        if downstream_name:
            self.downstream_name = downstream_name.lower()
        else:
            self.downstream_name = 'dense'

    def _from_pretrained_warehouse(self):
        pretrained_dir = get_pretrained_model(pretrained_name=self.nlu_name)
        self.nlu_vocab_file = os.path.join(pretrained_dir, 'vocab.txt')
        self.nlu_config_file = os.path.join(pretrained_dir, 'config.json')
        self.nlu_init_checkpoint_file = os.path.join(pretrained_dir, 'model.ckpt')

    def _check_pretrained_model(self):
        if self.nlu_model_type == 'bert':
            self.Config = BertConfig
            self.PretrainedModel = BertModel
        elif self.nlu_model_type == 'albert':
            self.Config = AlbertConfig
            self.PretrainedModel = AlbertModel
        else:
            self.Config = None
            self.PretrainedModel = None
            raise ValueError('nlu_model_type not support: %s'
                             % self.nlu_model_type)
        self.nlu_config = self.Config.from_json_file(self.nlu_config_file)

    def _build_pretrained(self, is_training,
                          input_ids_1, input_mask_1, segment_ids_1,
                          input_ids_2, input_mask_2, segment_ids_2,
                          use_one_hot_embeddings=False, use_einsum=True,
                          scope='sentence_pair'):
        self.nlu_model_1 = self.PretrainedModel(
            config=self.nlu_config,
            is_training=is_training,
            input_ids=input_ids_1,
            input_mask=input_mask_1,
            token_type_ids=segment_ids_1,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum,
            scope=scope)
        self.nlu_model_2 = self.PretrainedModel(
            config=self.nlu_config,
            is_training=is_training,
            input_ids=input_ids_2,
            input_mask=input_mask_2,
            token_type_ids=segment_ids_2,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum,
            scope=scope)

        pooled_output_1 = self.nlu_model_1.get_pooled_output()
        pooled_output_2 = self.nlu_model_2.get_pooled_output()
        diff = tf.abs(pooled_output_1 - pooled_output_2)

        # method 1
        cos_sim = tf.math.reduce_sum(
            tf.math.multiply(pooled_output_1, pooled_output_2), axis=1, keepdims=True)

        # method 2
        pooled_output = tf.concat([pooled_output_1, pooled_output_2, diff], axis=1)

        return pooled_output, pooled_output_1, pooled_output_2, cos_sim

    def _build_downstream(self, is_training, pooled_output, labels, num_labels):
        if self.downstream_name == 'dense':
            return siamese_dense(is_training, pooled_output, labels, num_labels)
        # TODO 增加多类下游模型
        else:
            raise ValueError('downstream_name not support.')

    def __call__(self, is_training, input_ids_1, input_mask_1, segment_ids_1,
                 input_ids_2, input_mask_2, segment_ids_2, labels, num_labels,
                 use_one_hot_embeddings=False, use_einsum=True, scope='sentence_pair'):
        pooled_output, pooled_output_1, pooled_output_2, cos_sim = self._build_pretrained(
            is_training, input_ids_1, input_mask_1, segment_ids_1,
            input_ids_2, input_mask_2, segment_ids_2,
            use_one_hot_embeddings, use_einsum, scope)
        batch_loss, per_example_loss, probabilities, logits, predictions = self._build_downstream(
            is_training, pooled_output, labels, num_labels)
        return batch_loss, per_example_loss, probabilities, logits, \
               predictions, pooled_output_1, pooled_output_2, cos_sim


def siamese_dense(is_training, pooled_output, labels, num_labels):
    """Fully connected layer for sentence_pair."""

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