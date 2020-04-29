#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
from functools import reduce

import tensorflow as tf

vars_naming_format = {
    'albert_brightmart': {
        'rename_vars': [
            {
                'new_var_name': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self',
                'origin_var_name': 'bert/encoder/layer_shared/attention/self'
            },
            {
                'new_var_name': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense',
                'origin_var_name': 'bert/encoder/layer_shared/attention/output/dense'
            },
            {
                'new_var_name': 'bert/encoder/transformer/group_0/inner_group_0/LayerNorm',
                'origin_var_name': 'bert/encoder/layer_shared/attention/output/LayerNorm'
            },
            {
                'new_var_name': 'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense',
                'origin_var_name': 'bert/encoder/layer_shared/intermediate/dense'
            },
            {
                'new_var_name': 'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense',
                'origin_var_name': 'bert/encoder/layer_shared/output/dense'
            },
            {
                'new_var_name': 'bert/encoder/layer_shared/output/LayerNorm',
                'origin_var_name': 'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1'
            }
        ],
        'keep_vars': [
            'bert/embeddings/LayerNorm/beta',
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/position_embeddings',
            'bert/embeddings/token_type_embeddings',
            'bert/pooler/dense/bias',
            'bert/pooler/dense/kernel',
            'cls/predictions/output_bias',
            'cls/predictions/transform/LayerNorm/beta',
            'cls/predictions/transform/LayerNorm/gamma',
            'cls/predictions/transform/dense/bias',
            'cls/predictions/transform/dense/kernel',
            'cls/seq_relationship/output_bias',
            'cls/seq_relationship/output_weights'
        ],
        'create_vars': [
            {
                'new_var_name': 'bert/embeddings/word_embeddings',
                'origin_var_names': ['bert/embeddings/word_embeddings',
                                     'bert/embeddings/word_embeddings_2'],
                'op': tf.matmul
            }
        ]
    },
    'bert': {
        'rename_vars': [
            {
                'new_var_name': 'bert/encoder/transformer',
                'origin_var_name': 'bert/encoder'
            },
        ],
        'keep_vars': [
            'bert/embeddings/LayerNorm/beta',
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/position_embeddings',
            'bert/embeddings/token_type_embeddings',
            'bert/embeddings/word_embeddings',
            'bert/pooler/dense/bias',
            'bert/pooler/dense/kernel',
            'cls/predictions/output_bias',
            'cls/predictions/transform/LayerNorm/beta',
            'cls/predictions/transform/LayerNorm/gamma',
            'cls/predictions/transform/dense/bias',
            'cls/predictions/transform/dense/kernel',
            'cls/seq_relationship/output_bias',
            'cls/seq_relationship/output_weights',
        ],
        'create_vars': []
    }
}

configs_naming_format = {
    'albert_brightmart': {
        'attention_probs_dropout_prob': 0.0,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0,
        'embedding_size': 128,
        'hidden_size': 768,
        'initializer_range': 0.02,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'num_hidden_groups': 1,
        'net_structure_type': 0,
        'layers_to_keep': [],
        'gap_size': 0,
        'num_memory_blocks': 0,
        'inner_group_num': 1,
        'down_scale_factor': 1,
        'type_vocab_size': 2,
        'vocab_size': 21128
    }
}


class Converter(object):

    def __init__(self, model_dir, model_type):
        self.checkpoint_file = os.path.abspath(os.path.join(model_dir, 'model.ckpt'))
        self.config_file = os.path.abspath(os.path.join(model_dir, 'config.json'))
        self.is_convert = any([model_type in to_convert_item for to_convert_item in
                               [vars_naming_format, configs_naming_format]])
        self.vars_naming_format = copy.deepcopy(vars_naming_format)
        self.vars_info = self.vars_naming_format.get(model_type)
        self.configs_naming_format = copy.deepcopy(configs_naming_format)
        self.configs_info = self.configs_naming_format.get(model_type)

    def _regularize_checkpoint(self):
        with tf.Graph().as_default():
            tvar_names = [item[0] for item in tf.train.list_variables(self.checkpoint_file)]

            # rename_vars
            for rename_var in self.vars_info.get('rename_vars'):
                origin_var_name_pattern = rename_var.get('origin_var_name')
                new_var_name_pattern = rename_var.get('new_var_name')
                for tvar_name in tvar_names:
                    if origin_var_name_pattern in tvar_name:
                        new_var_name = tvar_name.replace(origin_var_name_pattern, new_var_name_pattern)
                        weight = self._load_var(tvar_name)
                        self._create_var(new_var_name, weight)

            # keep_vars
            for keep_var in self.vars_info.get('keep_vars'):
                weight = self._load_var(keep_var)
                self._create_var(keep_var, weight)

            # create_vars
            for create_var in self.vars_info.get('create_vars'):
                new_var_name = create_var.get('new_var_name')
                origin_var_names = create_var.get('origin_var_names')
                op = create_var.get('op')

                checked_origin_var_names = []
                if not isinstance(origin_var_names, list):
                    origin_var_names = [origin_var_names]
                for origin_var_name in origin_var_names:
                    checked_origin_var_name = None
                    for tvar_name in tvar_names:
                        if origin_var_name == tvar_name:
                            checked_origin_var_name = origin_var_name
                    checked_origin_var_names.append(checked_origin_var_name)
                if all(checked_origin_var_names):
                    origin_vars = [origin_var for origin_var in map(self._load_var, checked_origin_var_names)]
                    weight = reduce(op, origin_vars)

                self._create_var(new_var_name, weight)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, self.checkpoint_file, write_meta_graph=False)

    def _load_var(self, var_name):
        var = tf.train.load_variable(self.checkpoint_file, var_name)
        return var

    def _create_var(self, var_name, weight):
        return tf.Variable(weight, name=var_name)

    def _regularize_config(self):
        if self.configs_info:
            with tf.gfile.Open(self.config_file, 'r') as f:
                configs = json.load(f)
            for key, value in configs.items():
                if self.configs_info.get(key):
                    self.configs_info[key] = value
            with tf.gfile.Open(self.config_file, 'w') as f:
                json.dump(self.configs_info, f, ensure_ascii=False, indent=4)

    def run(self):
        if self.is_convert:
            self._regularize_checkpoint()
            self._regularize_config()
