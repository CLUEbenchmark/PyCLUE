#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import numpy as np
import tensorflow as tf

from pyclue.tf1.open_sources.configs import pretrained_names, pretrained_types
from pyclue.tf1.open_sources.download import get_pretrained_model
from pyclue.tf1.tasks.sentence_pair.siamese.inputs import Processor
from pyclue.tf1.tokenizers.bert_tokenizer import FullTokenizer  # Add more tokenizers


class Predictor(object):

    def __init__(self, model_file):
        self.model_file = os.path.abspath(model_file)

        # label
        label_map_reverse_file = os.path.join(
            self.model_file, 'label_map_reverse.json')
        with tf.gfile.GFile(label_map_reverse_file, 'r') as f:
            self.label_map_reverse = json.load(f)
        self.labels = [item[1] for item in sorted(
            self.label_map_reverse.items(), key=lambda i: i[0])]

        # model
        model_config_file = os.path.join(
            self.model_file, 'model_config.json')
        with tf.gfile.GFile(model_config_file, 'r') as f:
            self.model_config = json.load(f)
        self.model_name = self.model_config.get('model_name') or None
        self.model_type = self.model_config.get('model_type') or None
        self.vocab_file = self.model_config.get('vocab_file') or None
        self.max_seq_len = self.model_config.get('max_seq_len') or 512
        if not self.model_name:
            assert all([self.vocab_file, self.model_type]), \
                'If not given model_name provided by open_sources, ' \
                'you should specify the model_type and vocab_file.'
        else:
            assert self.model_name in pretrained_names, \
                '%s not provided by open_sources' % self.model_name
            self.model_type = pretrained_types.get(self.model_name).split('_')[0]
            pretrained_dir = get_pretrained_model(pretrained_name=self.model_name)
            self.vocab_file = os.path.join(pretrained_dir, 'vocab.txt')

        # tokenizer
        if self.model_type == 'bert':
            self.tokenizer = FullTokenizer(self.vocab_file)
        elif self.model_type == 'albert':
            self.tokenizer = FullTokenizer(self.vocab_file)
        else:
            raise ValueError('model_type %s unknown.' % self.model_type)

        # processor
        self._load_processor()

        # build graph
        self._build()

    def _load_processor(self):
        self.processor = Processor(
            max_seq_len=self.max_seq_len, tokenizer=self.tokenizer, labels=self.labels)

    def _build(self):
        self.graph = tf.Graph()
        self.sess = tf.Session()
        self.meta_graph_def = tf.saved_model.loader.load(
            self.sess, tags=['serve'], export_dir=self.model_file)
        self.signature = self.meta_graph_def.signature_def
        self.input_ids_1 = self.signature['serving_default'].inputs['input_ids_1'].name
        self.input_mask_1 = self.signature['serving_default'].inputs['input_mask_1'].name
        self.segment_ids_1 = self.signature['serving_default'].inputs['segment_ids_1'].name
        self.input_ids_2 = self.signature['serving_default'].inputs['input_ids_2'].name
        self.input_mask_2 = self.signature['serving_default'].inputs['input_mask_2'].name
        self.segment_ids_2 = self.signature['serving_default'].inputs['segment_ids_2'].name
        self.label_ids = self.signature['serving_default'].inputs['label_ids'].name
        self.text_a_embedding = self.signature['serving_default'].outputs['text_a_embedding'].name
        self.text_b_embedding = self.signature['serving_default'].outputs['text_b_embedding'].name
        self.cos_sims = self.signature['serving_default'].outputs['cos_sims'].name
        self.predictions = self.signature['serving_default'].outputs['predictions'].name
        self.probabilities = self.signature['serving_default'].outputs['probabilities'].name

    def _predict_for_single_example(self, feature):
        cos_sim, prediction, probability = self.sess.run(
            [self.cos_sims, self.predictions, self.probabilities],
            feed_dict={
                self.input_ids_1: [feature.input_ids_1],
                self.input_mask_1: [feature.input_mask_1],
                self.segment_ids_1: [feature.segment_ids_1],
                self.input_ids_2: [feature.input_ids_2],
                self.input_mask_2: [feature.input_mask_2],
                self.segment_ids_2: [feature.segment_ids_2],
                self.label_ids: [feature.label_id]})
        return cos_sim, prediction, probability

    def predict(self, texts):
        assert isinstance(texts, list), 'texts format should be `list`'
        assert all([isinstance(item, list) for item in texts]), 'texts item format should be `list`'
        new_texts = []
        for item in texts:
            if len(item) == 2 or len(item) == 3:
                new_texts.append([self.labels[0], item[-2], item[-1]])
            else:
                raise ValueError('text item should contain 2 or 3 elements')
        assert all([len(item) == 3 for item in new_texts]), \
            'texts item should contain 3 elements'

        features = self.processor.get_features_for_inputs(new_texts)
        results = []
        for text, feature in zip(new_texts, features):
            cos_sim, prediction, probability = self._predict_for_single_example(feature)
            results.append({
                'text_a': text[1],
                'text_b': text[2],
                'cos_sim': np.squeeze(cos_sim).tolist() / 100,
                'prediction': self.label_map_reverse[str(np.squeeze(prediction).tolist())],
                'probability': np.squeeze(probability).tolist()})
        return results

    def predict_from_file(self, input_file):
        texts = self.processor.read_file(input_file)
        texts = np.squeeze(texts).tolist()
        return self.predict(texts)

    def quality_inspection(self, input_file, save_path):
        texts = self.processor.read_file(input_file)
        if np.array(texts).ndim == 1:
            texts = [texts]
        texts = [item for item in texts if len(item) == 3]

        features = self.processor.get_features_for_inputs(texts)
        cos_sims, predictions, probabilities = [], [], []
        for feature in features:
            cos_sim, prediction, probability = self._predict_for_single_example(feature)
            cos_sims.append(cos_sim)
            predictions.append(prediction)
            probabilities.append(probability.tolist())

        if not tf.gfile.Exists(save_path):
            tf.gfile.MakeDirs(save_path)
        with tf.gfile.GFile(os.path.join(save_path, input_file.split('/')[-1]), 'w') as writer:
            for text, prediction, probability in zip(texts, predictions, probabilities):
                prediction = self.label_map_reverse[str(np.squeeze(prediction).tolist())]
                if text[0] != prediction:
                    writer.write(
                        'text_a = %s, text_b = %s, '
                        'true = %s, pred = %s, '
                        'probability = %s, cos_sim = %s\n'
                        % (text[1], text[2], text[0], prediction, probability, cos_sim / 100))

    def close(self):
        self.sess.close()

    def restart(self):
        self._build()
