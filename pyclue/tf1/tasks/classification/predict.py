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
from pyclue.tf1.nlu.from_pretrained.warehouse.prepare import get_pretrained_model
from pyclue.tf1.nlu.from_pretrained.warehouse.information import pretrained_names
from pyclue.tf1.nlu.modeling.utils.tokenization_utils import FullTokenizer
from pyclue.tf1.tasks.classification.inputs import ClassifierProcessorBase


class ClassifierPredictor(object):

    def __init__(self, model_file):
        self.model_file = os.path.abspath(model_file)
        label_map_reverse_file = os.path.join(
            self.model_file, 'label_map_reverse.json')
        with tf.gfile.GFile(label_map_reverse_file, 'r') as f:
            self.label_map_reverse = json.load(f)
        self.labels = [item[1] for item in sorted(
            self.label_map_reverse.items(), key=lambda i: i[0])]

    def build_model(self, nlu_name=None, from_pretrained=False, vocab_file=None, max_seq_len=512):
        self.nlu_name = nlu_name
        self.from_pretrained = from_pretrained
        if self.from_pretrained:
            if not self.nlu_name:
                assert vocab_file, \
                    'if is from_pretrained and not given pretrained_name ' \
                    'the warehouse provided, you should specify the vocab_file'
                self.nlu_vocab_file = vocab_file
            else:
                assert self.nlu_name in pretrained_names, \
                    '%s not provided by warehouse' % self.nlu_name
                pretrained_dir = get_pretrained_model(pretrained_name=self.nlu_name)
                self.nlu_vocab_file = os.path.join(pretrained_dir, 'vocab.txt')
        else:
            self.nlu_vocab_file = vocab_file

        self.max_seq_len = max_seq_len

        self._load_tokenizer()
        self._load_processor()
        self._build()

    def _load_tokenizer(self):
        if self.from_pretrained:
            self.tokenizer = FullTokenizer(self.nlu_vocab_file)
        else:
            pass  # TODO: add traditional tokenizer

    def _load_processor(self):
        self.processor = ClassifierProcessorBase(
            max_seq_len=self.max_seq_len, tokenizer=self.tokenizer, labels=self.labels)

    def _build(self):
        self.graph = tf.Graph()
        self.sess = tf.Session()
        self.meta_graph_def = tf.saved_model.loader.load(
            self.sess, tags=['serve'], export_dir=self.model_file)
        self.signature = self.meta_graph_def.signature_def
        self.input_ids = self.signature['serving_default'].inputs['input_ids'].name
        self.input_mask = self.signature['serving_default'].inputs['input_mask'].name
        self.segment_ids = self.signature['serving_default'].inputs['segment_ids'].name
        self.label_ids = self.signature['serving_default'].inputs['label_ids'].name
        self.predictions = self.signature['serving_default'].outputs['predictions'].name
        self.probabilities = self.signature['serving_default'].outputs['probabilities'].name

    def _predict_for_single_example(self, feature):
        prediction, probability = self.sess.run(
            [self.predictions, self.probabilities],
            feed_dict={
                self.input_ids: [feature.input_ids],
                self.input_mask: [feature.input_mask],
                self.segment_ids: [feature.segment_ids],
                self.label_ids: [feature.label_id]})
        return prediction, probability

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        features = self.processor.get_features_for_inputs(texts)

        results = []
        for text, feature in zip(texts, features):
            prediction, probability = self._predict_for_single_example(feature)
            results.append({
                'text': text,
                'prediction': self.label_map_reverse[str(np.squeeze(prediction).tolist())],
                'probability': np.squeeze(probability).tolist()})
        return results

    def predict_from_file(self, input_file):

        texts = self.processor._read_file(input_file)
        texts = np.squeeze(texts).tolist()

        return self.predict(texts)

    def quality_inspection(self, input_file, save_path):

        texts = self.processor._read_file(input_file)

        features = self.processor.get_features_for_inputs(texts)

        predictions, probabilities = [], []

        for feature in features:
            prediction, probability = self._predict_for_single_example(feature)
            predictions.append(prediction)
            probabilities.append(probability.tolist())

        if not tf.gfile.Exists(save_path):
            tf.gfile.MakeDirs(save_path)

        with tf.gfile.GFile(os.path.join(save_path, input_file.split('/')[-1]), 'w') as writer:
            for text, prediction, probability in zip(texts, predictions, probabilities):
                prediction = self.label_map_reverse[str(np.squeeze(prediction).tolist())]
                if text[0] != prediction:
                    writer.write(
                        'text = %s, true = %s, pred = %s, probability = %s\n'
                        % (text[1], text[0], prediction, probability))

    def close(self):
        self.sess.close()

    def restart(self):
        self._build()
