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

import hnswlib
import numpy as np
import tensorflow as tf

from collections import Counter
from pyclue.tf1.open_sources.configs import pretrained_names, pretrained_types
from pyclue.tf1.open_sources.download import get_pretrained_model
from pyclue.tf1.tasks.text_matching.siamese.inputs import Processor
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

        # load cache
        self.cache_file = os.path.join(
            self.model_file, 'cache.txt')
        self._load_cache()

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

    def _load_cache(self):
        self.ef = self.model_file.get('ef')
        self.cache_texts, self.cache_embeddings, self.cache_labels = self.get_embedding_from_file(self.cache_file)
        self.num_cache, self.embedding_dim = self.cache_embeddings.shape

        # application of hnswlib
        # declaring index
        self.index_nms = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        # initializing index - the maximum number of elements should be know beforehand
        self.index_nms.init_index(max_elements=self.num_cache, ef_construction=100, M=100)
        # element insertion (can be called several times)
        self.index_nms.add_items(data=self.cache_embeddings, ids=range(self.num_cache))
        self.index_nms.save_index(os.path.join(self.pb_model_file, 'cache.index'))
        # controlling the recall by setting ef:
        self.index_nms.set_ef(ef=self.ef)  # ef should always be > k (knn)

    @staticmethod
    def _apply_top_k_strategy(similarities, labels, texts, strategy=None):
        if strategy == 'weighted_max':
            new_similarities, new_labels, new_texts = [], [], []
            for i in range(len(similarities)):
                counter = Counter()
                for similarity, label, text in zip(similarities[i], labels[i], texts[i]):
                    counter[label] += similarity / len(similarities[i])
                reversed_counter = {v: k for k, v in counter.items()}
                max_similarity = max(reversed_counter)
                max_label = reversed_counter[max_similarity]
                max_text = texts[i][labels[i].index(max_label)]
                new_similarities.append([max_similarity])
                new_labels.append([max_label])
                new_texts.append([max_text])
            return new_similarities, new_labels, new_texts
        elif strategy == 'count_max':
            new_similarities, new_labels, new_texts = [], [], []
            for i in range(len(similarities)):
                counter = Counter()
                for similarity, label, text in zip(similarities[i], labels[i], texts[i]):
                    counter[label] += 1
                reversed_counter = {v: k for k, v in counter.items()}
                max_similarity = max(reversed_counter)
                max_label = reversed_counter[max_similarity]
                max_text = texts[i][labels[i].index(max_label)]
                new_similarities.append([max_similarity])
                new_labels.append([max_label])
                new_texts.append([max_text])
            return new_similarities, new_labels, new_texts
        elif strategy == 'reduce_max':
            new_similarities = [[item[0]] for item in similarities]
            new_labels = [[item[0]] for item in labels]
            new_texts = [[item[0]] for item in texts]
            return new_similarities, new_labels, new_texts
        else:
            return similarities, labels, texts

    def _get_embedding_for_single_example(self, feature):
        text_a_embedding = self.sess.run(
            self.text_a_embedding,
            feed_dict={
                self.input_ids_1: [feature.input_ids_1],
                self.input_mask_1: [feature.input_mask_1],
                self.segment_ids_1: [feature.segment_ids_1],
                self.input_ids_2: [feature.input_ids_2],
                self.input_mask_2: [feature.input_mask_2],
                self.segment_ids_2: [feature.segment_ids_2],
                self.label_ids: [feature.label_id]})
        text_a_embedding = np.squeeze(text_a_embedding)
        return text_a_embedding

    def get_embedding(self, texts):
        if isinstance(texts, str):
            new_texts = [self.labels[0], texts, '']
        elif isinstance(texts, list):
            new_texts = []
            for item in texts:
                if len(item) == 1 or len(item) == 2:
                    new_texts.append([self.labels[0], item[-1], ''])
                else:
                    raise ValueError('texts item should contain 1 or 2 elements')
            assert all([len(item) == 3 for item in new_texts]), \
                'texts item should contain 3 elements'
        else:
            raise ValueError('texts format not support')
        features = self.processor.get_features_for_inputs(new_texts)

        embeddings = []
        for text, feature in zip(new_texts, features):
            embedding = self._get_embedding_for_single_example(feature)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_embedding_from_file(self, input_file):
        inputs = np.squeeze(self.processor.read_file(input_file)).tolist()
        inputs_ndim = np.squeeze(inputs).ndim
        if inputs_ndim == 1:
            labels = [-1] * len(inputs)
            texts = inputs
        elif inputs_ndim == 2:
            labels = [item[0] for item in inputs]
            texts = [item[1] for item in inputs]
        else:
            raise TypeError('input_file format should be `label\\ttext` or `text`')
        texts = np.array(texts)
        embeddings = self.get_embedding(texts)
        labels = np.array(labels)

        return texts, embeddings, labels

    def predict(self, texts, top_k=3, strategy=None):
        assert top_k < self.ef, 'Parameter top_k should be less than self.ef.'

        embeddings = self.get_embedding(texts)
        # query dataset, k - number of closest elements (returns 2 numpy arrays)
        indexes, distances = self.index_nms.knn_query(embeddings, k=top_k)
        similar_labels = [self.cache_labels[item] for item in indexes]
        similar_texts = [self.cache_texts[item] for item in indexes]
        scaled_similarities = 1 - np.squeeze(distances) / 2
        scaled_similarities, similar_labels, similar_texts = self._apply_top_k_strategy(
            scaled_similarities, similar_labels, similar_texts, strategy)

        results = []
        for i, text in enumerate(texts):
            result = {'text': text, 'rank': {}}
            for j, scaled_similarity, similar_label, similar_text \
                    in zip(range(1, len(similar_texts[i] + 1)),
                           scaled_similarities,
                           similar_labels,
                           similar_texts):
                result['rank'][str(j)] = {
                    'similar_text': similar_text,
                    'similar_label': similar_label,
                    'cos_sim': scaled_similarity}
            results.append(result)
        return results

    def predict_from_file(self, input_file, top_k=3, strategy=None):
        assert top_k < self.ef, 'Parameter top_k should be less than self.ef.'

        texts, embeddings, labels = self.get_embedding_from_file(input_file)
        # query dataset, k - number of closest elements (returns 2 numpy arrays)
        indexes, distances = self.index_nms.knn_query(embeddings, k=top_k)
        similar_labels = [self.cache_labels[item] for item in indexes]
        similar_texts = [self.cache_texts[item] for item in indexes]
        scaled_similarities = 1 - np.squeeze(distances) / 2
        scaled_similarities, similar_labels, similar_texts = self._apply_top_k_strategy(
            scaled_similarities, similar_labels, similar_texts, strategy)

        results = []
        for i, text in enumerate(texts):
            result = {'text': text, 'rank': {}}
            for j, scaled_similarity, similar_label, similar_text \
                    in zip(range(1, len(similar_texts[i] + 1)),
                           scaled_similarities,
                           similar_labels,
                           similar_texts):
                result['rank'][str(j)] = {
                    'similar_text': similar_text,
                    'similar_label': similar_label,
                    'cos_sim': scaled_similarity}
            results.append(result)
        return results

    def quality_inspection(self, input_file, top_k=3, strategy='reduce_max',
                           threshold=0.75, save_path=None):
        assert strategy, 'Parameter strategy should not be `None` during `quality_inspection`.'
        assert top_k < self.ef, 'Parameter top_k should be less than self.ef.'

        texts, embeddings, labels = self.get_embedding_from_file(input_file)
        # query dataset, k - number of closest elements (returns 2 numpy arrays)
        indexes, distances = self.index_nms.knn_query(embeddings, k=top_k)
        similar_labels = [self.cache_labels[item] for item in indexes]
        similar_texts = [self.cache_texts[item] for item in indexes]
        scaled_similarities = 1 - np.squeeze(distances) / 2
        scaled_similarities, similar_labels, similar_texts = self._apply_top_k_strategy(
            scaled_similarities, similar_labels, similar_texts, strategy)

        with open(save_path, 'w') as f:
            for i, text, label in zip(range(len(texts)), texts, labels):
                if label not in set(self.cache_labels):
                    if scaled_similarities[i][0] >= threshold:
                        f.write('text = %s, label = %s, surpass threshold %.6f\n'
                                % (text, label, threshold))
                else:
                    if scaled_similarities[i][0] < threshold:
                        f.write('text = %s, label = %s, less than threshold %.6f\n'
                                % (text, label, threshold))
                    elif similar_labels[i] != label:
                        f.write('text = %s, label = %s, similar_text = %s, '
                                'similar_label = %s, scaled_similarity = %.6f\n'
                                % (text, label, similar_texts[i], similar_labels[i], scaled_similarities[i]))

    def close(self):
        self.sess.close()

    def restart(self):
        self._build()
