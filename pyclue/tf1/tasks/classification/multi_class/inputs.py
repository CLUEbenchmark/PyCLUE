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
import numpy as np
import collections

from pyclue.tf1.tokenizers.utils import convert_to_unicode


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class Processor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, max_seq_len, tokenizer, labels=None, task_type='single'):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.labels = labels
        self.task_type = task_type
        self._get_labels()

    def _get_labels(self):
        """Gets the list of labels."""
        if not self.labels:
            self.labels = ['pseudo_label']
        assert isinstance(self.labels, list), 'labels should be `list` instance.'
        self.num_labels = len(self.labels)
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.label_map_reverse = {i: label for i, label in enumerate(self.labels)}

    def _create_examples(self, lines, set_type):
        """Creates examples."""
        examples = []
        print('# {} data: {}'.format(set_type, len(lines)))
        for i, line in enumerate(lines):
            origin_line = ' '.join(line)
            guid = '{}-{}'.format(set_type, i)
            try:
                label = convert_to_unicode(line[0])
                label = label.replace('"', '').replace('\\', '')
                text_a = convert_to_unicode(line[1])
                if self.task_type == 'single':
                    text_b = ''
                else:
                    text_b = convert_to_unicode(line[2])
                if label in self.labels or set_type == 'predict':
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except Exception as e:
                print(e)
                print('### {}-example error {}: {}'.format(set_type, i, origin_line))
        return examples

    def _get_feature_for_example(self, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * self.max_seq_len,
                input_mask=[0] * self.max_seq_len,
                segment_ids=[0] * self.max_seq_len,
                label_id=0,
                is_real_example=False)

        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_len - 2:
                tokens_a = tokens_a[:self.max_seq_len - 2]

        # The convention in BERT for single sequences is:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        tokens.extend(tokens_a)
        segment_ids.extend([0] * len(tokens_a))
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            tokens.extend(tokens_b)
            segment_ids.extend([1] * len(tokens_b))
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        label_id = self.label_map[example.label]
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def get_features_for_inputs(self, lines, set_type='predict'):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        examples = self._create_examples(lines, set_type)
        features = []
        for example in examples:
            feature = self._get_feature_for_example(example=example)
            features.append(feature)
        return features

    @staticmethod
    def read_file(file_path):
        """Read files."""
        with tf.gfile.GFile(file_path, 'r') as f:
            data = f.readlines()
            lines = []
            for line in data:
                lines.append(line.strip().split('\t'))
        return lines


class FileProcessor(Processor):
    """Data converters for sequence classification data with file inputs."""

    def __init__(self, max_seq_len, tokenizer, data_dir, task_type='single',
                 save_tfrecord_dir=None, recreate_tfrecord=True):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_dir = os.path.abspath(data_dir)
        self.task_type = task_type
        self.save_tfrecord_dir = os.path.abspath(save_tfrecord_dir) if save_tfrecord_dir else None
        self.recreate_tfrecord = recreate_tfrecord
        self._get_labels()
        self._get_data()

    def _get_data(self):
        """Gets a collection of `InputExample`s."""
        # train
        train_example_path = os.path.join(self.data_dir, 'train.txt')
        train_example_path_tfrecord = None if not self.save_tfrecord_dir \
            else os.path.join(self.save_tfrecord_dir, 'train.tfrecord')
        self.train_examples = self._create_examples(
            self.read_file(train_example_path), 'train')
        self.num_train_examples = len(self.train_examples)
        if self.recreate_tfrecord and train_example_path_tfrecord and self.num_train_examples > 0:
            self._save_tfrecords(
                examples=self.train_examples, output_file=train_example_path_tfrecord)

        # dev
        dev_example_path = os.path.join(self.data_dir, 'dev.txt')
        dev_example_path_tfrecord = None if not self.save_tfrecord_dir \
            else os.path.join(self.save_tfrecord_dir, 'dev.tfrecord')
        self.dev_examples = self._create_examples(
            self.read_file(dev_example_path), 'dev')
        self.num_dev_examples = len(self.dev_examples)
        if self.recreate_tfrecord and dev_example_path_tfrecord and self.num_dev_examples > 0:
            self._save_tfrecords(
                examples=self.dev_examples, output_file=dev_example_path_tfrecord)

        # test
        test_example_path = os.path.join(self.data_dir, 'test.txt')
        test_example_path_tfrecord = None if not self.save_tfrecord_dir \
            else os.path.join(self.save_tfrecord_dir, 'test.tfrecord')
        if tf.gfile.Exists(test_example_path):
            self.test_examples = self._create_examples(
                self.read_file(test_example_path), 'test')
            self.num_test_examples = len(self.test_examples)
            if self.recreate_tfrecord and test_example_path_tfrecord and self.num_test_examples > 0:
                self._save_tfrecords(
                    examples=self.test_examples, output_file=test_example_path_tfrecord)
        else:
            self.test_examples = None
            self.num_test_examples = 0

    def _get_labels(self):
        """Gets the list of labels."""
        self.labels = []
        lines = self.read_file(
            os.path.join(self.data_dir, 'labels.txt'))
        for line in lines:
            self.labels.append(line[0])
        self.num_labels = len(self.labels)
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.label_map_reverse = {i: label for i, label in enumerate(self.labels)}

    def _save_tfrecords(self, examples, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        writer = tf.python_io.TFRecordWriter(output_file)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        for example in examples:
            feature = self._get_feature_for_example(example)
            features = collections.OrderedDict()
            features['input_ids'] = create_int_feature(feature.input_ids)
            features['input_mask'] = create_int_feature(feature.input_mask)
            features['segment_ids'] = create_int_feature(feature.segment_ids)
            features['label_ids'] = create_int_feature([feature.label_id])
            features['is_real_example'] = create_int_feature(
                [int(feature.is_real_example)])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


class InputFn(object):
    """Data converters for sequence classification data sets."""

    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def predict_input_fn(self, features):
        """Creates an `input_fn` closure to be passed to Estimator."""
        x = {
            'input_ids': [],
            'input_mask': [],
            'segment_ids': [],
            'label_ids': []
        }

        for feature in features:
            x['input_ids'].append(feature.input_ids)
            x['input_mask'].append(feature.input_mask)
            x['segment_ids'].append(feature.segment_ids)
            x['label_ids'].append(feature.label_id)

        x = {item: np.array(x[item]) for item in x}

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=x, num_epochs=1, shuffle=False)

        return input_fn


class FileInputFn(InputFn):
    """Data converters for sequence classification data sets saved as tfrecord format."""

    def __init__(self, max_seq_len, input_file_dir, batch_size):
        super(FileInputFn, self).__init__(max_seq_len)
        self.input_file_dir = os.path.abspath(input_file_dir)
        self.batch_size = batch_size
        self._get_input_fn_from_file()

    def _get_input_fn_from_file(self):
        self.train_input_fn = self._file_based_input_fn_builder(
            input_file=os.path.join(self.input_file_dir, 'train.tfrecord'),
            is_training=True,
            drop_remainder=True)
        self.dev_input_fn = self._file_based_input_fn_builder(
            input_file=os.path.join(self.input_file_dir, 'dev.tfrecord'),
            is_training=False,
            drop_remainder=False)
        if tf.gfile.Exists(os.path.join(self.input_file_dir, 'test.tfrecord')):
            self.test_input_fn = self._file_based_input_fn_builder(
                input_file=os.path.join(self.input_file_dir, 'test.tfrecord'),
                is_training=False,
                drop_remainder=False)
        else:
            self.test_input_fn = None

    def _file_based_input_fn_builder(self, input_file, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to Estimator."""
        name_to_features = {
            'input_ids': tf.FixedLenFeature([self.max_seq_len], tf.int64),
            'input_mask': tf.FixedLenFeature([self.max_seq_len], tf.int64),
            'segment_ids': tf.FixedLenFeature([self.max_seq_len], tf.int64),
            'label_ids': tf.FixedLenFeature([], tf.int64),
            'is_real_example': tf.FixedLenFeature([], tf.int64)}

        def _decode_record(record, name_to_features):
            """Decodes a record to a Tensorflow example."""
            example = tf.parse_single_example(record, name_to_features)
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn():
            """The actual input function."""
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=1000)
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=self.batch_size,
                    drop_remainder=drop_remainder))
            return d

        return input_fn
