#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import numpy as np
import tensorflow as tf

from pyclue.tf1.models.engine.checkpoint import get_assignment_map_from_checkpoint
from pyclue.tf1.models.engine.hooks import LoggingMetricsHook
from pyclue.tf1.models.engine.metrics import precision, recall, f1
from pyclue.tf1.models.engine.optimizations import create_optimizer
from pyclue.tf1.tasks.sentence_pair.siamese.inputs import FileProcessor, FileInputFn
from pyclue.tf1.tasks.sentence_pair.siamese.models import SiameseModel
from pyclue.tf1.tokenizers.bert_tokenizer import FullTokenizer  # Add more tokenizers


class Trainer(object):

    def __init__(self, output_dir, random_seed=0):
        self.base_output_dir = os.path.abspath(output_dir)
        self.random_seed = random_seed
        tf.set_random_seed(self.random_seed)

    def build_model(self, model_name=None, model_type=None, vocab_file=None,
                    config_file=None, init_checkpoint_file=None, max_seq_len=512):
        # model
        self.model = SiameseModel(
            model_name=model_name,
            model_type=model_type,
            vocab_file=vocab_file,
            config_file=config_file,
            init_checkpoint_file=init_checkpoint_file)
        self.model_name = self.model.model_name
        self.model_type = self.model.model_type
        self.vocab_file = self.model.vocab_file
        self.init_checkpoint_file = self.model.init_checkpoint_file
        self.pretrained_config = self.model.pretrained_config

        # max_seq_len and embedding_dim
        assert max_seq_len < self.pretrained_config.max_position_embeddings, \
            'Can not use max_seq_len %d because the %s model ' \
            'was only trained up to seq_len %d.' \
            % (max_seq_len, self.model_name, self.pretrained_config.max_position_embeddings)
        self.max_seq_len = max_seq_len
        self.embedding_dim = self.pretrained_config.hidden_size

        # tokenizer
        if self.model_type == 'bert':
            self.tokenizer = FullTokenizer(self.vocab_file)
        elif self.model_type == 'albert':
            self.tokenizer = FullTokenizer(self.vocab_file)
        else:
            raise ValueError('model_type %s unknown.' % self.model_type)

        # output_dir
        self.output_dir = os.path.join(self.base_output_dir, self.model_name)
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)

    def load_data(self, data_dir, batch_size, recreate_tfrecord=True):
        self.data_dir = os.path.abspath(data_dir)
        self.batch_size = batch_size
        self.recreate_tfrecord = recreate_tfrecord
        self._load_processor()
        self._load_input_fn()

    def _load_processor(self):
        self.processor = FileProcessor(
            max_seq_len=self.max_seq_len, tokenizer=self.tokenizer,
            data_dir=self.data_dir, save_tfrecord_dir=self.data_dir,
            recreate_tfrecord=self.recreate_tfrecord)
        self.labels = self.processor.labels
        self.label_map = self.processor.label_map
        self.label_map_reverse = self.processor.label_map_reverse
        self.num_labels = self.processor.num_labels

        self.train_examples = self.processor.train_examples
        self.num_train_examples = self.processor.num_train_examples
        self.dev_examples = self.processor.dev_examples
        self.num_dev_examples = self.processor.num_dev_examples
        self.test_examples = self.processor.test_examples
        self.num_test_examples = self.processor.num_test_examples

    def _load_input_fn(self):
        self.input_fn_builder = FileInputFn(
            self.max_seq_len, self.data_dir, self.batch_size)
        self.train_input_fn = self.input_fn_builder.train_input_fn
        self.dev_input_fn = self.input_fn_builder.dev_input_fn
        self.test_input_fn = self.input_fn_builder.test_input_fn

    def _load_estimator(self):
        self.build_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.checkpoint_dir = os.path.join(
            self.output_dir, self.build_time, 'checkpoints')
        self.serving_model_dir = os.path.join(
            self.output_dir, self.build_time, 'serving_model')
        self.result_output_dir = os.path.join(
            self.output_dir, self.build_time, 'outputs')
        self.run_config = tf.estimator.RunConfig(
            model_dir=self.checkpoint_dir,
            tf_random_seed=self.random_seed,
            save_checkpoints_steps=self.save_checkpoints_steps,
            keep_checkpoint_max=10)
        self.estimator = tf.estimator.Estimator(
            self._load_model_fn(), config=self.run_config)

    def _load_model_fn(self):
        """Returns `model_fn` for estimator."""

        def model_fn(features, labels, mode, params):
            """The `model_fn` for estimator."""
            input_ids_1 = features['input_ids_1']
            input_mask_1 = features['input_mask_1']
            segment_ids_1 = features['segment_ids_1']
            input_ids_2 = features['input_ids_2']
            input_mask_2 = features['input_mask_2']
            segment_ids_2 = features['segment_ids_2']
            label_ids = features['label_ids']

            if 'is_real_example' in features:
                is_real_example = tf.cast(features['is_real_example'], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            batch_loss, per_example_loss, probabilities, logits, predictions, \
            pooled_output_1, pooled_output_2, cos_sim = self.model(
                is_training=is_training,
                input_ids_1=input_ids_1,
                input_mask_1=input_mask_1,
                segment_ids_1=segment_ids_1,
                input_ids_2=input_ids_2,
                input_mask_2=input_mask_2,
                segment_ids_2=segment_ids_2,
                labels=label_ids,
                num_labels=self.num_labels)

            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
                tvars, self.init_checkpoint_file)
            tf.train.init_from_checkpoint(self.init_checkpoint_file, assignment_map)

            # metrics: returns tuple (value_op, update_op)
            value_accuracy_op, update_accuracy_op = tf.metrics.accuracy(
                labels=label_ids, predictions=predictions, weights=is_real_example)
            value_loss_op, update_loss_op = tf.metrics.mean(
                values=per_example_loss, weights=is_real_example)
            value_precision_op, update_precision_op = precision(
                labels=label_ids, predictions=predictions, num_classes=self.num_labels,
                weights=is_real_example, average=None)
            value_recall_op, update_recall_op = recall(
                labels=label_ids, predictions=predictions, num_classes=self.num_labels,
                weights=is_real_example, average=None)
            value_f1_op, update_f1_op = f1(
                labels=label_ids, predictions=predictions, num_classes=self.num_labels,
                weights=is_real_example, average=None)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_metric_ops = {
                    'accuracy': value_accuracy_op,
                    'accuracy_update': update_accuracy_op,
                    'loss': value_loss_op,
                    'loss_update': update_loss_op,
                    'loss_batch': batch_loss,
                    'precision': value_precision_op,
                    'precision_update': update_precision_op,
                    'recall': value_recall_op,
                    'recall_update': update_recall_op,
                    'f1': value_f1_op,
                    'f1_update': update_f1_op}

                train_op = create_optimizer(
                    batch_loss, self.optimizer_name, self.learning_rate, self.num_train_steps, self.num_warmup_steps)
                train_metrics_hook = LoggingMetricsHook(
                    metric_ops=train_metric_ops,
                    label_map_reverse=self.label_map_reverse,
                    save_steps=self.log_steps,
                    output_dir=self.result_output_dir)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=batch_loss,
                    train_op=train_op,
                    training_hooks=[train_metrics_hook])
            elif mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {
                    'accuracy': (value_accuracy_op, update_accuracy_op),
                    'precision': (value_precision_op, update_precision_op),
                    'recall': (value_recall_op, update_recall_op),
                    'f1': (value_f1_op, update_f1_op)}
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=batch_loss,
                    eval_metric_ops=eval_metric_ops)
            else:
                predictions = {
                    'text_a_embedding': pooled_output_1,
                    'text_b_embedding': pooled_output_2,
                    'cos_sims': cos_sim,
                    'predictions': predictions,
                    'probabilities': probabilities}
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)
            return output_spec

        return model_fn

    def train(self, num_train_epochs, warmup_proportion, learning_rate, optimizer_name, log_steps=50):
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.num_train_steps = int(self.num_train_examples / self.batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.log_steps = log_steps
        self._load_estimator()
        self._asynchronous_train()
        self.pb_model_file = self.save_model(checkpoint_path=self.ckpt_model_file)
        model_file_dict = {
            'ckpt_model_file': self.ckpt_model_file,
            'pb_model_file': self.pb_model_file
        }
        return model_file_dict

    def train_and_evaluate(self, num_train_epochs, warmup_proportion, learning_rate, optimizer_name,
                           log_steps=50, metric_name='accuracy', save_checkpoints_steps=200,
                           max_steps_without_increase=None, min_steps=None, mode=0,
                           apply_best_checkpoint=True):
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.num_train_steps = int(self.num_train_examples / self.batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.log_steps = log_steps
        self.save_checkpoints_steps = save_checkpoints_steps
        self._load_estimator()

        if mode == 0:
            model_file_dict = self._asynchronous_train_and_eval(
                metric_name, apply_best_checkpoint)
        elif mode == 1:
            model_file_dict = self._synchronous_train_and_eval(
                metric_name, max_steps_without_increase, min_steps)
        else:
            raise ValueError('`mode` argument can only be 0 (asynchronous) or 1 (synchronous)'
                             ' during train_and_evaluate')
        return model_file_dict

    def _asynchronous_train(self):
        # train
        print('***** train phase *****')
        print('  Num train examples = %d' % self.num_train_examples)
        print('  Batch size = %d' % self.batch_size)
        print('  Num train steps = %d' % self.num_train_steps)
        self.estimator.train(input_fn=self.train_input_fn, max_steps=self.num_train_steps)

        self.steps_and_files = []
        files = tf.gfile.ListDirectory(self.checkpoint_dir)
        for file in files:
            if file.endswith('.index'):
                file_name = os.path.join(self.checkpoint_dir, file.strip('.index'))
                global_step = int(file_name.split('-')[-1])
                self.steps_and_files.append([global_step, file_name])
        self.steps_and_files = sorted(self.steps_and_files, key=lambda i: i[0])
        self.last_checkpoint_file = self.steps_and_files[-1][-1]
        self.ckpt_model_file = self.last_checkpoint_file

    def _asynchronous_eval(self, metric_name, apply_best_checkpoint=True):
        # dev
        print('***** evaluate phase *****')
        print('  Num evaluate examples = %d' % self.num_dev_examples)

        files_and_results = []
        output_eval_file = os.path.join(self.result_output_dir, 'dev_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            for global_step, file_name in self.steps_and_files[1:]:
                result = self.estimator.evaluate(
                    input_fn=self.dev_input_fn,
                    checkpoint_path=file_name)
                files_and_results.append([file_name, result[metric_name]])
                writer.write('***** dev results %s *****\n' % file_name)
                for key in sorted(result.keys()):
                    writer.write('%s = %s\n' % (key, str(result[key])))
        files_and_results = sorted(files_and_results, key=lambda i: i[1], reverse=True)

        self.best_checkpoint_file = files_and_results[0][0]
        if apply_best_checkpoint:
            self.ckpt_model_file = self.best_checkpoint_file
        else:
            self.ckpt_model_file = self.last_checkpoint_file

    def _test(self):
        print('***** test phase *****')
        print('  Num test examples = %d' % self.num_test_examples)
        if self.num_test_examples != 0:
            output_eval_file = os.path.join(self.result_output_dir, 'test_results.txt')
            with tf.gfile.GFile(output_eval_file, 'w') as writer:
                result = self.estimator.evaluate(
                    input_fn=self.test_input_fn,
                    checkpoint_path=self.ckpt_model_file)
                writer.write('***** test results %s *****\n' % self.ckpt_model_file)
                for key in sorted(result.keys()):
                    writer.write('%s = %s\n' % (key, str(result[key])))

    def _asynchronous_train_and_eval(self, metric_name, apply_best_checkpoint=True):
        # train
        self._asynchronous_train()
        # eval
        self._asynchronous_eval(
            metric_name=metric_name, apply_best_checkpoint=apply_best_checkpoint)
        self.pb_model_file = self.save_model(checkpoint_path=self.ckpt_model_file)
        # test
        self._test()
        model_file_dict = {
            'ckpt_model_file': self.ckpt_model_file,
            'pb_model_file': self.pb_model_file
        }
        return model_file_dict

    def _synchronous_train_and_eval(self, metric_name, max_steps_without_increase, min_steps):
        # train and dev
        print('***** train and evaluate phase *****')
        print('  Num train examples = %d' % self.num_train_examples)
        print('  Num evaluate examples = %d' % self.num_dev_examples)
        print('  Batch size = %d' % self.batch_size)
        print('  Num train steps = %d' % self.num_train_steps)

        if not max_steps_without_increase:
            max_steps_without_increase = int(self.num_train_steps // 10)
        if not min_steps:
            min_steps = self.num_warmup_steps

        early_stop_hook = tf.estimator.experimental.stop_if_no_increase_hook(
            self.estimator,
            metric_name=metric_name,
            max_steps_without_increase=max_steps_without_increase,
            min_steps=min_steps)
        exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=self._serving_input_receiver_fn(),
            exports_to_keep=1)
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.train_input_fn,
            max_steps=self.num_train_steps,
            hooks=[early_stop_hook])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.dev_input_fn,
            exporters=exporter,
            steps=None,
            start_delay_secs=120,
            throttle_secs=1)

        result, _ = tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        for file in tf.gfile.ListDirectory(self.checkpoint_dir):
            if file.endswith('.index'):
                self.ckpt_model_file = os.path.join(self.checkpoint_dir, file.strip('.index'))
        self.pb_model_file = self._save_model(checkpoint_path=self.ckpt_model_file)
        output_eval_file = os.path.join(self.result_output_dir, 'dev_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            writer.write('***** dev results %s *****\n' % self.ckpt_model_file)
            for key in sorted(result.keys()):
                writer.write('%s = %s\n' % (key, str(result[key])))

        # test
        self._test()

        model_file_dict = {
            'ckpt_model_file': self.ckpt_model_file,
            'pb_model_file': self.pb_model_file
        }
        return model_file_dict

    def _serving_input_receiver_fn(self):
        feature_map = {
            'input_ids_1': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_ids_1'),
            'input_mask_1': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_mask_1'),
            'segment_ids_1': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='segment_ids_1'),
            'input_ids_2': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_ids_2'),
            'input_mask_2': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_mask_2'),
            'segment_ids_2': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='segment_ids_2'),
            'label_ids': tf.placeholder(tf.int32, shape=[None], name='label_ids')}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        return serving_input_receiver_fn

    def save_model(self, checkpoint_path):
        if not tf.gfile.Exists(self.serving_model_dir):
            tf.gfile.MakeDirs(self.serving_model_dir)

        serving_input_receiver_fn = self._serving_input_receiver_fn()
        saved_path = self.estimator.export_saved_model(
            export_dir_base=self.serving_model_dir,
            serving_input_receiver_fn=serving_input_receiver_fn,
            checkpoint_path=checkpoint_path,
            experimental_mode=tf.estimator.ModeKeys.PREDICT)
        saved_path = saved_path.decode('utf-8')

        # save label_map_reverse for prediction
        label_map_reverse_file = os.path.join(
            saved_path, 'label_map_reverse.json')
        with tf.gfile.GFile(label_map_reverse_file, 'w') as f:
            json.dump(self.label_map_reverse, f, ensure_ascii=False, indent=4)

        # save model_config for prediction
        model_configs = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'vocab_file': self.vocab_file,
            'max_seq_len': self.max_seq_len
        }
        model_configs_file = os.path.join(
            saved_path, 'model_config.json')
        with tf.gfile.GFile(model_configs_file, 'w') as f:
            json.dump(model_configs, f, ensure_ascii=False, indent=4)
        return saved_path

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
        result = self.estimator.predict(
            input_fn=self.input_fn_builder.predict_input_fn(features=features),
            checkpoint_path=self.ckpt_model_file)
        result = list(result)
        cos_sims = [item['cos_sims'] for item in result]
        predictions = [item['predictions'] for item in result]
        probabilities = [item['probabilities'].tolist() for item in result]

        return [{
            'text_a': text[1],
            'text_b': text[2],
            'cos_sim': np.squeeze(cos_sim).tolist() / 100,
            'prediction': self.label_map_reverse[prediction],
            'probability': probability
        } for text, cos_sim, prediction, probability in
            zip(new_texts, cos_sims, predictions, probabilities)]

    def predict_from_file(self, input_file):
        texts = self.processor.read_file(input_file)
        texts = np.squeeze(texts).tolist()
        return self.predict(texts)

    def quality_inspection(self, input_file, save_path=None):
        texts = self.processor.read_file(input_file)
        if np.array(texts).ndim == 1:
            texts = [texts]
        texts = [item for item in texts if len(item) == 3]
        features = self.processor.get_features_for_inputs(texts)

        result = self.estimator.predict(
            input_fn=self.input_fn_builder.predict_input_fn(features=features),
            checkpoint_path=self.ckpt_model_file)
        result = list(result)

        cos_sims = [item['cos_sims'] for item in result]
        predictions = [item['predictions'] for item in result]
        probabilities = [item['probabilities'].tolist() for item in result]

        if not save_path:
            save_path = os.path.join(self.result_output_dir, 'quality_inspection')
        if not tf.gfile.Exists(save_path):
            tf.gfile.MakeDirs(save_path)

        with tf.gfile.GFile(os.path.join(save_path, input_file.split('/')[-1]), 'w') as writer:
            for text, cos_sim, prediction, probability in zip(texts, cos_sims, predictions, probabilities):
                prediction = self.label_map_reverse[prediction]
                if text[0] != prediction:
                    writer.write(
                        'text_a = %s, text_b = %s, '
                        'true = %s, pred = %s, '
                        'probability = %s, cos_sim = %s\n'
                        % (text[1], text[2], text[0], prediction, probability, np.squeeze(cos_sim) / 100))
