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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pyclue.tf1.nlu.modeling.optimizations import create_optimizer
from pyclue.tf1.nlu.modeling.utils.checkpoint_utils import get_assignment_map_from_checkpoint
from pyclue.tf1.nlu.modeling.utils.tokenization_utils import FullTokenizer
from pyclue.tf1.tasks.classification.inputs import ClassifierProcessorFile
from pyclue.tf1.tasks.classification.inputs import ClassifierInputFnFile
from pyclue.tf1.tasks.classification.models import PretrainedClassifierModel
from pyclue.tf1.tasks.classification.models import TraditionalClassifierModel
from pyclue.tf1.nlu.modeling.metrics import precision, recall, f1

mpl.rcParams['font.sans-serif'] = ['SimHei']


class ClassifierTrainer(object):

    def __init__(self, output_dir, random_seed=0):
        self.base_output_dir = os.path.abspath(output_dir)
        self.random_seed = random_seed
        tf.set_random_seed(self.random_seed)

    def build_model(self, nlu_name=None, from_pretrained=True, nlu_model_type=None,
                    nlu_vocab_file=None, nlu_config_file=None, nlu_init_checkpoint_file=None,
                    downstream_name=None, max_seq_len=512):
        self.from_pretrained = from_pretrained
        if self.from_pretrained:
            self.classifier_model = PretrainedClassifierModel(
                nlu_name=nlu_name,
                nlu_model_type=nlu_model_type,
                nlu_vocab_file=nlu_vocab_file,
                nlu_config_file=nlu_config_file,
                nlu_init_checkpoint_file=nlu_init_checkpoint_file,
                downstream_name=downstream_name)
        else:
            self.classifier_model = TraditionalClassifierModel(
                nlu_name=nlu_name,
                nlu_model_type=nlu_model_type,
                nlu_vocab_file=nlu_vocab_file,
                nlu_config_file=nlu_config_file,
                nlu_init_checkpoint_file=nlu_init_checkpoint_file,
                downstream_name=downstream_name)
        self.nlu_name = self.classifier_model.nlu_name
        self.nlu_model_type = self.classifier_model.nlu_model_type
        self.nlu_vocab_file = self.classifier_model.nlu_vocab_file
        self.nlu_init_checkpoint_file = self.classifier_model.nlu_init_checkpoint_file
        self.nlu_config = self.classifier_model.nlu_config
        assert max_seq_len < self.nlu_config.max_position_embeddings, \
            'Cannot use sequence length %d because the %s model' \
            'was only trained up to sequence length %d' \
            % (max_seq_len, self.nlu_model_type,
               self.nlu_config.max_position_embeddings)
        self.max_seq_len = max_seq_len
        self.embedding_dim = self.nlu_config.hidden_size

        self.output_dir = os.path.join(self.base_output_dir, self.nlu_name)
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)

    def load_data(self, data_dir, batch_size, inplace_tf_record=True):
        self.data_dir = os.path.abspath(data_dir)
        self.batch_size = batch_size
        self.inplace_tf_record = inplace_tf_record
        self._load_tokenizer()
        self._load_processor()
        self._load_input_fn()

    def train_and_evaluate(self, num_train_epochs, warmup_proportion, learning_rate,
                           log_steps=50, metric_name='accuracy', save_checkpoints_steps=200,
                           max_steps_without_increase=None, min_steps=None, mode=0,
                           apply_best_checkpoint=True):
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.num_train_steps = int(self.num_train_examples / self.batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        self.learning_rate = learning_rate
        self.log_steps = log_steps
        self.save_checkpoints_steps = save_checkpoints_steps

        self._load_estimator()

        if mode == 0:
            model_file_dict = self._asynchronous_train_and_eval(apply_best_checkpoint)
        elif mode == 1:
            model_file_dict = self._synchronous_train_and_eval(metric_name,
                                                               max_steps_without_increase,
                                                               min_steps)
        else:
            raise ValueError('`mode` argument can only be 0 (asynchronous) or 1 (synchronous)'
                             ' during train_and_evaluate')
        return model_file_dict

    def _asynchronous_train_and_eval(self, apply_best_checkpoint=True):
        # train
        print('***** train phase *****')
        print('  Num train examples = %d' % self.num_train_examples)
        print('  Batch size = %d' % self.batch_size)
        print('  Num train steps = %d' % self.num_train_steps)
        self.estimator.train(input_fn=self.train_input_fn, max_steps=self.num_train_steps)

        # dev
        print('***** evaluate phase *****')
        print('  Num evaluate examples = %d' % self.num_dev_examples)
        steps_and_files = []
        files = tf.gfile.ListDirectory(self.checkpoint_dir)
        for file in files:
            if file.endswith('.index'):
                file_name = os.path.join(self.checkpoint_dir, file.strip('.index'))
                global_step = int(file_name.split('-')[-1])
                steps_and_files.append([global_step, file_name])
        steps_and_files = sorted(steps_and_files, key=lambda i: i[0])

        files_and_results = []
        output_eval_file = os.path.join(self.result_output_dir, 'dev_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            for global_step, file_name in steps_and_files[1:]:
                result = self.estimator.evaluate(
                    input_fn=self.dev_input_fn,
                    checkpoint_path=file_name)
                files_and_results.append([file_name, result['accuracy']])
                writer.write('***** dev results %s *****\n' % file_name)
                for key in sorted(result.keys()):
                    writer.write('%s = %s\n' % (key, str(result[key])))
        files_and_results = sorted(files_and_results, key=lambda i: i[1], reverse=True)

        best_checkpoint_file = files_and_results[0][0]
        last_checkpoint_file = steps_and_files[-1][-1]
        if apply_best_checkpoint:
            self.ckpt_model_file = best_checkpoint_file
        else:
            self.ckpt_model_file = last_checkpoint_file
        self.pb_model_file = self._save_model(checkpoint_path=self.ckpt_model_file)

        # test
        print('***** test phase *****')
        print('  Num test examples = %d' % self.num_test_examples)
        output_eval_file = os.path.join(self.result_output_dir, 'test_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            result = self.estimator.evaluate(
                input_fn=self.test_input_fn,
                checkpoint_path=self.ckpt_model_file)
            writer.write('***** test results %s *****\n' % self.ckpt_model_file)
            for key in sorted(result.keys()):
                writer.write('%s = %s\n' % (key, str(result[key])))

        return {
            'ckpt_model_file': self.ckpt_model_file,
            'pb_model_file': self.pb_model_file
        }

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
            self.estimator, metric_name=metric_name,
            max_steps_without_increase=max_steps_without_increase,
            min_steps=min_steps)
        exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=self._serving_input_receiver_fn(),
            exports_to_keep=1)
        train_spec = tf.estimator.TrainSpec(input_fn=self.train_input_fn,
                                            max_steps=self.num_train_steps,
                                            hooks=[early_stop_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=self.dev_input_fn,
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
        print('***** test phase *****')
        print('  Num test examples = %d' % self.num_test_examples)
        output_eval_file = os.path.join(self.result_output_dir, 'test_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            result = self.estimator.evaluate(
                input_fn=self.test_input_fn,
                checkpoint_path=self.ckpt_model_file)
            writer.write('***** test results %s *****\n' % self.ckpt_model_file)
            for key in sorted(result.keys()):
                writer.write('%s = %s\n' % (key, str(result[key])))

        return {
            'ckpt_model_file': self.ckpt_model_file,
            'pb_model_file': self.pb_model_file
        }

    def _serving_input_receiver_fn(self):
        feature_map = {
            'input_ids': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_ids'),
            'input_mask': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='input_mask'),
            'segment_ids': tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name='segment_ids'),
            'label_ids': tf.placeholder(tf.int32, shape=[None], name='label_ids')}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        return serving_input_receiver_fn

    def _save_model(self, checkpoint_path):
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
        return saved_path

    def _load_tokenizer(self):
        if self.from_pretrained:
            self.tokenizer = FullTokenizer(self.nlu_vocab_file)
        else:
            pass  # TODO: add traditional tokenizer

    def _load_processor(self):
        self.processor = ClassifierProcessorFile(
            max_seq_len=self.max_seq_len, tokenizer=self.tokenizer,
            data_dir=self.data_dir, save_tf_record_dir=self.data_dir,
            inplace_tf_record=self.inplace_tf_record)

        self.label_list = self.processor.labels
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
        self.input_fn_builder = ClassifierInputFnFile(
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
            input_ids = features['input_ids']
            input_mask = features['input_mask']
            segment_ids = features['segment_ids']
            label_ids = features['label_ids']

            if 'is_real_example' in features:
                is_real_example = tf.cast(features['is_real_example'], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            batch_loss, per_example_loss, probabilities, logits, predictions = self.classifier_model(
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                labels=label_ids,
                num_labels=self.num_labels)

            if self.from_pretrained:
                tvars = tf.trainable_variables()
                assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
                    tvars, self.nlu_init_checkpoint_file)
                tf.train.init_from_checkpoint(self.nlu_init_checkpoint_file, assignment_map)

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
                    batch_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps)
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
                    'predictions': predictions,
                    'probabilities': probabilities}
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)
            return output_spec

        return model_fn

    def predict(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        features = self.processor.get_features_for_inputs(texts)

        result = self.estimator.predict(
            input_fn=self.input_fn_builder.predict_input_fn(features=features),
            checkpoint_path=self.ckpt_model_file)
        result = list(result)

        predictions = [item['predictions'] for item in result]
        probabilities = [item['probabilities'].tolist() for item in result]

        return [{
            'text': text,
            'prediction': self.label_map_reverse[prediction],
            'probability': probability
        } for text, prediction, probability in zip(texts, predictions, probabilities)]

    def predict_from_file(self, input_file):

        texts = self.processor._read_file(input_file)
        texts = np.squeeze(texts).tolist()

        return self.predict(texts)

    def quality_inspection(self, input_file, save_path=None):

        texts = self.processor._read_file(input_file)

        features = self.processor.get_features_for_inputs(texts)

        result = self.estimator.predict(
            input_fn=self.input_fn_builder.predict_input_fn(features=features),
            checkpoint_path=self.ckpt_model_file)
        result = list(result)

        predictions = [item['predictions'] for item in result]
        probabilities = [item['probabilities'].tolist() for item in result]

        if not save_path:
            save_path = os.path.join(self.result_output_dir, 'quality_inspection')
        if not tf.gfile.Exists(save_path):
            tf.gfile.MakeDirs(save_path)

        with tf.gfile.GFile(os.path.join(save_path, input_file.split('/')[-1]), 'w') as writer:
            for text, prediction, probability in zip(texts, predictions, probabilities):
                prediction = self.label_map_reverse[prediction]
                if text[0] != prediction:
                    writer.write(
                        'text = %s, true = %s, pred = %s, probability = %s\n'
                        % (text[1], text[0], prediction, probability))


class LoggingMetricsHook(tf.train.SessionRunHook):

    def __init__(self, metric_ops, label_map_reverse, save_steps, output_dir):
        self.metric_ops = metric_ops
        self.metrics_num = len({item: metric_ops[item]
                                for item in metric_ops
                                if not item.endswith('update')})
        self.label_map_reverse = label_map_reverse
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.metric_save_path = os.path.join(self.output_dir, 'train_metrics.txt')
        self.figure_save_path = os.path.join(self.output_dir, 'train_metrics.png')
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)

    def begin(self):
        self.step = -1
        self.metric_results = {metric_name: [] for metric_name in self.metric_ops}
        self.metric_writer = tf.gfile.GFile(self.metric_save_path, 'w')

    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.metric_ops)

    def after_run(self, run_context, run_values):
        if self.step % self.save_steps == 0:
            results = run_values.results
            for metric_name in results:
                self.metric_results[metric_name].append(results[metric_name])
            results = {item: results[item] for item in results if not item.endswith('update')}
            results_list = {item: results[item] for item in results if isinstance(results[item], list)}
            results_scalar = {item: results[item] for item in results if not isinstance(results[item], list)}
            logging_info = 'step = %d, %s, %s' \
                           % (self.step,
                              ', '.join(['%s = %.6f' % item for item in results_scalar.items()]),
                              ', '.join(['%s = %s' % (item[0],
                                                      [float(i) for i in list(map(lambda x: '%.6f' % x, item[1]))])
                                         for item in results_list.items()]))
            print(logging_info)
            self.metric_writer.write(logging_info + '\n')

    def end(self, session):
        self.metric_writer.close()
        self.metric_results = {item: self.metric_results[item]
                               for item in self.metric_results
                               if not item.endswith('update')}
        fig, axs = plt.subplots(self.metrics_num, 1, sharex=True)
        fig.set_size_inches(16, 4.5 * self.metrics_num)
        for i, metric_name in enumerate(self.metric_results):
            metric_result = self.metric_results[metric_name]
            steps = np.arange(0, self.save_steps*len(metric_result), self.save_steps)
            p = axs[i].plot(steps, self.metric_results[metric_name])
            axs[i].set_ylabel(metric_name)
            axs[i].grid(True)
            if np.array(self.metric_results[metric_name]).ndim == 2:
                axs[i].legend(p, [self.label_map_reverse[item] for item in
                                  range(len(self.metric_results[metric_name][0]))])
        fig.tight_layout()
        fig.savefig(self.figure_save_path)
