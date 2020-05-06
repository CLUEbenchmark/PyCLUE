# -*- coding: utf-8 -*-

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
            steps = np.arange(0, self.save_steps * len(metric_result), self.save_steps)
            p = axs[i].plot(steps, self.metric_results[metric_name])
            axs[i].set_ylabel(metric_name)
            axs[i].grid(True)
            if np.array(self.metric_results[metric_name]).ndim == 2:
                axs[i].legend(p, [self.label_map_reverse[item] for item in
                                  range(len(self.metric_results[metric_name][0]))])
        fig.tight_layout()
        fig.savefig(self.figure_save_path)
