#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from pyclue.tf1.tasks.classification.train import ClassifierTrainer

train_configs = {
    'data_dir': '', # your data file, should contained train.txt/dev.txt/test.txt
    'inplace_tf_record': False,
    'output_dir': '', # result output path
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'downstream_name': 'dense',
    'max_seq_len': 80,
    'num_train_epochs': 20,
    'warmup_proportion': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'save_checkpoints_steps': 200,
    'log_steps': 50,
    'train_and_evaluate_mode': 1,
    'max_steps_without_increase': 1000,
    'min_steps': 500
}


def run_train():
    # read configs
    data_dir = train_configs.get('data_dir')
    inplace_tf_record = train_configs.get('inplace_tf_record')
    output_dir = train_configs.get('output_dir')
    from_pretrained = train_configs.get('from_pretrained')
    nlu_name = train_configs.get('nlu_name')
    downstream_name = train_configs.get('downstream_name')
    max_seq_len = train_configs.get('max_seq_len')
    num_train_epochs = train_configs.get('num_train_epochs')
    warmup_proportion = train_configs.get('warmup_proportion')
    batch_size = train_configs.get('batch_size')
    learning_rate = train_configs.get('learning_rate')
    save_checkpoints_steps = train_configs.get('save_checkpoints_steps')
    log_steps = train_configs.get('log_steps')
    train_and_evaluate_mode = train_configs.get('train_and_evaluate_mode')
    max_steps_without_increase = train_configs.get('max_steps_without_increase')
    min_steps = train_configs.get('min_steps')

    # train test
    trainer = ClassifierTrainer(output_dir=output_dir)
    trainer.build_model(
        nlu_name=nlu_name,
        from_pretrained=from_pretrained,
        downstream_name=downstream_name,
        max_seq_len=max_seq_len)
    trainer.load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        inplace_tf_record=inplace_tf_record)
    model_file_dict = trainer.train_and_evaluate(
        num_train_epochs=num_train_epochs,
        warmup_proportion=warmup_proportion,
        learning_rate=learning_rate,
        log_steps=log_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        max_steps_without_increase=max_steps_without_increase,
        min_steps=min_steps,
        mode=train_and_evaluate_mode)
    print('best model save path: \n%s' % '\n'.join(['%s: %s' % item for item in model_file_dict.items()]))

    # predict test
    single_result = trainer.predict('') # single predict text
    print('single predict test: ')
    print(single_result)

    batch_results = trainer.predict(
        ['',
         '']) # batch predict text
    print('batch predict test: ')
    print(batch_results)


if __name__ == '__main__':
    run_train()
