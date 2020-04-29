#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

pretrained_urls = {
    'bert_zh': 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
    'bert_wwm_ext_zh': 'https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip',
    'albert_xlarge_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip',
    'albert_large_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_large_zh.zip',
    'albert_base_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_base_zh.zip',
    'albert_base_ext_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip',
    'albert_small_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip',
    'albert_tiny_zh_brightmart': 'https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip',
    'roberta_zh_brightmart': 'https://storage.googleapis.com/chineseglue/pretrain_models/roeberta_zh_L-24_H-1024_A-16.zip',
    'roberta_wwm_ext_zh_brightmart': 'https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_ext_L-12_H-768_A-12.zip',
    'roberta_wwm_ext_large_zh_brightmart': 'https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip'
}

pretrained_names = list(pretrained_urls.keys())

pretrained_type = {}
for model_name in pretrained_names:
    if model_name.startswith('bert'):
        pretrained_type[model_name] = 'bert'
    else:
        pretrained_type[model_name] = '_'.join(
            [model_name.split('_')[0], model_name.split('_')[-1]])
