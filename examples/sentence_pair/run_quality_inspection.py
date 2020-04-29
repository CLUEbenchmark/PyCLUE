#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pyclue.tf1.tasks.siamese.predict import SiamesePredictor

save_path = './quality_inspection'
data_dir = '/workspace/cpfs-data/siameses/lcqmc'

quality_inspection_configs_20200424 = {
    'pb_model_file': './results/albert_tiny_zh_brightmart/20200424-212228/serving_model/1587734690',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'max_seq_len': 35,
    'save_dir': '2020-0424'
}


def run_quality_inspection(quality_inspection_configs):
    # read configs
    pb_model_file = quality_inspection_configs.get('pb_model_file')
    from_pretrained = quality_inspection_configs.get('from_pretrained')
    nlu_name = quality_inspection_configs.get('nlu_name')
    max_seq_len = quality_inspection_configs.get('max_seq_len')
    save_dir = quality_inspection_configs.get('save_dir')
    save_dir = os.path.join(save_path, save_dir)

    # predict test
    predictor = SiamesePredictor(model_file=pb_model_file)
    predictor.build_model(
        nlu_name=nlu_name,
        from_pretrained=from_pretrained,
        max_seq_len=max_seq_len)

    predictor.quality_inspection(os.path.join(data_dir, 'train.txt'), save_dir)
    predictor.quality_inspection(os.path.join(data_dir, 'dev.txt'), save_dir)
    predictor.quality_inspection(os.path.join(data_dir, 'test.txt'), save_dir)


if __name__ == '__main__':
    run_quality_inspection(quality_inspection_configs_20200424)
