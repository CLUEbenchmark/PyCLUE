#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyclue.tf1.tasks.siamese.predict import SiamesePredictor

predict_configs = {
    'pb_model_file': './results/albert_tiny_zh_brightmart/20200424-212228/serving_model/1587734690',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'max_seq_len': 35
}


def run_predict():
    # read configs
    pb_model_file = predict_configs.get('pb_model_file')
    from_pretrained = predict_configs.get('from_pretrained')
    nlu_name = predict_configs.get('nlu_name')
    max_seq_len = predict_configs.get('max_seq_len')

    # predict test
    predictor = SiamesePredictor(model_file=pb_model_file)
    predictor.build_model(
        nlu_name=nlu_name,
        from_pretrained=from_pretrained,
        max_seq_len=max_seq_len)

    single_result = predictor.predict(['花呗要怎么还钱啊', '花呗还款在哪个地方能告诉我不'])
    print('single predict test: ')
    print(single_result)

    batch_results = predictor.predict(
        [['花呗要怎么还钱啊', '花呗还款在哪个地方能告诉我不'],
         ['明天是周末不上班', '星期六需要去公司吗']])
    print('batch predict test: ')
    print(batch_results)


if __name__ == '__main__':
    run_predict()
