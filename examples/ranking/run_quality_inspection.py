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

import hnswlib
import numpy as np
from pyclue.tf1.tasks.siamese.predict import SiamesePredictor

save_path = './quality_inspection'
data_dir = ''

quality_inspection_configs_0427_104052 = {
    'cache_file': '',
    'pb_model_file': './results/albert_tiny_zh_brightmart/20200427-104052/serving_model/1587955899',
    'inspect_file_dir': './test_files',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'max_seq_len': 30,
    'save_dir': '20200427-104052',
    'threshold': 0.75
}

quality_inspection_configs_0427_183134 = {
    'cache_file': '',
    'pb_model_file': './results/albert_tiny_zh_brightmart/20200427-183134/serving_model/1587984152',
    'inspect_file_dir': './test_files',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'max_seq_len': 30,
    'save_dir': '20200427-183134',
    'threshold': 0.75
}


def single_file_inspection(predictor, input_file, save_dir, index_nms, cache_texts, cache_labels, threshold):
    output_file = os.path.abspath(os.path.join(save_dir, input_file.split('/')[-1]))
    writer = open(output_file, 'w')
    # batch predict
    start_time = time.time()
    batch_input_texts, batch_embeddings, batch_labels = predictor.get_embedding_from_file(input_file)
    # query dataset, k - number of closest elements (returns 2 numpy arrays)
    batch_indexes, batch_distances = index_nms.knn_query(batch_embeddings, k=1)
    duration = time.time() - start_time
    batch_similar_labels = np.squeeze([[cache_labels[item] for item in term] for term in batch_indexes])
    batch_similar_texts = np.squeeze([[cache_texts[item] for item in term] for term in batch_indexes])
    batch_distances = np.squeeze(batch_distances)
    batch_similarities = 1 - batch_distances
    scaled_batch_similarities = (batch_similarities + 1) / 2
    for i, text, label in zip(range(len(batch_input_texts)), batch_input_texts, batch_labels):
        if label not in set(cache_labels):
            if scaled_batch_similarities[i] >= threshold:
                writer.write('text = %s, labels = %s, surpass threshold %.2f\n'
                             % (text, label, threshold))
        else:
            if scaled_batch_similarities[i] < threshold:
                writer.write('text = %s, labels = %s, less than threshold %.2f\n'
                             % (text, label, threshold))
            elif batch_similar_labels[i] != label:
                writer.write('text = %s, label = %s, similar_text = %s, similar_label = %s, scaled_similarity = %.6f\n'
                             % (text, label, batch_similar_texts[i], batch_similar_labels[i],
                                scaled_batch_similarities[i]))
    print('cost time = %s ms' % str(1000 * duration))
    writer.close()


def run_quality_inspection(quality_inspection_configs):
    # read configs
    cache_file = quality_inspection_configs.get('cache_file')
    pb_model_file = quality_inspection_configs.get('pb_model_file')
    inspect_file_dir = quality_inspection_configs.get('inspect_file_dir')
    from_pretrained = quality_inspection_configs.get('from_pretrained')
    nlu_name = quality_inspection_configs.get('nlu_name')
    max_seq_len = quality_inspection_configs.get('max_seq_len')
    save_dir = quality_inspection_configs.get('save_dir')
    save_dir = os.path.abspath(os.path.join(save_path, save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    threshold = quality_inspection_configs.get('threshold')

    # predict test
    predictor = SiamesePredictor(model_file=pb_model_file)
    predictor.build_model(
        nlu_name=nlu_name,
        from_pretrained=from_pretrained,
        max_seq_len=max_seq_len)

    cache_texts, cache_embeddings, cache_labels = predictor.get_embedding_from_file(cache_file)
    num_cache, embedding_dim = cache_embeddings.shape

    caches = np.squeeze(predictor.processor._read_file(cache_file)).tolist()
    caches_ndim = np.squeeze(caches).ndim
    if caches_ndim == 1:
        cache_labels = [-1] * len(caches)
        cache_texts = caches
    elif caches_ndim == 2:
        cache_labels = [item[0] for item in caches]
        cache_texts = [item[1] for item in caches]
    else:
        raise TypeError('cache_file format should be `label\\ttext` or `text`')

    # application of hnswlib
    # declaring index
    index_nms = hnswlib.Index(space='cosine', dim=embedding_dim)
    index_nms.load_index(os.path.join(pb_model_file, 'cache.index'))
    # controlling the recall by setting ef:
    index_nms.set_ef(50)  # ef should always be > k

    for input_file in os.listdir(inspect_file_dir):
        input_file = os.path.abspath(os.path.join(inspect_file_dir, input_file))
        single_file_inspection(predictor, input_file, save_dir, index_nms, cache_texts, cache_labels, threshold)


if __name__ == '__main__':
    run_quality_inspection(quality_inspection_configs_0427_104052)
    run_quality_inspection(quality_inspection_configs_0427_183134)
