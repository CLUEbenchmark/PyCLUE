#/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hnswlib
import numpy as np
import time
from pyclue.tf1.tasks.siamese.predict import SiamesePredictor

predict_configs = {
    'cache_file': '',
    'pb_model_file': './results/albert_tiny_zh_brightmart/20200427-104052/serving_model/1587955899',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'max_seq_len': 30
}


def run_predict():
    # read configs
    cache_file = predict_configs.get('cache_file')
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

    cache_texts, cache_embeddings, cache_labels = predictor.get_embedding_from_file(cache_file)
    num_cache, embedding_dim = cache_embeddings.shape
    # application of hnswlib
    # declaring index
    index_nms = hnswlib.Index(space='cosine', dim=embedding_dim)
    index_nms.load_index(os.path.join(pb_model_file, 'cache.index'))
    # controlling the recall by setting ef:
    index_nms.set_ef(50)  # ef should always be > k

    # predict test
    # single predict
    start_time = time.time()
    single_input_text = 's i na 微薄 坤 ikun ＋ 交流'
    single_embedding = predictor.get_embedding(single_input_text)
    # query dataset, k - number of closest elements (returns 2 numpy arrays)
    single_indexes, single_distances = index_nms.knn_query(single_embedding, k=5)
    duration = time.time() - start_time
    single_similar_labels = [cache_labels[item] for item in single_indexes]
    single_similar_texts = [cache_texts[item] for item in single_indexes]
    single_distances = np.squeeze(single_distances)
    single_similarities = 1 - single_distances
    scaled_single_similarities = (single_similarities + 1) / 2
    print('input = %s' % single_input_text)
    for i, single_distance, single_similarity, scaled_single_similarity, single_similar_label, single_similar_text \
            in zip(range(1, len(single_similar_texts) + 1),
                   single_distances,
                   single_similarities,
                   scaled_single_similarities,
                   single_similar_labels,
                   single_similar_texts):
        print('rank = %s, distance = %.6f, '
              'similarity = %.6f, scaled_similarity = %.6f, '
              'similar_label = %s, similar_text = %s'
              % (i, single_distance, single_similarity, scaled_single_similarity,
                 single_similar_label, single_similar_text))
    print('cost time = %s ms' % str(1000 * duration))

    # batch predict
    start_time = time.time()
    batch_input_texts = ['s i na 微薄 坤 ikun ＋ 交流', '没有救赎可言']
    batch_embeddings = predictor.get_embedding(batch_input_texts)
    # query dataset, k - number of closest elements (returns 2 numpy arrays)
    batch_indexes, batch_distances = index_nms.knn_query(batch_embeddings, k=5)
    duration = time.time() - start_time
    batch_similar_labels = [[cache_labels[item] for item in term] for term in batch_indexes]
    batch_similar_texts = [[cache_texts[item] for item in term] for term in batch_indexes]
    batch_distances = np.squeeze(batch_distances)
    batch_similarities = 1 - batch_distances
    scaled_batch_similarities = (batch_similarities + 1) / 2
    print('batch rank test: ')
    for i, text in enumerate(batch_input_texts):
        print('input = %s' % text)
        for j, distance, similarity, scaled_similarity, similar_label, similar_text \
                in zip(range(1, len(batch_similar_texts[i]) + 1),
                       batch_distances[i],
                       batch_similarities[i],
                       scaled_batch_similarities[i],
                       batch_similar_labels[i],
                       batch_similar_texts[i]):
            print('rank = %s, distance = %.6f, '
                  'similarity = %.6f, scaled_similarity = %.6f, '
                  'similar_label = %s, similar_text = %s'
                  % (j, distance, similarity, scaled_similarity, similar_label, similar_text))
    print('cost time = %s ms' % str(1000 * duration))


if __name__ == '__main__':
    run_predict()
