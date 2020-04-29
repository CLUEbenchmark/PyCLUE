#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os

import hnswlib
import numpy as np
from pyclue.tf1.tasks.siamese.train import SiameseTrainer

train_configs = {
    'data_dir': '', # your data file path
    'inplace_tf_record': False,
    'output_dir': './results',
    'from_pretrained': True,
    'nlu_name': 'albert_tiny_zh_brightmart',
    'downstream_name': 'dense',
    'max_seq_len': 30,
    'num_train_epochs': 5,
    'warmup_proportion': 0.1,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'save_checkpoints_steps': 2000,
    'log_steps': 50,
    'train_and_evaluate_mode': 1,
    'max_steps_without_increase': 2000,
    'min_steps': 5000
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
    cache_file = os.path.join(data_dir, 'cache.txt')

    # train test
    trainer = SiameseTrainer(output_dir=output_dir)
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

    # generate and save embeddings for cache.txt
    cache_texts, cache_embeddings, cache_labels = trainer.get_embedding_from_file(cache_file)
    num_cache, embedding_dim = cache_embeddings.shape
    print('num_cache = %s, embedding_dim = %s' % (num_cache, embedding_dim))

    # application of hnswlib
    # declaring index
    index_nms = hnswlib.Index(space='cosine', dim=embedding_dim)
    # initializing index - the maximum number of elements should be know beforehand
    index_nms.init_index(max_elements=num_cache, ef_construction=100, M=100)
    # element insertion (can be called several times)
    index_nms.add_items(data=cache_embeddings, ids=range(num_cache))
    index_nms.save_index(os.path.join(model_file_dict['pb_model_file'], 'cache.index'))
    # controlling the recall by setting ef:
    index_nms.set_ef(ef=50)  # ef should always be > k (knn)

    # predict test
    # single predict
    single_input_text = 's i na 微薄 坤 ikun ＋ 交流'
    single_embedding = trainer.get_embedding(single_input_text)
    # query dataset, k - number of closest elements (returns 2 numpy arrays)
    single_indexes, single_distances = index_nms.knn_query(single_embedding, k=5)
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

    # batch predict
    batch_input_texts = ['s i na 微薄 坤 ikun ＋ 交流', '没有救赎可言']
    batch_embeddings = trainer.get_embedding(batch_input_texts)
    # query dataset, k - number of closest elements (returns 2 numpy arrays)
    batch_indexes, batch_distances = index_nms.knn_query(batch_embeddings, k=5)
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


if __name__ == '__main__':
    run_train()
