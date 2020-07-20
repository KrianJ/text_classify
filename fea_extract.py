# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/16 17:41"
__doc__ = """ 特征提取"""

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

"""读取数据集，标签集，词向量权重集"""
# allow_pickle:
# 允许使用 Python pickles 保存对象数组，
# Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
data = np.load('collect_data/data.npy', allow_pickle=True)
label = np.load('collect_data/labels.npy', allow_pickle=True)
words_weight = pd.read_csv('collect_data/tfidf_collect.csv', index_col=0)

"""训练词向量"""
param_explanation = """windows: 当前词与预测词在一个句子中的最大距离， 
min_count:筛选掉词频过低的词，
sg: CBOW(0);skip-gram(1)
hs: negative sampling(0); softmax(1)
workers: 并行数"""
word2vec = Word2Vec(data, size=200, window=3, min_count=5, sg=0, hs=1, iter=10, workers=25)
# word2vec.save('collect_data/word2vec.model')


def sentence_vec(sentence):
    """样本中分词的词向量与各自权重相乘，然后相加得到整个样本的词向量
    :param words: 一个句子的分词 <class list>
    :return words的词向量"""
    vec = np.zeros(200).reshape((1, 200))
    for word in sentence:
        try:
            # 包含信息熵的tf-idf
            weight = words_weight.loc[word, 'entropy_tfidf']
            vec += word2vec.wv[word].reshape((1, 200)) * weight
            # 原始tf-idf算法
            # vec += word2vec.wv[word].reshape((1, 200))
        except KeyError:
            continue
    return vec


def get_train_vec():
    train_vec = np.concatenate([sentence_vec(sentence) for sentence in data], axis=0)
    return train_vec


def get_label():
    return label


if __name__ == '__main__':
    print(word2vec.similarity('贴心', '不错'))
    train_vec = get_train_vec()
    print(train_vec.shape)


