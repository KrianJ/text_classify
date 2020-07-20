# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/17 10:07"
__doc__ = """ 基于tf-idf算法计算每个单词的权重，从而得到句子的权重"""

import numpy as np
import pandas as pd


def count_times(word, data):
    """计算word在数据集样本中出现的次数"""
    count = 0
    for i in range(len(data)):
        if word in data[i]:
            count += 1
    return count


def count_nums(word, data):
    nums = 0
    for i in range(len(data)):
        if word in data[i]:
            nums += data[i].count(word)
    return nums


def entropy_tfidf(data, label):
    """计算每个单词的信息熵tf-idf"""
    pos = data[:np.count_nonzero(label)]
    neg = data[np.count_nonzero(label):]
    # 提取唯一单词
    words = []
    for i in range(len(data)):
        words += data[i]
    unique_words = list(set(words))         # 29227个单词

    # 计算信息熵tf-idf
    tf = []
    idf = []
    tfidf = []
    entropy = []
    for word in unique_words:
        # 计算该词的tf
        word_tf = words.count(word) / len(words)
        tf.append(word_tf)
        # 统计包含该词的句子总数
        count = count_times(word, data)
        # 计算该词的idf
        word_idf = np.log(len(data) / (1 + count))
        idf.append(word_idf)
        # tf-idf
        tfidf.append(word_tf * word_idf)
        # 该词信息熵
        prob_1 = count_nums(word, pos) / count_nums(word, data)
        prob_0 = 1 - prob_1
        if prob_0 == 1 or prob_1 == 1:
            e = 0
        else:
            e = - prob_1 * np.log(prob_1) - prob_0 * np.log(prob_0)       # 信息熵
        entropy.append(e)
    # 每个词的正则化信息熵项
    regular_entropy = [e-min(entropy)/max(entropy)-min(entropy) for e in entropy]
    # 带信息熵的tf-idf
    e_tfidf = np.array(tfidf) + np.array(regular_entropy)

    # 存入csv，方便读取
    df = pd.DataFrame({'tf': tf, 'idf': idf, 'tfidf': tfidf, 'entropy_tfidf': e_tfidf}, index=unique_words)
    df.to_csv('collect_data/tfidf_collect.csv')

    return e_tfidf


if __name__ == '__main__':
    np.seterr(invalid='ignore')
    data = np.load('collect_data/data.npy', allow_pickle=True)
    label = np.load('collect_data/labels.npy', allow_pickle=True)
    e_tfidf = entropy_tfidf(data, label)
    print(e_tfidf[:20])
