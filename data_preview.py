# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/16 12:08"
__doc__ = """ 数据预处理：填充缺失值，写入data,label文件"""

import numpy as np
import pandas as pd
# import thulac
import jieba


def remove_marks(data):
    """去除标点符号
    :param data: 分词之后句子的集合 <class ndarray>
    :return 去除标点符号之后的集合"""
    data = list(data)
    # 创建停用词表
    marks = r"1234567890[\s+\.\!\/_,$%^*())?;；:-【】+\"\']+|[+——！！，;:。？、~@#￥%……&*（）]+ ，。"
    stopwords = []
    f = open('collect_data/stopwords.txt', 'r', encoding='utf-8')
    line = f.readline()[:-1]
    while line:
        stopwords.append(line)
        line = f.readline()[:-1]
    f.close()
    all_stopwords = list(marks) + stopwords
    # 去停用词
    for i in range(len(data)):
        new = [ele for ele in data[i] if ele not in all_stopwords]
        data[i] = new
    return data


csv_path = r'text_resource\ChineseNlpCorpus-master\datasets\ChnSentiCorp_htl_all\ChnSentiCorp_htl_all.csv'
comments_df = pd.read_csv(csv_path, encoding='utf-8')

"""检查缺失值并按照标签填补缺失值"""
nan_df = comments_df[comments_df.isnull().values == True]
nan_index = nan_df.index
for i in nan_index:
    if comments_df.loc[i, 'label'] == 0:
        comments_df.loc[i, 'review'] = '非常不好'
    comments_df.loc[i, 'review'] = '非常好'
# print(comments_df.isnull().any())

"""区分正负语料"""
pos = comments_df[comments_df['label'] == 1]
neg = comments_df[comments_df['label'] == 0]

"""对正负语料进行分词"""
pos = np.array([jieba.lcut(pos_cut) for pos_cut in pos.values[:, -1]])
neg = np.array([jieba.lcut(neg_cut) for neg_cut in neg.values[:, -1]])


"""写入数据，方便加载"""
data = np.concatenate((pos, neg), axis=0)
data = remove_marks(data)
labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
np.save('collect_data/data.npy', data)
np.save('collect_data/labels.npy', labels)


