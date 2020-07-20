# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/16 21:37"
__doc__ = """ 对训练集进行训练"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import jieba
from data_preview import remove_marks
from fea_extract import get_train_vec, get_label, sentence_vec


def predict_raw_sentence(queries, model):
    """
    :param queries: 句子集合 <class list>
    :param model: 用来预测的模型
    :return: 预测结果 <class list>
    """
    prec = []
    queries = np.array([jieba.lcut(query) for query in queries])
    queries = remove_marks(queries)
    for query in queries:
        words_vec = sentence_vec(query)
        result = model.predict(words_vec)
        if int(result) == 1:
            prec.append('正面')
        elif int(result) == 0:
            prec.append('负面')
    return prec


if __name__ == '__main__':
    # 获取词向量训练集和标签
    data_vec = get_train_vec()
    data_label = get_label()
    pos = data_vec[:np.count_nonzero(data_label), :]
    neg = data_vec[np.count_nonzero(data_label):, :]

    """样本总数7766(5322 + 2444)，原始数据集正负样本比例约为2：1
    10折交叉验证
    训练集(6990=4660 + 2330)90%; 测试集(776=662 + 114)(10%)
    由于数据集本身有序，故分别从正负样本集中抽取"""
    folds = 10
    pos_data = np.array_split(pos, folds)
    neg_data = np.array_split(neg, folds)
    for i in range(folds):
        test_data = np.concatenate([pos_data[i], neg_data[i]])
        test_label = np.concatenate([np.ones(pos_data[i].shape[0]), np.zeros(neg_data[i].shape[0])])

        pos_train = np.concatenate(pos_data[:i] + pos_data[i+1:])
        neg_train = np.concatenate(neg_data[:i] + neg_data[i+1:])
        train_data = np.concatenate([pos_train, neg_train])
        train_label = np.concatenate([np.ones(pos_train.shape[0]), np.zeros(neg_train.shape[0])])

        # 训练一个svm模型，并将训练好的模型保存起来
        svm_model = SVC(kernel='rbf', verbose=True)
        svm_model.fit(train_data, train_label)
        # joblib.dump(svm_model, 'comments_svm.pkl')

        """对原始输入进行预测"""
        # queries = ['价格偏高,服务一般.窗外风景ok.早餐还不错', '卫生比较差，服务态度不好', '酒店房间小，设施陈旧，酒店靠马路，窗户很不隔音，影响休息']
        # prec = predict_raw_sentence(queries, svm_model)
        # print(prec)

        """对验证集进行预测"""
        # TP, FP, FN, TN = 0, 0, 0, 0
        # pred = svm_model.predict(test_data)
        # for j in range(len(test_label)):
        #     if pred[j] == 1 and test_label[j] == 1:
        #         TP += 1
        #     elif pred[j] == 1 and test_label[j] == 0:
        #         FP += 1
        #     elif pred[j] == 0 and test_label[j] == 1:
        #         FN += 1
        #     elif pred[j] == 0 and test_label[j] == 0:
        #         TN += 1
        # acc = (TP + TN) / len(test_label)
        # precision = TP / (TP+FP)
        # recall = TP / (TP+FN)
        # F1 = 2*precision*recall / (precision+recall)
        # print("第 %d 次预测" % i)
        # print('精度:', acc, 'P值:', precision, 'R值:', recall, 'F1:', F1)

        """对测试集进行预测"""
        random_pos = np.random.randint(0, 5322, size=80)
        random_neg = np.random.randint(5323, 7766, size=40)
        r_index = np.hstack((random_pos, random_neg))
        r_label = np.array([1]*80 + [0]*40)
        random_data = data_vec[list(r_index), :]

        r_pred = svm_model.predict(random_data)
        r_TP, r_FP, r_FN, r_TN = 0, 0, 0, 0
        for j in range(len(r_label)):
            if r_pred[j] == 1 and r_label[j] == 1:
                r_TP += 1
            elif r_pred[j] == 1 and r_label[j] == 0:
                r_FP += 1
            elif r_pred[j] == 0 and r_label[j] == 1:
                r_FN += 1
            elif r_pred[j] == 0 and r_label[j] == 0:
                r_TN += 1
        acc = (r_TP + r_TN) / len(r_label)
        precision = r_TP / (r_TP + r_FP)
        recall = r_TP / (r_TP + r_FN)
        F1 = 2 * precision * recall / (precision + recall)
        print("第 %d 次预测" % i)
        print('精度:', acc, 'P值:', precision, 'R值:', recall, 'F1:', F1)

