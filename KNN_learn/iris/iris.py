import numpy as np
import pandas as pd
import os
from KNN_learn.data import KNN
import time
import random

os.chdir(r'D:\tzh\MLproject\KNN_learn\iris')


# 读取鸢尾花数据并做归一化处理
def iris_data(path):
    '''
    读取鸢尾花数据并做归一化处理
    :param path: 鸢尾花数据存放文件
    :return: 鸢尾花数据的数据集，标签，长度
    '''
    iris_file = pd.read_csv(path)
    iris_data_set = iris_file.iloc[:, :4]
    iris_data_set, data_min, cha = KNN.auto_norm(iris_data_set)  # 归一化数据集
    iris_data_set = np.array(iris_data_set)
    labels = iris_file.iloc[:, -1]
    return iris_data_set, labels, len(labels)


# 留出法：一定比例作为训练集，剩余作为测试集
def load_percent(path, p):
    '''
    留出法：一定比例作为训练集，剩余作为测试集
    :param path: 鸢尾花数据存放文件
    :param k: 作为训练集的比率
    :return: 训练集的数据集，标签，长度；测试集的数据集，标签，长度
    '''
    data, labels, file_len = iris_data(path)
    data_train = []
    labels_train = []
    data_test = []
    labels_test = []
    for i in range(file_len):
        if random.random() <= p:
            data_train.append(data[i, :])
            labels_train.append(labels[i])
        else:
            data_test.append(data[i, :])
            labels_test.append(labels[i])
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    labels_len_train = len(labels_train)
    labels_len_test = len(labels_test)
    return data_train, labels_train, labels_len_train, data_test, labels_test, labels_len_test


# 测试函数并且作为完整系统
def iris_classsify_test(p):
    '''
    测试函数并且作为完整系统
    :param k: 作为训练集的比率
    :return:
    '''
    ##########取出数据
    data_train, labels_train, labels_len_train, data_test, labels_test, labels_len_test = \
        load_percent(r'D:\tzh\MLproject\KNN_learn\iris\iris_data.csv', p)
    ##########开始测试
    time_begin = time.time()
    err_count = 0.0
    for i in range(labels_len_test):
        class_result = KNN.classify(data_test[i, :],
                                    data_train,
                                    labels_train, 15)
        print('分类器返回:%s,真实答案为:%s' % (class_result, labels_test[i]))
        if class_result != labels_test[i]:
            err_count += 1.0
    print('我们的分类器错误率为:%0.2f%%' % (err_count / float(labels_len_test) * 100))  # 输出错误率)
    time_end = time.time()
    time_use = time_end - time_begin
    print('所用时间为:', time_use)


if __name__ == '__main__':
    # iris_data()
    iris_classsify_test(0.9)
