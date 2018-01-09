import numpy as np
import os
from data import KNN
import time
import random

os.chdir(r'D:\tzh\MLproject\hand\digits\training_digits')


# 处理每一个手写数字的二进制文件
def image_to_vector(filename):
    '''
    传入一个像素数据点文件32X32，将其中数据放到1X1024的array中
    :param filename: 文件名
    :return: vector_one 1X1024的array
    '''

    with open(filename, 'r') as fl:
        file_read = fl.readlines()  # 读入并成为list
        vector_one = np.zeros((1, 1024))  # 生成1X1024的array
        for i in range(32):
            for j in range(32):
                vector_one[0, i*32+j] = float(file_read[i][j])

    return vector_one


#加载并处理数据
def load_file(path):
    '''
    加载数据并且处理为数据集及其标签，还有size
    :param path:
    :return: data, labels, file_len 数据集，标签， size（行数）
    '''
    file_name_list = os.listdir(path)  # 将文件夹下的文件名读入为一个list
    file_len = len(file_name_list)  # 文件个数 = 标签数量
    data = np.zeros((file_len, 1024))
    labels = []
    for i in range(file_len):
        vec_one = image_to_vector(file_name_list[i])
        data[i,:] = vec_one
        labels.append(int(file_name_list[i][0]))
    return data, labels, file_len


# 获得一定比例的训练集
def load_percent(path, k):
    data, labels, file_len = load_file(path)
    data_per = []
    labels_per = []
    for i in range(file_len):
        if random.random() <= k:
            data_per.append(data[i, :])
            labels_per.append(int(labels[i]))
    data_per = np.array(data_per)
    labels_len = len(labels_per)
    return data_per, labels_per, labels_len


# 测试函数
def hand_writing_class_test():
    '''
    测试识别手写数字分类系统的正确率
    :return:
    '''
    # 处理测试集数据
    # file_name_list = os.listdir(r'D:\tzh\MLproject\hand\digits\test_digits')  # 将文件夹下的文件名读入为一个list
    # file_len = len(file_name_list)  # 文件个数 = 标签数量
    # data_test = np.zeros((file_len, 1024))
    # labels_test = []
    # for i in range(file_len):
    #     vec_one = image_to_vector(file_name_list[i])
    #     data_test[i,:] = vec_one
    #     labels_test.append(int(file_name_list[i][0]))
    data_test, labels_test, file_len_test = load_percent(r'D:\tzh\MLproject\hand\digits\test_digits', 0.8)
    # 处理训练集数据
    # file_name_list_train = os.listdir(r'D:\tzh\MLproject\hand\digits\training_digits')
    # file_len_train = len(file_name_list_train)
    # data_train = np.zeros((file_len_train, 1024))
    # labels_train = []
    # for i in range(file_len_train):
    #     vec_one = image_to_vector(file_name_list_train[i])
    #     data_train[i, :] = vec_one
    #     labels_train.append(int(file_name_list_train[i][0]))
    data_train, labels_train, file_len_train = load_percent(r'D:\tzh\MLproject\hand\digits\training_digits', 0.8)

    # 开始测试
    time_begin = time.time()
    err_count = 0.0
    for i in range(file_len_test):
        class_result = KNN.classify(data_test[i, :],
                                    data_train,
                                    labels_train, 15)
        print('分类器返回:%d,真实答案为:%d' % (int(class_result), int(labels_test[i])))  # 输出每一次判断的结果
        if class_result != labels_test[i]:
            err_count += 1.0
    print('我们的分类器错误率为:%0.2f%%' % (err_count / float(file_len_test)*100))  # 输出错误率)
    time_end = time.time()
    time_use = time_end - time_begin
    print('所用时间为:', time_use)

if __name__ == '__main__':
    # vec_one = image_to_vector('0_0.txt')
    hand_writing_class_test()  # 调用测试函数
