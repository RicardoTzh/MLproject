import numpy as np
import operator as op


# KNN算法
def classify(in_x, data_set, labels, k):
    '''
    KNN方法判断某点的标签以及概率
    :in_x: 被判断点
    :data_set: 训练数据的特征矩阵,
    :labels: 训练数据的标签,是list格式
    : k: 算法的k值
    :return class_count_sort[0][0]: 判断结果,输出标签
    '''
    data_set_size = data_set.shape[0]
    in_x_copy = np.tile(in_x, (data_set_size, 1))  # 将in_x改成跟data_set一样的行列数
    # 相减，平方，第一维度上求和，开平方,求出欧氏距离的列表
    dis_list = (np.sum((data_set-in_x_copy)**2, axis=1))**(1.0/2.0)
    dis_index_sort = dis_list.argsort()# 从小到大排序，并返回原始下标
    # KNN 求出前K个值中每个标签的出现次数
    label_count = {}
    for i in range(k):
        k_label = labels[dis_index_sort[i]]
        label_count[k_label] = label_count.get(k_label, 0) + 1

    # 对次数字典进行排序，按照value的降序排列
    class_count_sort = sorted(label_count.items(), key=op.itemgetter(1), reverse=True)
    return class_count_sort[0][0]


# 数据集的归一化,输入可以为array
def auto_norm(data_set):
    '''
    归一化数据：将任意取值范围内的特征化为0-1区间的值
    公式：new_value = (current - min) / (max - min)
    :param data_set: 训练集数据
    :return: norm_data_set 归一化之后的训练集数据
    '''
    data_max = data_set.max(0)  # 获取每列的最大值，axis=1时求每行的最大值
    data_min = data_set.min(0)  # 获取每列的最小值
    cha = data_max-data_min

    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(data_min, (m,1))  # current - min
    norm_data_set = norm_data_set / np.tile(cha,(m,1))  # max - min
    return norm_data_set, data_min, cha
