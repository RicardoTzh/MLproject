import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform
from KNN_learn import data
from KNN_learn.data import KNN
import os

os.chdir(r'D:\tzh\MLproject')  # 修改工作路径


# 创建训练数据以及标签
def create_data_set():
    '''
    输出所需的训练数据
    '''
    groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return groups, labels


# 读取数据集文件并转化为矩阵
def file_to_matrix(filename):
    '''
    将文件转换到numpy数组的程序
    filename:文件名
    :return return_mat,class_label: 特征矩阵，标签数列
    '''

    with open(filename, 'r') as fl:
        array_lines = fl.readlines()
        count_file = len(array_lines)  # 得到行数

        return_mat = np.zeros((count_file, 3))  #创建初始矩阵
        class_label = []

        index = 0
        for line in array_lines:
            line = line.strip()  # 处理每一行的空格以及制表符
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]  # 将前三列赋给return_mat,第四列赋给class_label
            class_label.append(int(list_from_line[-1]))
            index += 1
    return return_mat,class_label


# 画图
def create_matplot_img(data_set, labels):
    '''
    创建散点图并展示
    :param data_set: 训练数据矩阵
    :param labels: 标签列表
    :return: 展示图片
    '''
    if platform.system() == 'Windows':
        zh_font = font_manager.FontProperties(fname='c:\Windows\Fonts\msyh.ttf')
    else:
        pass

    # 初始化数据
    type_1_x = []
    type_1_y = []
    type_2_x = []
    type_2_y = []
    type_3_x = []
    type_3_y = []

    for i in range(len(labels)):
        if labels[i] == 1:
            type_1_x.append(data_set[i][0])
            type_1_y.append(data_set[i][1])
        if labels[i] == 2:
            type_2_x.append(data_set[i][0])
            type_2_y.append(data_set[i][1])
        if labels[i] == 3:
            type_3_x.append(data_set[i][0])
            type_3_y.append(data_set[i][1])

    print(type_1_x)
    print(type_2_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #设置数据属性
    type_1 = ax.scatter(type_1_x, type_1_y, s=20, c='r',alpha=0.7)
    type_2 = ax.scatter(type_2_x, type_2_y, s=40, c='b',alpha=0.7)
    type_3 = ax.scatter(type_3_x, type_3_y, s=60, c='g',alpha=0.7)

    plt.title('约会对象数据分析',fontproperties=zh_font)
    plt.ylabel('玩游戏所消耗时间百分比',fontproperties=zh_font)
    plt.xlabel('每周消耗的冰淇淋公升数',fontproperties=zh_font)
    plt.legend((type_1,type_2,type_3),('不喜欢','魅力一般','极具魅力'), loc='upper left',prop=zh_font)
    plt.show()


# 测试函数
def dating_class_test():
    test_percent = 0.1  # 测试集的比例
    data_set, labels = file_to_matrix('dating_test_set_2.txt')  # 导入数据集
    norm_data_set = KNN.auto_norm(data_set)  # 数据归一化
    size = norm_data_set.shape[0]  # 数据集的总行数
    num_test_size = int(size*test_percent)  # 作为测试集的行数

    err_count = 0.0  # 记录错误的个数
    for i in range(num_test_size):
        classi_result = KNN.classify(norm_data_set[i, :],
                                 norm_data_set[num_test_size:size],
                                 labels[num_test_size:size], 5)  # 调用KNN算法
        print('分类器返回:%d,真实答案为:%d' % (classi_result, labels[i]))  # 打印每次KNN判断之后的结果并与真实答案对比
        if classi_result != labels[i]:  # 不相等时加一
            err_count += 1.0
    print('我们的分类器错误率为:%0.2f%%' % (err_count / float(num_test_size)*100))  # 输出错误率


# 相亲判断系统
def classify_person():
    '''
    对给定的数据进行人群分类判断
    :return: 判断结果
    '''
    kilo = float(input('每年获得的飞行常客里程数是：'))
    ice = float(input('每周消费的冰琪淋公升数是：'))
    game = float(input('玩视频游戏所耗时间百分比是：'))

    data_set, labels = file_to_matrix('data/dating_test_set_2.txt')  # 读取数据
    norm_data_set, data_min, cha = KNN.auto_norm(data_set)  # 归一化

    in_x = np.array([kilo, ice, game])  # 待验证的数据
    in_x = (in_x -data_min) / cha # 归一化
    label = KNN.classify(in_x, norm_data_set, labels, 5)  # 调用KNN分类函数

    if label == 1:
        print('放弃了！悲伤.GIF')
    elif label == 2:
        print('还是有机会的！出去转转.JPG')
    elif label == 3:
        print('咱们现在可以见面吗？开心.GIF')


if __name__ == '__main__':
    # groups, labels = create_data_set()  # 创建小型数据
    # classify([2.0, 1.0], groups, labels, 3)  # 测试KNN算法函数
    #################################################################
    # data_set, labels = file_to_matrix('dating_test_set_2.txt')  # 读取相亲训练集
    # create_matplot_img(data_set, labels)  # 画图分析数据
    # auto_norm(data_set)  # 归一化
    # dating_class_test()  # 测试KNN的函数
    #################################################################
    classify_person()  # 完整的相亲系统
