import feedparser
import numpy as np
import re
import random
import operator


# 文本处理
def text_parse(big_string):
    """
    文本处理器
    :param big_string: 待处理文本
    :return: 处理完成文本词汇列表
    """
    words_list = [word for word in re.compile('\\W+').split(big_string) if len(word) > 0]
    return words_list


# 创建词汇表
def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现且不重复的词汇表
    :param data_set: 待处理文本数据集
    :return: 不重复的词汇列表
    """
    vocab_set = set([])  # 创建一个空集
    for doc in data_set:
        vocab_set = vocab_set | set(doc)  # 创建两个集合的并集
    return list(vocab_set)


# 词袋模型
def bag_of_words_to_vec(vocab_list, input_set):
    """
    将目标文档转化为向量：词袋模型（bag of words model）
    :param vocab_list:词汇表
    :param input_set:待处理文档
    :return:转化后的文档向量
    """
    return_vec = [0] * len(vocab_list)  # 创建一个所含元素都为零的列表
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb(train_data_set, train_category):
    """
    训练朴素贝叶斯算法模型
    :param train_data_set: 训练集
    :param train_category: 分类向量
    :return: None
    """
    num_train_docs = len(train_data_set)
    num_words = len(train_data_set[0])
    print(num_train_docs, num_words)
    p_abusive = sum(train_category) / float(num_train_docs)  # 标注侮辱性文档的概率

    p_0_num = np.ones(num_words)  # 正常概率向量
    p_1_num = np.ones(num_words)  # 侮辱性概率向量

    p_0_denom = 2.0
    p_1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 0:  # 正常文档
            p_0_num += train_data_set[i]
            p_0_denom += sum(train_data_set[i])
        else:
            p_1_num += train_data_set[i]
            p_1_denom += sum(train_data_set[i])

    p_0_vect = np.log(p_0_num / p_0_denom)
    p_1_vect = np.log(p_1_num / p_1_denom)

    return p_0_vect, p_1_vect, p_abusive


def classify_nb(vec_to_classify, p_0_vect, p_1_vect, p_class):
    """
    朴素贝叶斯分类器
    :param vec_to_classify:待分类向量
    :param p_0_vect: p(0)向量
    :param p_1_vect: p(1)向量
    :param p_class: 类别概率
    :return: None
    """
    p_0 = sum(vec_to_classify * p_0_vect) + np.log(p_class)
    p_1 = sum(vec_to_classify * p_1_vect) + np.log(1.0 - p_class)

    if p_0 > p_1:
        return 0
    else:
        return 1


# 计算出现频率最多的词汇
def calc_most_freq(vocab_list, full_text, most_count=30):
    '''
    计算出现频率最多的词汇
    :param vocab_list: 不重复词汇表
    :param full_text: 文本全部词汇表
    :param most_count: 最多移除数量
    :return: 移除目标高频词汇后的词表
    '''
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_freq[:most_count]


# 测试函数 同时作为词汇表 概率
def local_words(feed_0, feed_1):
    '''
    测试函数
    :param feed_0:
    :param feed_1:
    :return:
    '''
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed_0['entries']), len(feed_1['entries']))
    for i in range(min_len):
        word_list = text_parse(feed_0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

        word_list = text_parse(feed_1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

    vocab_list = create_vocab_list(doc_list)
    top_words = calc_most_freq(vocab_list=vocab_list, full_text=full_text, most_count=20)
    for pair_w in top_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    training_set = list(range(2 * min_len))
    test_set = []

    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    training_mat = []
    training_classes = []
    #生成训练集
    for doc_index in training_set:
        training_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p_0_v, p_1_v, p_class = train_nb(np.array(training_mat), np.array(training_classes))  # 训练模型
    err_count = 0.0
    for doc_index in test_set:
        word_vector = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(np.array(word_vector), p_0_v, p_1_v, p_class) != class_list[doc_index]:
            err_count += 1.0
    print('错误率是:', float(err_count) / len(test_set))
    return vocab_list, p_0_v, p_1_v


# 得到每个停用词及其概率
def get_top_words(feed_ny, feed_sf):
    vocab_list, p_0_v, p_1_v = local_words(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p_0_v)):
        if p_0_v[i] > -5.0:
            top_ny.append((vocab_list[i], p_0_v[i]))
        if p_1_v[i] > -5.0:
            top_sf.append((vocab_list[i], p_1_v[i]))
    sorted_ny = sorted(top_ny, key=lambda x: x[1], reverse=True)
    sorted_sf = sorted(top_sf, key=lambda x: x[1], reverse=True)

    print('NY **' * 20)
    for item in sorted_ny:
        print(item)

    print('SF **' * 20)
    for item in sorted_sf:
        print(item)


if __name__ == '__main__':
    ny_rss = 'https://newyork.craigslist.org/search/stp?format=rss'
    sf_rss = 'https://sfbay.craigslist.org/search/stp?format=rss'

    ny = feedparser.parse(ny_rss)
    sf = feedparser.parse(sf_rss)
    if len(ny['entries']) * len(sf['entries']) == 0:
        print('获取数据出错')
    else:
        get_top_words(ny, sf)
