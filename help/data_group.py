#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd

data_all = pd.DataFrame(columns=["标签", "问句", "模板"])


def _read_file(filename, category):
    """
    读取一个文件
    :param filename:
    :return:
    """
    global data_all
    data = pd.read_excel(filename)  # 读取到一张表的完整数据
    data["标签"] = category[0:-5]  # 给数据添加标签
    frames = [data_all, data]
    data_all = pd.concat(frames, ignore_index=True)


def save_file(dirname):
    """
    将多个文件合并到三个文件中  train test test
    :param dirname:
    :return:
    """
    global data_all
    f_train = open(dirname + '/data_solve/data.train.txt', 'w', encoding='utf-8')
    f_test = open(dirname + '/data_solve/data.test.txt', 'w', encoding='utf-8')
    f_val = open(dirname + '/data_solve/data.val.txt', 'w', encoding='utf-8')

    file_names = os.listdir(dirname + '/data')
    for file_name in file_names:
        category = file_name
        full_name = dirname + '/data/' + file_name
        _read_file(full_name, category)
    data_all = data_all.drop(['Unnamed: 2'], axis=1)
    data_all = data_all.sample(frac=1).reset_index(drop=True)  # 打乱样本
    len_data = len(data_all)  # 数据长度

    # 分成训练集和测试集
    len_train = int(len_data * 0.8)
    for i in range(len_train):
        f_train.write(data_all.iloc[i]['标签'] + '\t' + data_all.iloc[i]['问句'] + '\n')
    f_train.close()

    for i in range(len_train,len_data):
        f_test.write(data_all.iloc[i]['标签'] + '\t' + data_all.iloc[i]['问句'] + '\n')
    f_test.close()

    for i in range(len_train,len_data):
        f_val.write(data_all.iloc[i]['标签'] + '\t' + data_all.iloc[i]['问句'] + '\n')
    f_val.close()


if __name__ == '__main__':
    save_file('D:/文本分类_TextCNN/Company')