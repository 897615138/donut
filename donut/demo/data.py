# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from donut import complete_timestamp, standardize_kpi

__all__ = ['show_photos', 'show_photo', 'plot_missing', 'prepare_data']


def show_photos(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing,
                test_missing):
    """
      原始数据与测试训练数据多图显示
    """
    plt.figure(figsize=(50, 10), dpi=1024)
    plt.figure(1)
    plt.title("prepare the data")
    ax1 = plt.subplot(211)
    ax1.plot(base_timestamp, base_values, color='r', label='original data')
    ax1.set_title("original data")

    ax2 = plt.subplot(212)
    ax2.plot(train_timestamp, train_values, label='train_data', color="y")
    ax2.plot(test_timestamp, test_values, label='test_data', color='b')
    plot_missing(ax2, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
    ax2.set_title("training and testing data")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def show_photo(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing,
               test_missing):
    """
      原始数据与测试训练数据单图显示
    """
    plt.figure(figsize=(50, 10), dpi=1024)
    plt.plot(base_timestamp, base_values, label='original data')
    plt.plot(train_timestamp, train_values, label='train_data')
    plt.plot(test_timestamp, test_values, label='test_data')
    plot_missing(plt, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
    plt.title("prepare the data")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def plot_missing(p, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values):
    """
      缺失点在图上特别标注
    """
    for i, is_missing in enumerate(train_missing):
        if 1 == is_missing:
            p.plot(train_timestamp[i], train_values[i], 'r')
    for i, is_missing in enumerate(test_missing):
        if 1 == is_missing:
            p.plot(test_timestamp[i], test_values[i], 'r')


def prepare_data(file_name, show_configure):
    """
      数据准备
      1.解析csv文件
      2.转化为初始np.array
      3.补充缺失时间戳(与数据)获得缺失点
      4.按照比例获得训练和测试数据
      5.标准化训练和测试数据
    """
    base_timestamp = []
    base_values = []
    # 默认无标签
    base_labels = []
    # 1.解析csv文件
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i in reader:
            base_timestamp.append(int(i[0]))
            base_values.append(float(i[1]))
            base_labels.append(int(i[2]))
    # 检查数据
    # 2.转化为初始np.array
    timestamp = np.array(base_timestamp, dtype='int64')
    labels = np.array(base_labels, dtype='int32')
    # 3.补充缺失时间戳(与数据)获得缺失点
    timestamp, missing, (values, labels) = complete_timestamp(timestamp, (base_values, labels))
    # 4.按照比例获得训练和测试数据
    test_portion = 0.3
    test_amount = int(len(values) * test_portion)
    train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
    train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
    train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
    train_timestamp, test_timestamp = timestamp[:-test_amount], timestamp[-test_amount:]
    # 5.标准化训练和测试数据
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    if show_configure == 1:
        show_photos(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                    train_missing, test_missing)
    else:
        show_photo(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                   train_missing, test_missing)
    return base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing,\
           train_labels,test_labels,mean,std

# prepare_data("1.csv", 1)
