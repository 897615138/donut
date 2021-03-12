# coding=utf-8
import csv

import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi

base_timestamp = []
base_values = []
# 默认无标签
base_labels = []
# 1.解析csv文件
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for i in reader:
        base_timestamp.append(i[0])
        base_values.append(i[1])
        base_labels.append(i[2])
# 检查数据
# 2.转化为初始np.array
timestamp = np.array(base_timestamp)
labels = np.array(base_labels)
# 3.补充缺失时间戳(与数据)获得缺失点
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (base_values, labels))
# 4.区分训练和测试数据
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
