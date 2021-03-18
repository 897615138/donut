# coding=utf-8
import csv
import numpy as np
import donut.demo.show_sl as sl
from donut import complete_timestamp, standardize_kpi

__all__ = ['prepare_data']


def prepare_data(file_name, test_portion=0.3):
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
    test_amount = int(len(values) * test_portion)
    train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
    train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
    train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
    train_timestamp, test_timestamp = timestamp[:-test_amount], timestamp[-test_amount:]
    # 5.标准化训练和测试数据
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    return base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, \
           train_labels, test_labels, mean, std


def gain_data(file_name="sample_data/1.csv"):
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
    return timestamp, labels, base_timestamp, base_values


def fill_data(timestamp, labels, base_values):
    # 3.补充缺失时间戳(与数据)获得缺失点
    timestamp, missing, (values, labels) = complete_timestamp(timestamp, (base_values, labels))
    return timestamp, missing, values, labels


def get_test_training_data(values, labels, missing, timestamp, test_portion=0.3):
    test_amount = int(len(values) * test_portion)
    train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
    train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
    train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
    train_timestamp, test_timestamp = timestamp[:-test_amount], timestamp[-test_amount:]
    return train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp


def standardize_data(train_labels, train_missing, train_values, test_values):
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    return train_values, test_values, train_missing, train_labels, mean, std


def handle_test_data(test_score, test_num):
    # 因为对于每个窗口的检测实际返回的是最后一个窗口的 score，也就是说第一个窗口的前面一部分的点都没有检测，默认为正常数据。因此需要在检测结果前面补零或者测试数据的真实 label。
    test_score = np.pad(test_score, (test_num - test_score.size, 0), 'constant', constant_values=(0, 0))
    test_score = 0 - test_score
    return test_score


def label_catch(test_labels, test_score):
    labels_index = np.where(test_labels == 1)[0].tolist()
    labels_score = test_score[labels_index]
    labels_score_mean=np.mean(labels_score)
    sl.text(labels_score)
    sl.text(labels_score_mean)
    # labels_score_max = np.max(labels_score)
    # labels_score_min = np.min(labels_score)
    catch_index = np.where(test_score > labels_score_mean)[0].tolist()
    catch_num = np.size(catch_index)
    labels_num = np.size(labels_index)
    return labels_num, catch_num, catch_index, labels_index, labels_score_mean
