# coding=utf-8
import csv

import numpy as np
import pandas as pd
import streamlit as st

from donut.preprocessing import complete_timestamp


def gain_data(file_name="sample_data/1.csv"):
    """
    1.从csv文件中获取数据 时间戳 值 异常标签
    Args:
        file_name: 文件名
    Returns:
        数据
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
    values = np.array(base_values, dtype='float64')
    return timestamp, labels, values


def fill_data(timestamp, labels, values):
    """
    3.补充缺失时间戳(与数据)获得缺失点
    Args:
        timestamp: 时间戳
        labels: 异常标志
        values: 值
    Returns: 填充后的信息与缺失点信息
    """
    timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))
    return timestamp, values, labels, missing


@st.cache
def gain_sl_cache_data(file):
    """
    从streamlit缓存获取
    Args:
        file: 缓存文件
    Returns:
        初始数据集
    """
    df = pd.read_csv(file, header=0)
    base_timestamp = df.iloc[:, 0]
    base_values = df.iloc[:, 1]
    base_labels = df.iloc[:, 2]
    timestamp = np.array(base_timestamp, dtype='int64')
    labels = np.array(base_labels, dtype='int32')
    values = np.array(base_values, dtype='float64')
    return timestamp, labels, values
