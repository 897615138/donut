# coding=utf-8
import csv

import numpy as np
import streamlit as st

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
        base_values.append(float(i[1]))
        base_labels.append(int(i[2]))
# 检查数据
# 2.转化为初始np.array
timestamp = np.array(base_timestamp)
labels = np.array(base_labels)
# 3.补充缺失时间戳(与数据)获得缺失点
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (base_values, labels))
# 4.按照比例获得训练和测试数据
test_portion = 0.3
test_amount = int(len(values) * test_portion)
train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
# 5.标准化训练和测试数据
exclude_array = np.logical_or(train_labels, train_missing)
train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)


number = st.button("click it")
st.write("返回值:", number)
# zip_array = np.random.randn(10, 2)
# zip_array = np.asarray(zip(timestamp, values))
# st.line_chart(zip_array)
#
# chart_data = pd.DataFrame(
#     np.random.randn(50, 3),
# )
# st.bar_chart(chart_data)
