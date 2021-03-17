import time

import streamlit as st
import data
import numpy as np
import donut.demo.show_photo_sl as sl
from donut.demo.train_prediction import train_prediction
from donut.utils import get_time

st.title('Donut')
base_timestamp = []
base_values = []
timestamp = np.array([])
labels = np.array([])
train_timestamp = np.array([])
train_values = np.array([])
test_timestamp = np.array([])
test_values = np.array([])
train_missing = np.array([])
test_missing = np.array([])
train_labels = np.array([])
test_labels = np.array([])
mean = 0.0
std = 0.0
base_sum = 0
file_name = str(st.text_input('file name', "donut/1.csv"))
test_portion = float(st.text_input('test portion', 0.3))
button_pd = st.button("分析数据")
# button_fm = st.button("填充缺失数据")
# has_gain_data = 0
# base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
#     data.prepare_data("donut/1.csv")
# test_score = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
# sl.show_test_score(test_timestamp, test_values, test_score)

if button_pd:
    start_time = time.time()
    timestamp, labels, base_timestamp, base_values = data.gain_data(file_name)
    end_time = time.time()
    base_sum = timestamp.size
    sl.source_data(base_timestamp, base_values)
    # has_gain_data = 1
    source_sum=timestamp.size
    label_num=np.sum(labels == 1)
    st.text("共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】".format(source_sum, label_num,label_num/source_sum, get_time(start_time, end_time)))
    # if button_fm and has_gain_data:
    start_time = time.time()
    timestamp, missing, values, labels = data.fill_data(timestamp, labels, base_values)
    end_time = time.time()
    fill_sum = timestamp.size
    sl.fill_data(timestamp, values)
    st.text("填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
            .format(fill_sum, timestamp[1] - timestamp[0], fill_sum - base_sum, get_time(start_time, end_time)))
    # if st.button("按照比例获得测试与训练数据"):
    start_time = time.time()
    train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp = \
        data.get_test_training_data(values, labels, missing, timestamp, test_portion)
    end_time = time.time()
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    # prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
    #                  train_missing, test_missing)
    st.text("训练数据量：{}，测试数据量：{} \n【填充缺失数据,共用时{}】"
            .format(train_values.size, test_values.size, get_time(start_time, end_time)))
    # if st.button("标准化训练和测试数据"):
    start_time = time.time()
    train_values, test_values, train_missing, train_labels, mean, std = \
        data.standardize_data(train_labels, train_missing, train_values, test_values)
    end_time = time.time()
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    # prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp,
    #                  test_values, train_missing, test_missing)
    st.text("平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, get_time(start_time, end_time)))
    # if st.button("训练模型与预测获得测试分数"):
    start_time = time.time()
    test_scores = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
    end_time = time.time()
    sl.show_test_score(test_timestamp, test_values, test_scores)
    st.text("【训练模型与预测获得测试分数,共用时{}】".format(get_time(start_time, end_time)))




# base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
#     data.prepare_data("donut/1.csv", test_portion)
