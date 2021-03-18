import time

import streamlit as st
import data
import numpy as np
import donut.demo.show_sl as sl
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
file_name = str(st.text_input('file name', "sample_data/1.csv"))
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
    sl.line_chart(base_timestamp, base_values, 'original csv_data')
    source_sum = timestamp.size
    label_num = np.sum(labels == 1)
    st.text("共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】".format(source_sum, label_num, label_num / source_sum,
                                                                  get_time(start_time, end_time)))
    start_time = time.time()
    timestamp, missing, values, labels = data.fill_data(timestamp, labels, base_values)
    end_time = time.time()
    fill_sum = timestamp.size
    sl.line_chart(timestamp.tolist(), values.tolist(), 'fill_data')
    st.text("填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
            .format(fill_sum, timestamp[1] - timestamp[0], fill_sum - base_sum, get_time(start_time, end_time)))
    start_time = time.time()
    train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp = \
        data.get_test_training_data(values, labels, missing, timestamp, test_portion)
    end_time = time.time()
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    st.text("训练数据量：{}，有{}个标注，测试数据量：{}，有{}个标注  \n【填充缺失数据,共用时{}】"
            .format(train_values.size, np.sum(train_labels == 1), test_values.size, np.sum(test_labels == 1),
                    get_time(start_time, end_time)))
    start_time = time.time()
    train_values, test_values, train_missing, train_labels, mean, std = \
        data.standardize_data(train_labels, train_missing, train_values, test_values)
    end_time = time.time()
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    st.text("平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, get_time(start_time, end_time)))
    start_time = time.time()
    test_score, epoch_list, lr_list = train_prediction(train_values, train_labels, train_missing, test_values,
                                                       test_missing, mean, std)
    end_time = time.time()
    sl.line_chart(epoch_list, lr_list, 'annealing_learning_rate')
    test_score = data.handle_test_data(test_score, test_values.size)
    sl.show_test_score(test_timestamp, test_values, test_score)
    labels_num, catch_num, catch_index, labels_index, labels_score_mean = data.label_catch(test_labels, test_score)
    st.text("默认阈值：{},根据默认阈值获得的异常点数量：{},实际异常标注数量:{}".format(labels_score_mean, catch_num, labels_num))
    # 准确度
    accuracy = labels_num / catch_num
    st.text("标签准确度:{:.2%}".format(accuracy))
    if accuracy < 1:
        a = set(catch_index)
        b = set(labels_index)
        special_anomaly_index = list(a.difference(b))
        special_anomaly_t = test_timestamp[special_anomaly_index]
        special_anomaly_s = test_score[special_anomaly_index]
        special_anomaly_v = test_values[special_anomaly_index]
        st.text("未标记但超过默认阈值的点（数量：{}）：".format(len(special_anomaly_t)))
        for i, timestamp in enumerate(special_anomaly_t):
            st.text("时间戳:{},值:{},分数：{}".format(timestamp, special_anomaly_v[i], special_anomaly_s[i]))
    st.text("【训练模型与预测获得测试分数,共用时{}】".format(get_time(start_time, end_time)))
