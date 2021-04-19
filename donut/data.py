# coding=utf-8
import csv

import numpy as np
import pandas as pd
import streamlit as st

from donut.cache import gain_data_cache, save_data_cache
from donut.preprocessing import standardize_kpi, complete_timestamp
from donut.threshold import catch_label_v1, catch_label_v2
from donut.train_prediction import train_prediction_v1, train_prediction_v2
from donut.util.out.out import print_info, show_line_chart, print_text, show_prepare_data_one, print_warn, \
    show_test_score
from donut.util.time_util import TimeCounter, get_constant_timestamp, TimeUse


def prepare_data(file_name, test_portion=0.3):
    """
      数据准备
      1.解析csv文件
      2.转化为初始np.array
      3.补充缺失时间戳(与数据)获得缺失点
      4.按照比例获得训练和测试数据
      5.标准化训练和测试数据
    """
    src_timestamps = []
    base_values = []
    # 默认无标签
    base_labels = []
    # 1.解析csv文件
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i in reader:
            src_timestamps.append(int(i[0]))
            base_values.append(float(i[1]))
            base_labels.append(int(i[2]))
    # 检查数据
    # 2.转化为初始np.array
    timestamp = np.array(src_timestamps, dtype='int64')
    labels = np.array(base_labels, dtype='int32')
    # 3.补充缺失时间戳(与数据)获得缺失点
    timestamp, missing, (values, labels) = complete_timestamp(timestamp, (base_values, labels))
    # 4.按照比例获得训练和测试数据
    test_amount = int(len(values) * test_portion)
    train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
    train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
    train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
    train_timestamps, test_timestamps = timestamp[:-test_amount], timestamp[-test_amount:]
    # 5.标准化训练和测试数据
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, train_mean, train_std = standardize_kpi(train_values,
                                                          excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=train_mean, std=train_std)
    return src_timestamps, base_values, train_timestamps, train_values, test_timestamps, test_values, train_missing, test_missing, \
           train_labels, test_labels, train_mean, train_std


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


def get_test_training_data(fill_values, fill_labels, src_misses, fill_timestamps, test_portion=0.3):
    """
    获得测试与训练数据集
    Args:
        fill_values: 值数据集
        fill_labels: 异常标识数据集
        src_misses: 缺失点数据集
        fill_timestamps: 时间戳数据集
        test_portion (float):验证数据与所有指定的训练数据之比。(default 0.3)
    Returns:
        获得测试与训练数据集
    """
    test_amount = int(len(fill_values) * test_portion)
    train_values, test_values = np.asarray(fill_values[:-test_amount]), np.asarray(fill_values[-test_amount:])
    train_labels, test_labels = fill_labels[:-test_amount], fill_labels[-test_amount:]
    train_missing, test_missing = src_misses[:-test_amount], src_misses[-test_amount:]
    train_timestamp, test_timestamp = fill_timestamps[:-test_amount], fill_timestamps[-test_amount:]
    return train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp


def standardize_data_v1(train_labels, train_missing, train_values, test_values):
    """
    标准化数据
    Args:
        train_labels: 训练数据的异常标识
        train_missing: 训练数据的缺失点数据
        train_values: 训练数据的值数据集
        test_values: 测试数据的值数据集
    Returns:
        标准化数据
    """
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, train_mean, train_std = standardize_kpi(train_values,
                                                          excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=train_mean, std=train_std)
    return train_values, test_values, train_labels, train_mean, train_std


def standardize_data_v2(mean, std, fill_train_values, fill_train_labels, train_missing, fill_test_values):
    """
    标准化数据
    Args:
        std: 标准差
        mean: 平均值
        fill_train_labels: 训练数据的异常标识
        train_missing: 训练数据的缺失点数据
        fill_train_values: 训练数据的值数据集
        fill_test_values: 测试数据的值数据集
    Returns:
        标准化数据
    """
    exclude_array = np.logical_or(fill_train_labels, train_missing)
    train_values, _, _ = standardize_kpi(fill_train_values, mean=mean, std=std,
                                         excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(fill_test_values, mean=mean, std=std)
    return train_values, test_values


def handle_refactor_probability_v1(refactor_probability, data_num):
    """
    处理测试数据分数结果，补零，倒置去小于0
    Args:
        refactor_probability: 重构概率
        data_num: 数据数量

    Returns:
        处理过的测试分数，补上的0的数量
    """
    # 因为对于每个窗口的检测实际返回的是最后一个窗口的重建概率，
    # 也就是说第一个窗口的前面一部分的点都没有检测，默认为正常数据。
    # 因此需要在检测结果前面补零或者测试数据的真实 label。
    zero_num = data_num - refactor_probability.size
    refactor_probability = np.pad(refactor_probability, (zero_num, 0), 'constant', constant_values=(0, 0))
    refactor_probability = 0 - refactor_probability
    refactor_probability = np.where(refactor_probability < 0, refactor_probability, 0)
    return refactor_probability, zero_num


def handle_refactor_probability_v2(refactor_probability, data_num, timestamps, value, label, missing):
    """
    处理测试数据分数结果
    1.得出默认为正常数据的点数
    2.截取实际测试数据 相关
    3.重构概率负数，获得分数

    Args:
        refactor_probability: 重构概率
        data_num: 数据数量

    Returns:
        处理过的测试分数，补上的0的数量
    """
    # 因为对于每个窗口的检测实际返回的是最后一个窗口的重建概率，
    # 也就是说第一个窗口的前面一部分的点都没有检测，默认为正常数据。
    # 因此需要在检测结果前面补零或者测试数据的真实 label。
    zero_num = data_num - refactor_probability.size
    timestamps = timestamps[zero_num:np.size(timestamps)]
    value = value[zero_num:np.size(value)]
    label = label[zero_num:np.size(label)]
    missing = missing[zero_num:np.size(missing)]
    refactor_probability = 0 - refactor_probability
    return refactor_probability, zero_num, timestamps, value, label, missing


def show_cache_data(use_plt, file_name, test_portion, src_threshold_value, is_local):
    """
    展示缓存数据
    Args:
        is_local: 本地照片显示
        use_plt: 显示方式
        file_name: 数据文件名
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值
    """
    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
    fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
    train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
    train_mean, train_std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
    accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s, \
    test_timestamps, test_values, test_scores, model_time, trainer_time, predictor_time, fit_time, probability_time \
        , threshold_value, train_message, train_timestamps, train_values, t_use, t_name, src_train_values, src_test_values \
        = gain_data_cache(use_plt, file_name, test_portion, src_threshold_value, is_local)

    print_info(use_plt, "1.分析csv数据【共用时{}】".format(first_time))
    show_line_chart(use_plt, src_timestamps, src_values, 'original csv data')
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}".format(src_data_num, src_label_num, src_label_proportion))

    print_info(use_plt, "2.填充数据，【共用时{}】".format(second_time))
    show_line_chart(use_plt, fill_timestamps, fill_values, 'filled data')
    print_text(use_plt, "填充至{}条数据，时间戳步长:{},补充{}个时间戳数据".format(fill_data_num, fill_step, fill_num))

    print_info(use_plt, "3.获得训练与测试数据集【共用时{}】".format(third_time))
    print_text(use_plt, "训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                        "测试数据量：{}，有{}个标注,标签比例约为{:.2%}"
               .format(train_data_num, train_label_num, train_label_proportion,
                       test_data_num, test_label_num, test_label_proportion))
    # 显示测试与训练数据集
    show_prepare_data_one \
        (use_plt, "original and training and test data", "original data", "training data", "test_data", src_timestamps,
         src_values, train_timestamps, src_train_values, test_timestamps, src_test_values)
    print_info(use_plt, "4.标准化训练和测试数据【共用时{}】".format(forth_time))
    print_text(use_plt, "平均值：{}，标准差：{}".format(train_mean, train_std))
    # 显示标准化后的数据
    show_prepare_data_one \
        (use_plt, "standard data", "original data", "training data", "test_data", src_timestamps, src_values,
         train_timestamps,
         train_values, test_timestamps, test_values)
    print_info(use_plt, "5.构建Donut模型【共用时{}】\n6.构建训练器【共用时{}】\n7.构造预测器【共用时{}】\n"
               .format(model_time, trainer_time, predictor_time))
    for text in train_message:
        print_text(use_plt, text)
    print_info(use_plt, "8.训练器训练模型【共用时{}】".format(fit_time))
    print_text(use_plt, "所有epoch【共用时：{}】".format(epoch_time))
    print_text(use_plt, "退火学习率 学习率随epoch变化")
    show_line_chart(use_plt, epoch_list, lr_list, 'annealing learning rate')
    print_info(use_plt, "9.预测器获取重构概率【共用时{}】".format(probability_time))
    show_test_score(use_plt, test_timestamps, test_values, test_scores)
    print_text(use_plt, "训练模型与预测获得测试分数【共用时{}】".format(epoch_time, fifth_time))
    print_text(use_plt, "阈值：{},根据阈值获得的异常点数量：{},实际异常标注数量:{}".format(threshold_value, catch_num, labels_num))
    if accuracy is not None:
        print_text(use_plt, "标签准确度:{:.2%}".format(accuracy))
    print_text(use_plt,
               "未标记但超过阈值的点（数量：{}）：\n 共有{}段(处)异常 \n {}".format(special_anomaly_num, interval_num, interval_str))
    for i, use in enumerate(special_anomaly_t):
        print_text(use_plt, "时间戳:{},值:{},分数：{}".format(use, special_anomaly_v[i], special_anomaly_s[i]))
    print_info(use_plt, "用时排名正序")
    for i, use in enumerate(t_use):
        print_text(use_plt, "第{}：{}用时{}".format(i + 1, t_name[i], use))


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


def get_info_from_file(is_upload, is_local, file):
    try:
        if is_upload:
            src_timestamps, src_labels, src_values = gain_sl_cache_data(file)
        elif is_local:
            src_timestamps, src_labels, src_values = gain_data("../sample_data/" + file)
            a = file.split("/")
            file = a[len(a) - 1]
        else:
            src_timestamps, src_labels, src_values = gain_data("sample_data/" + file)
        return src_timestamps, src_labels, src_values, file, True
    except Exception:
        return None, None, None, None, False


def show_new_data(use_plt, file, test_portion, src_threshold_value, is_upload, is_local):
    """
    非缓存运行
    Args:
        is_local: 本地图片展示
        is_upload: 是否为上传文件
        use_plt: 展示方式使用plt？
        file: 文件名
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值
    """
    tc = TimeCounter()
    tc.start()
    src_timestamps, src_labels, src_values, file, success = get_info_from_file(is_upload, is_local, file)
    tc.end()
    if not success:
        print_warn(use_plt, "找不到数据文件，请检查文件名与路径")
        return
        # 原数据数量
    src_data_num = src_timestamps.size
    # 原数据标签数
    src_label_num = np.sum(src_labels == 1)
    # 原数据标签占比
    src_label_proportion = src_label_num / src_data_num
    first_time = tc.get_s() + "秒"
    # 显示原始数据信息
    print_info(use_plt, "1.分析csv数据【共用时{}】".format(first_time))
    show_line_chart(use_plt, src_timestamps, src_values, 'original csv data')
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}".format(src_data_num, src_label_num, src_label_proportion))
    tc.start()
    # 处理时间戳 获得缺失点
    fill_timestamps, fill_values, fill_labels, src_misses = fill_data(src_timestamps, src_labels, src_values)
    tc.end()
    fill_data_num = fill_timestamps.size
    fill_num = fill_data_num - src_data_num
    # 显示填充信息
    fill_step = fill_timestamps[1] - fill_timestamps[0]
    second_time = tc.get_s() + "秒"
    print_info(use_plt, "2.填充数据，【共用时{}】".format(second_time))
    show_line_chart(use_plt, fill_timestamps, fill_values, 'filled data')
    print_text(use_plt, "填充至{}条数据，时间戳步长:{},补充{}个时间戳数据".format(fill_data_num, fill_step, fill_num))
    # 获得测试与训练数据集
    tc.start()
    src_train_values, src_test_values, train_labels, test_labels, train_missing, test_missing, train_timestamps, test_timestamps = \
        get_test_training_data(fill_values, fill_labels, src_misses, fill_timestamps, test_portion)
    tc.end()
    third_time = tc.get_s() + "秒"
    train_data_num = src_train_values.size
    train_label_num = np.sum(train_labels == 1)
    train_label_proportion = train_label_num / train_data_num
    test_data_num = src_test_values.size
    test_label_num = np.sum(test_labels == 1)
    test_label_proportion = test_label_num / test_data_num
    print_info(use_plt, "3.获得训练与测试数据集【共用时{}】".format(third_time))
    print_text(use_plt, "训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                        "测试数据量：{}，有{}个标注,标签比例约为{:.2%}"
               .format(train_data_num, train_label_num, train_label_proportion,
                       test_data_num, test_label_num, test_label_proportion))
    # 显示测试与训练数据集
    # 显示测试与训练数据集
    show_prepare_data_one \
        (use_plt, "original and training and test data", "original data", "training data", "test_data", src_timestamps,
         src_values, train_timestamps, src_train_values, test_timestamps, src_test_values)
    # 标准化数据
    tc.start()
    train_values, test_values, train_labels, train_mean, train_std = \
        standardize_data_v1(train_labels, train_missing, src_train_values, src_test_values)
    tc.end()
    forth_time = tc.get_s() + "秒"
    print_info(use_plt, "4.标准化训练和测试数据【共用时{}】".format(forth_time))
    print_text(use_plt, "平均值：{}，标准差：{}".format(train_mean, train_std))
    # 显示标准化后的数据
    show_prepare_data_one \
        (use_plt, "standard data", "original data", "training data", "test_data", src_timestamps, src_values,
         train_timestamps,
         train_values, test_timestamps, test_values)
    # 进行训练，预测，获得重构概率
    tc.start()
    refactor_probability, epoch_list, lr_list, epoch_time, model_time, trainer_time, predictor_time, fit_time, probability_time, train_message = \
        train_prediction_v1(use_plt, train_values, train_labels, train_missing, test_values, test_missing, test_labels,
                            train_mean, train_std, test_data_num)
    tc.end()
    fifth_time = tc.get_s() + "秒"
    # 获得重构概率对应分数
    test_scores, zero_num = handle_refactor_probability_v1(refactor_probability, test_values.size)
    show_test_score(use_plt, test_timestamps, test_values, test_scores)
    print_text(use_plt, "训练模型与预测获得测试分数【共用时{}】".format(epoch_time, fifth_time))
    # 根据分数捕获异常 获得阈值
    labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy = \
        catch_label_v1(use_plt, test_labels, test_scores, zero_num, src_threshold_value)
    print_text(use_plt, "阈值：{},根据阈值获得的异常点数量：{},实际异常标注数量:{}".format(threshold_value, catch_num, labels_num))
    # 如果有标签准确率 显示
    if accuracy is not None:
        print_text(use_plt, "标签准确度:{:.2%}".format(accuracy))
        # 比较异常标注与捕捉的异常的信息
    special_anomaly_index = list(set(catch_index) - set(labels_index))
    special_anomaly_t = test_timestamps[special_anomaly_index]
    special_anomaly_s = test_scores[special_anomaly_index]
    special_anomaly_v = test_values[special_anomaly_index]
    special_anomaly_num = len(special_anomaly_t)
    interval_num, interval_str = get_constant_timestamp(special_anomaly_t, fill_step)
    print_text(use_plt, "未标记但超过阈值的点（数量：{}）：\n 共有{}段连续异常 \n ".format(special_anomaly_num, interval_num))
    if special_anomaly_num is not 0:
        print_text(use_plt, interval_str)
    for i, t in enumerate(special_anomaly_t):
        print_text(use_plt, "时间戳:{},值:{},分数：{}".format(t, special_anomaly_v[i], special_anomaly_s[i]))
    # 比较用时时间
    time_list = [TimeUse(first_time, "1.分析csv数据"), TimeUse(second_time, "2.填充数据"), TimeUse(third_time, "3.获得训练与测试数据集"),
                 TimeUse(forth_time, "4.标准化训练和测试数据"), TimeUse(model_time, "5.构建Donut模型"),
                 TimeUse(trainer_time, "6.构造训练器"), TimeUse(predictor_time, "7.构造预测器"),
                 TimeUse(fit_time, "8.训练模型"), TimeUse(probability_time, "9.获得重构概率")]
    time_list = np.array(time_list)
    sorted_time_list = sorted(time_list)
    t_use = []
    t_name = []
    print_info(use_plt, "用时排名正序")
    for i, t in enumerate(sorted_time_list):
        print_text(use_plt, "第{}：{}用时{}".format(i + 1, t.name, t.use))
        t_use.append(t.use)
        t_name.append(t.name)
    save_data_cache(use_plt, is_local, file, test_portion, src_threshold_value,
                    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion,
                    first_time, fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time,
                    third_time, train_data_num, train_label_num, train_label_proportion, test_data_num,
                    test_label_num, test_label_proportion, train_mean, train_std, forth_time, epoch_list, lr_list,
                    epoch_time, fifth_time, catch_num, labels_num, accuracy, special_anomaly_num, interval_num,
                    interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s, test_timestamps, test_values,
                    test_scores, model_time, trainer_time, predictor_time, fit_time, probability_time, threshold_value,
                    train_message, train_timestamps, train_values, t_use, t_name, src_train_values, src_test_values)


def standardize_train_data(fill_train_labels, train_missing, src_train_values):
    exclude_array = np.logical_or(fill_train_labels, train_missing)
    train_values, train_mean, train_std = standardize_kpi(src_train_values,
                                                          excludes=np.asarray(exclude_array, dtype='bool'))
    return train_values, train_mean, train_std


def check(use_plt, train_timestamp, test_timestamp, src_train_values, src_train_labels, src_test_values,
          src_test_labels):
    # 一维数组检验
    if len(train_timestamp.shape) != 1 or len(test_timestamp.shape) != 1:
        print_warn(use_plt, '`train_timestamp`必须为一维数组')
        return
    train_arrays = (src_train_values, src_train_labels)
    # np arrays-> arrays
    src_train_arrays = [np.asarray(src_array) for src_array in train_arrays]
    # 相同维度
    for i, src_array in enumerate(src_train_arrays):
        if src_array.shape != train_timestamp.shape:
            print_warn(use_plt, '`train_timestamp` 的形状必须与`src_array`的形状相同 ({} vs {}) src_array index {}'
                       .format(train_timestamp.shape, src_array.shape, i))
            return
    # src_test_arrays = [np.asarray(src_array) for src_array in train_arrays]
    # np arrays-> arrays
    test_arrays = (src_test_values, src_test_labels)
    src_test_arrays = [np.asarray(src_array) for src_array in test_arrays]
    # 相同维度
    for i, src_array in enumerate(src_test_arrays):
        if src_array.shape != test_timestamp.shape:
            print_warn(use_plt, '`test_timestamp` 的形状必须与`src_array`的形状相同 ({} vs {}) src_array index {}'
                       .format(test_timestamp.shape, src_array.shape, i))
    # src_train_arrays = [np.asarray(src_array) for src_array in train_arrays]


def merge_data(use_plt, src_train_timestamps, src_train_labels, src_train_values, src_test_timestamps, src_test_labels,
               src_test_values):
    # 1.检验数据合法性
    # np src_array-> src_array
    train_timestamp = np.asarray(src_train_timestamps, np.int64)
    test_timestamp = np.asarray(src_test_timestamps, np.int64)
    check(use_plt, train_timestamp, test_timestamp, src_train_values, src_train_labels, src_test_values,
          src_test_labels)
    # 2.检验时间戳数据 补充为有序等间隔时间戳数组
    # 时间戳排序 获得对数组排序后的原数组的对应索引以及有序数组
    src_train_index = np.argsort(train_timestamp)
    src_test_index = np.argsort(test_timestamp)
    train_timestamp_sorted = train_timestamp[src_train_index]
    test_timestamp_sorted = test_timestamp[src_test_index]
    train_values_sorted = src_train_values[src_train_index]
    test_values_sorted = src_test_values[src_test_index]
    train_labels_sorted = src_train_labels[src_train_index]
    test_labels_sorted = src_test_labels[src_test_index]
    # 沿给定轴计算离散差分 即获得所有存在的时间戳间隔
    train_intervals = np.unique(np.diff(train_timestamp_sorted))
    test_intervals = np.unique(np.diff(test_timestamp_sorted))
    # 最小的间隔数
    train_min_interval = np.min(train_intervals)
    test_min_interval = np.min(test_intervals)
    # 单独有重复
    if train_min_interval <= 0:
        repeat_timestamp = set(train_timestamp_sorted - np.unique(train_timestamp_sorted))
        repeat_timestamp_str = ""
        for t in repeat_timestamp:
            repeat_timestamp_str = repeat_timestamp_str + " " + str(t)
        print_warn(use_plt, '训练数据重复时间戳:\n' + repeat_timestamp_str)
        return
    if test_min_interval <= 0:
        repeat_timestamp = set(test_timestamp_sorted - np.unique(test_timestamp_sorted))
        repeat_timestamp_str = ""
        for t in repeat_timestamp:
            repeat_timestamp_str = repeat_timestamp_str + " " + str(t)
        print_warn(use_plt, '测试数据重复时间戳:\n' + repeat_timestamp_str)
        return
    # 合并有重复时检查
    merge_set = set(train_timestamp_sorted).intersection(set(test_timestamp_sorted))
    if len(merge_set) != 0:
        for i in merge_set:
            # 寻找指定数值的索引
            train_index = np.where(train_timestamp_sorted == i)
            test_index = np.where(test_timestamp_sorted == i)
            if train_values_sorted[train_index] != test_values_sorted[test_index]:
                print_warn(use_plt, "训练与测试数据中相同时间戳有不同KPI值，时间戳：{}，训练数据：{}，测试数据：{}"
                           .format(i, train_values_sorted[train_index], test_values_sorted[test_index]))
                return
            if train_labels_sorted[train_index] != test_labels_sorted[test_index]:
                print_warn(use_plt, "训练与测试数据中相同时间戳有不同标签，时间戳：{}，训练数据：{}，测试数据：{}"
                           .format(i, train_labels_sorted[train_index], test_labels_sorted[test_index]))
                return
        #  取并集
        union_timestamps = list(set(train_timestamp_sorted).union(set(test_timestamp_sorted)))
        union_amount = len(union_timestamps)
        union_values = np.zeros([union_amount], dtype=train_values_sorted.dtype)
        union_labels = np.zeros([union_amount], dtype=train_labels_sorted.dtype)
        for i, t in enumerate(union_timestamps):
            # 寻找指定数值的索引
            train_index = np.where(train_timestamp_sorted == t)
            test_index = np.where(test_timestamp_sorted == t)
            if np.size(train_index) != 0:
                union_values[i] = train_values_sorted[train_index][0]
                union_labels[i] = train_labels_sorted[train_index][0]
            elif np.size(test_index) != 0:
                union_values[i] = test_values_sorted[test_index][0]
                union_labels[i] = test_labels_sorted[test_index][0]
            else:
                print_warn(use_plt, "找不到指定数值的索引:{}".format(str(t)))
        mean = np.mean(union_values)
        std = np.std(union_values, ddof=1)
    else:
        n = len(train_values_sorted)
        m = len(test_values_sorted)
        m1 = np.mean(train_values_sorted)
        m2 = np.mean(test_values_sorted)
        s1 = np.std(train_values_sorted)
        s2 = np.std(train_values_sorted)
        mean = (n * m1 + m * m2) / (m + n)
        std = np.sqrt((n * s1 * s1 + m * s2 * s2 + (m * n * (m1 - m2) * (m1 - m2)) / (m + n)) / (m + n))
    if train_min_interval != test_min_interval:
        print_warn(use_plt, '最小间隔数不同训练数据最小间隔：{},测试数据最小间隔：{}'
                   .format(train_min_interval, test_min_interval))
    for i in train_intervals:
        if i % train_min_interval != 0 or i % test_min_interval != 0:
            print_warn(use_plt, '并不是所有时间间隔都是最小时间间隔的倍数,最小间隔：{},异常间隔：{}'
                       .format(train_min_interval, i))
    for i in test_intervals:
        if i % train_min_interval != 0 or i % test_min_interval != 0:
            print_warn(use_plt, '并不是所有时间间隔都是最小时间间隔的倍数,最小间隔：{},异常间隔：{}'
                       .format(test_min_interval, i))
    # 处理时间戳 获得缺失点
    train_amount = (train_timestamp_sorted[-1] - train_timestamp_sorted[0]) // train_min_interval + 1
    test_amount = (test_timestamp_sorted[-1] - test_timestamp_sorted[0]) // test_min_interval + 1
    # 初始化
    fill_train_values = np.zeros([train_amount], dtype=train_values_sorted.dtype)
    fill_test_values = np.zeros([test_amount], dtype=test_values_sorted.dtype)
    fill_train_labels = np.zeros([train_amount], dtype=train_labels_sorted.dtype)
    fill_test_labels = np.zeros([test_amount], dtype=test_labels_sorted.dtype)
    # 重构时间戳数组
    fill_train_timestamps = np.arange(train_timestamp_sorted[0], train_timestamp_sorted[-1] + train_min_interval,
                                      train_min_interval, dtype=np.int64)
    fill_test_timestamps = np.arange(test_timestamp_sorted[0], test_timestamp_sorted[-1] + test_min_interval,
                                     test_min_interval, dtype=np.int64)
    # 初始化缺失点数组与数值与标注数组
    train_missing = np.ones([train_amount], dtype=np.int32)
    test_missing = np.ones([test_amount], dtype=np.int32)
    # 3.填充数值
    # 获得与初始时间戳的差值数组
    train_diff_with_first = (train_timestamp_sorted - train_timestamp_sorted[0])
    test_diff_with_first = (test_timestamp_sorted - test_timestamp_sorted[0])
    # 获得与初始时间戳相差的最小间隔数 即应处的索引值
    diff_train_intervals_with_first = train_diff_with_first // train_min_interval
    dst_train_index = np.asarray(diff_train_intervals_with_first, dtype=np.int)
    diff_test_intervals_with_first = test_diff_with_first // test_min_interval
    dst_test_index = np.asarray(diff_test_intervals_with_first, dtype=np.int)
    # 标记有原值的时间戳为非缺失点
    train_missing[dst_train_index] = 0
    test_missing[dst_test_index] = 0
    # 分别组合
    fill_train_values[dst_train_index] = src_train_values[src_train_index]
    fill_test_values[dst_test_index] = src_test_values[src_test_index]
    fill_train_labels[dst_train_index] = src_train_labels[src_train_index]
    fill_test_labels[dst_test_index] = src_test_labels[src_test_index]
    return mean, std, \
           fill_train_timestamps, fill_train_values, fill_train_labels, train_missing, \
           fill_test_timestamps, fill_test_values, fill_test_labels, test_missing


def self_structure(use_plt=True, train_file="4096_14.21.csv", test_file="4096_1.88.csv", is_local=True, is_upload=False,
                   src_threshold_value=None):
    """
    非缓存运行
    Args:
        test_file: 测试文件
        is_local: 本地图片展示
        is_upload: 是否为上传文件
        use_plt: 展示方式使用plt？
        train_file: 文件名
        src_threshold_value: 初始阈值
    """
    # 获取训练数据
    tc = TimeCounter()
    tc1 = TimeCounter()
    tc1.start()
    tc.start()
    src_train_timestamps, src_train_labels, src_train_values, train_file, success \
        = get_info_from_file(is_upload, is_local, train_file)
    tc.end()
    if not success:
        print_warn(use_plt, "找不到数据文件，请检查文件名与路径")
        return
    get_train_file_time = tc.get_s() + "秒"
    tc.start()
    src_test_timestamps, src_test_labels, src_test_values, test_file, success \
        = get_info_from_file(is_upload, is_local, test_file)
    tc.end()
    if not success:
        print_warn(use_plt, "找不到数据文件，请检查文件名与路径")
        return
    get_test_file_time = tc.get_s() + "秒"
    # 原训练数据数量
    src_train_num = src_train_timestamps.size
    # 原训练数据标签数
    src_train_label_num = np.sum(src_train_labels == 1)
    # 原训练数据标签占比
    src_train_label_proportion = src_train_label_num / src_train_num
    # 原测试数据数量
    src_test_num = src_test_timestamps.size
    # 原测试数据标签数
    src_test_label_num = np.sum(src_test_labels == 1)
    # 原测试数据标签占比
    src_test_label_proportion = src_test_label_num / src_test_num
    tc1.end()
    get_file_time = tc1.get_s() + "秒"

    print_info(use_plt, "1.获取数据【共用时{}】".format(get_file_time))
    print_text("获取训练数据【共用时{}】", get_train_file_time)
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}"
               .format(src_train_num, src_train_label_num, src_train_label_proportion))
    show_line_chart(use_plt, src_train_timestamps, src_train_values, 'original csv train data')
    print_text("获取测试数据【共用时{}】", get_test_file_time)
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}"
               .format(src_test_num, src_test_label_num, src_test_label_proportion))
    show_line_chart(use_plt, src_test_timestamps, src_test_values, 'original csv test data')

    tc.start()
    # 合并数据
    mean, std, \
    fill_train_timestamps, fill_train_values, fill_train_labels, train_missing, \
    fill_test_timestamps, fill_test_values, fill_test_labels, test_missing \
        = merge_data(use_plt, src_train_timestamps, src_train_labels, src_train_values, src_test_timestamps,
                     src_test_labels, src_test_values)
    tc.end()
    fill_time = tc.get_s() + "秒"
    train_data_num = fill_train_timestamps.size
    test_data_num = fill_test_timestamps.size
    fill_train_num = train_data_num - src_train_num
    fill_test_num = test_data_num - src_test_num
    fill_train_label_num = np.sum(fill_train_labels == 1)
    fill_test_label_num = np.sum(fill_test_labels == 1)
    fill_train_label_proportion = fill_train_label_num / train_data_num
    fill_test_label_proportion = fill_test_label_num / test_data_num
    fill_step = fill_train_timestamps[1] - fill_train_timestamps[0]
    train_missing_index = np.where(train_missing == 1)
    train_missing_timestamps = fill_train_timestamps[train_missing_index]
    train_missing_interval_num, train_missing_str = get_constant_timestamp(train_missing_timestamps, fill_step)
    test_missing_index = np.where(test_missing == 1)
    test_missing_timestamps = fill_test_timestamps[test_missing_index]
    test_missing_interval_num, test_missing_str = get_constant_timestamp(test_missing_timestamps, fill_step)
    print_info(use_plt, "2.【数据处理】填充数据，计算平均值和标准差 【共用时{}】".format(fill_time))
    print_text(use_plt, "时间戳步长:{}".format(fill_step))
    print_text(use_plt, "平均值：{}，标准差：{}".format(mean, std))
    print_text(use_plt, "训练数据")
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}"
               .format(train_data_num, fill_train_label_num, fill_train_label_proportion))
    print_text(use_plt, "补充{}个时间戳数据,共有{}段连续缺失 \n {}"
               .format(fill_train_num, train_missing_interval_num, train_missing_str))
    show_line_chart(use_plt, fill_train_timestamps, fill_train_values, 'filled train data')
    print_text(use_plt, "测试数据")
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%}"
               .format(test_data_num, fill_test_label_num, fill_test_label_proportion))
    print_text(use_plt, "补充{}个时间戳数据,共有{}段连续缺失 \n {}"
               .format(fill_test_num, test_missing_interval_num, test_missing_str))
    show_line_chart(use_plt, fill_test_timestamps, fill_test_values, 'filled test data')
    # 标准化数据
    tc.start()
    std_train_values, std_test_values = standardize_data_v2(mean, std,
                                                            fill_train_values, fill_train_labels, train_missing,
                                                            fill_test_values)
    tc.end()
    std_time = tc.get_s() + "秒"
    print_info(use_plt, "3.【数据处理】标准化训练数据【共用时{}】".format(std_time))
    # 显示标准化后的训练数据
    show_line_chart(use_plt, fill_train_timestamps, std_train_values, 'standardized train data')
    show_line_chart(use_plt, fill_test_timestamps, std_test_values, 'standardized test data')
    # 进行训练，预测，获得重构概率
    tc.start()
    epoch_list, lr_list, epoch_time, \
    model_time, trainer_time, predictor_time, fit_time, train_message, \
    train_refactor_probability, train_probability_time, \
    test_refactor_probability, test_probability_time \
        = train_prediction_v2(
        use_plt, src_threshold_value,
        std_train_values, fill_train_labels, train_missing,
        std_test_values, fill_test_labels, test_missing,
        mean, std, test_data_num)
    tc.end()
    #  获得重构概率对应分数
    if src_threshold_value is None:
        train_scores, train_zero_num, real_train_timestamps, real_train_values, real_train_labels, real_train_missing \
            = handle_refactor_probability_v2(train_refactor_probability, train_data_num,
                                             fill_train_timestamps, std_train_values, fill_train_labels, train_missing)
    else:
        train_scores, train_zero_num, real_train_values, real_train_labels, real_train_missing \
            = None, None, std_train_values, fill_train_labels, train_missing
    # 处理重构概率
    test_scores, test_zero_num, real_test_timestamps, real_test_values, real_test_labels, real_test_missing \
        = handle_refactor_probability_v2(test_refactor_probability, test_data_num,
                                         fill_test_timestamps, std_test_values, fill_test_labels, test_missing)
    real_test_data_num = np.size(real_test_timestamps)
    real_test_label_num = np.sum(real_test_labels == 1)
    real_test_missing_num = np.sum(real_test_missing == 1)
    real_test_label_proportion = real_test_label_num / real_test_data_num
    print_text(use_plt, "实际测试数据集")
    show_test_score(use_plt, real_test_timestamps, real_test_values, test_scores)
    print_text(use_plt, "共{}条数据,有{}个标注，有{}个缺失数据，标签比例约为{:.2%}"
               .format(real_test_data_num, real_test_label_num, real_test_missing_num, real_test_label_proportion))
    # 根据分数捕获异常 获得阈值
    threshold_value, catch_num, catch_index, f_score, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall \
        = catch_label_v2(use_plt, src_threshold_value, train_scores, real_train_labels, test_scores, real_test_labels)
    fp_interval_num, fp_interval_str = get_constant_timestamp(fp_index, fill_step)
    print_text(use_plt, "未标记但超过阈值的点（数量：{}）：\n 共有{}段连续异常 \n ".format(fp_interval_num, fp_interval_str))
    # time_list = [TimeUse(get_train_file_time, "1.分析csv数据"), TimeUse(fill_train_time, "2.填充数据"),
    #              TimeUse(third_time, "3.获得训练与测试数据集"),
    #              TimeUse(std_time, "4.标准化训练和测试数据"), TimeUse(model_time, "5.构建Donut模型"),
    #              TimeUse(trainer_time, "6.构造训练器"), TimeUse(predictor_time, "7.构造预测器"),
    #              TimeUse(fit_time, "8.训练模型"), TimeUse(probability_time, "9.获得重构概率")]
    # time_list = np.array(time_list)
    # sorted_time_list = sorted(time_list)
    # t_use = []
    # t_name = []
    # print_info(use_plt, "用时排名正序")
    # for i, t in enumerate(sorted_time_list):
    #     print_text(use_plt, "第{}：{}用时{}".format(i + 1, t.name, t.use))
    #     t_use.append(t.use)
    #     t_name.append(t.name)
    # save_data_cache(use_plt, is_local, train_file, test_portion, src_threshold_value,
    #                 src_train_timestamps, src_train_labels, src_train_values, src_train_num, src_train_label_num,
    #                 src_train_label_proportion,
    #                 get_train_file_time, fill_train_timestamps, fill_train_values, train_data_num, fill_step,
    #                 fill_train_num,
    #                 fill_train_time,
    #                 third_time, train_data_num, fill_train_label_num, train_label_proportion, test_data_num,
    #                 test_label_num, test_label_proportion, mean, std, std_time, epoch_list, lr_list,
    #                 epoch_time, fifth_time, catch_num, labels_num, accuracy, special_anomaly_num,
    #                 train_missing_interval_num,
    #                 train_missing_str, special_anomaly_t, special_anomaly_v, special_anomaly_s, test_timestamps,
    #                 test_values,
    #                 test_scores, model_time, trainer_time, predictor_time, fit_time, probability_time, threshold_value,
    #                 train_message, train_timestamps, std_train_values, t_use, t_name, src_train_values, src_test_values)
