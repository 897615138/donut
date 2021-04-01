# coding=utf-8
import csv
import time

import numpy as np
import pandas as pd

from donut import complete_timestamp, standardize_kpi
from donut.demo.cache import gain_data_cache, save_data_cache
from donut.demo.out import print_text, show_line_chart, show_prepare_data_one, show_test_score, bar_chart
from donut.demo.train_prediction import train_prediction
from donut.utils import get_time, compute_default_threshold_value, get_constant_timestamp, TimeUse

__all__ = ['prepare_data', 'gain_data', 'fill_data', 'get_test_training_data', 'standardize_data', 'handle_test_data',
           'get_threshold_value_label', 'catch_label', 'show_cache_data', 'show_new_data']


def prepare_data(file_name, test_portion=0.3):
    """
      数据准备1.0
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
    train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    return src_timestamps, base_values, train_timestamps, train_values, test_timestamps, test_values, train_missing, test_missing, \
           train_labels, test_labels, mean, std


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
    return timestamp, missing, values, labels


def get_test_training_data(fill_values, fill_labels, src_misses, fill_timestamps, test_portion=0.3):
    """
    获得测试与训练数据集
    Args:
        fill_values: 值数据集
        fill_labels: 异常标识数据集
        src_misses: 缺失点数据集
        fill_timestamps: 时间戳数据集
        test_portion: 测试数据占比

    Returns:

    """
    test_amount = int(len(fill_values) * test_portion)
    train_values, test_values = np.asarray(fill_values[:-test_amount]), np.asarray(fill_values[-test_amount:])
    train_labels, test_labels = fill_labels[:-test_amount], fill_labels[-test_amount:]
    train_missing, test_missing = src_misses[:-test_amount], src_misses[-test_amount:]
    train_timestamp, test_timestamp = fill_timestamps[:-test_amount], fill_timestamps[-test_amount:]
    return train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp


def standardize_data(train_labels, train_missing, train_values, test_values):
    """
    标准化数据
    Args:
        train_labels: 训练数据的异常标识
        train_missing: 训练数据的缺失点数据
        train_values: 训练数据的值数据集
        test_values: 测试数据的值数据集

    Returns:

    """
    exclude_array = np.logical_or(train_labels, train_missing)
    train_values, mean, std = standardize_kpi(train_values, excludes=np.asarray(exclude_array, dtype='bool'))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    return train_values, test_values, train_missing, train_labels, mean, std


def handle_test_data(test_score, test_num):
    """
    处理测试数据分数结果
    Args:
        test_score: 测试分数
        test_num: 测试数据数量

    Returns:
        处理过的测试分数，补上的0的数量
    """
    # 因为对于每个窗口的检测实际返回的是最后一个窗口的 score，也就是说第一个窗口的前面一部分的点都没有检测，默认为正常数据。因此需要在检测结果前面补零或者测试数据的真实 label。
    zero_num = test_num - test_score.size
    test_score = np.pad(test_score, (zero_num, 0), 'constant', constant_values=(0, 0))
    test_score = 0 - test_score
    return test_score, zero_num


def get_threshold_value_label(labels_score, test_score, labels_num):
    """
    带异常标签的默认阈值
    Args:
        labels_num: 标签数量
        test_score: 所有分值
        labels_score: 异常标签对应分数

    Returns:
        默认阈值
    """
    labels_score = np.sort(labels_score)
    for i, score in enumerate(labels_score):
        catch_index = np.where(test_score > float(score))[0].tolist()
        catch_num = np.size(catch_index)
        accuracy = labels_num / catch_num
        if 0.9 < accuracy <= 1:
            return score, catch_num, catch_index, accuracy
        elif accuracy > 1:
            return labels_score[i - 1], catch_num, catch_index, accuracy


def catch_label(use_plt, test_labels, test_scores, zero_num, threshold_value):
    """
    根据阈值捕获异常点
    Args:
        use_plt: 使用plt
        test_labels: 测试异常标签
        test_scores: 测试数据分数
        zero_num: 补齐的0点数量
        threshold_value: 已有的阈值

    Returns:
        捕捉到的异常信息，阈值信息

    """
    labels_index = list(np.where(test_labels == 1)[0])
    labels_index = [ele for ele in labels_index if ele > test_labels[0] + zero_num]
    labels_num = np.size(labels_index)
    accuracy = None
    if threshold_value is not None:
        catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
        accuracy = labels_num / catch_num
        if accuracy <= 0.9:
            print_text(use_plt, "建议提高阈值或使用【默认阈值】")
        elif accuracy > 1:
            print_text(use_plt, "建议降低阈值或使用【默认阈值】")
    elif len(labels_index) == 0:
        threshold_value = compute_default_threshold_value(test_scores)
        catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
    else:
        labels_score = test_scores[labels_index]
        threshold_value, catch_num, catch_index, accuracy = get_threshold_value_label(labels_score,
                                                                                      test_scores, labels_num)
        # 准确度
        accuracy = labels_num / catch_num
    return labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy


def show_cache_data(use_plt, file_name, test_portion, src_threshold_value):
    if use_plt:
        pass
    else:
        src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
        fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
        train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
        mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
        accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s, \
        test_timestamps, test_values, test_scores, model_time, trainer_time, predictor_time, fit_time, probability_time \
            = gain_data_cache(use_plt, file_name, test_portion, src_threshold_value)

        show_line_chart(use_plt, src_timestamps, src_values, 'original csv_data')
        print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】"
                   .format(src_data_num, src_label_num, src_label_proportion, first_time))
        show_line_chart(use_plt, fill_timestamps, fill_values, 'fill_data')
        print_text(use_plt, "填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
                   .format(fill_data_num, fill_step, fill_num, second_time))
        print_text(use_plt, "训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                            "测试数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                            "【填充缺失数据,共用时{}】"
                   .format(train_data_num, train_label_num, train_label_proportion,
                           test_data_num, test_label_num, test_label_proportion, third_time))
        print_text(use_plt, "平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, forth_time))
        print_text(use_plt, "构建Donut模型【共用时{}】\n"
                            "构建训练器【共用时{}】\n"
                            "构造预测器【共用时{}】\n"
                            "训练模型【共用时{}】\n"
                            "获得重构概率【共用时{}】".format(model_time, trainer_time, predictor_time, fit_time,
                                                   probability_time))
        show_line_chart(use_plt, epoch_list, lr_list, 'annealing_learning_rate')
        print_text(use_plt, "退火学习率随epoch变化\n【所有epoch共用时：{}\n【训练模型与预测获得测试分数,共用时{}】】".format(epoch_time, fifth_time))
        show_test_score(use_plt, test_timestamps, test_values, test_scores)
        if accuracy is not None:
            print_text(use_plt, "标签准确度:{:.2%}".format(accuracy))
        print_text(use_plt,
                   "未标记但超过阈值的点（数量：{}）：\n 共有{}段(处)异常 \n {}".format(special_anomaly_num, interval_num, interval_str))
        for i, t in enumerate(special_anomaly_t):
            print_text(use_plt, "时间戳:{},值:{},分数：{}".format(t, special_anomaly_v[i], special_anomaly_s[i]))
        time_list = [TimeUse(first_time, "1.分析csv数据"), TimeUse(second_time, "2.填充数据"), TimeUse(third_time, "3.填充缺失数据"),
                     TimeUse(forth_time, "4.标准化训练和测试数据"), TimeUse(model_time, "5.构建Donut模型"),
                     TimeUse(trainer_time, "6.构造训练器"), TimeUse(predictor_time, "7.构造预测器"),
                     TimeUse(fit_time, "8.训练模型"), TimeUse(probability_time, "9.获得重构概率")]
        time_list = np.array(time_list)
        sorted_time_list = sorted(time_list)
        s_time = []
        n_time = []
        for t in sorted_time_list:
            s_time.append(t.use*1000)
            n_time.append(t.name)
        print_text(use_plt,"为使比较结果明显，将所有时间*1000")
        chart_data = pd.DataFrame(
            [s_time],
            columns=n_time)
        bar_chart(use_plt, chart_data)


def show_new_data(use_plt, file_name, test_portion, src_threshold_value):
    """
    非缓存运行
    Args:
        use_plt: 展示方式使用plt？
        file_name: 文件名
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值
    """
    start_time = time.time()
    src_timestamps, src_labels, src_values = gain_data("sample_data/" + file_name)
    end_time = time.time()
    # 原数据数量
    src_data_num = src_timestamps.size
    # 原数据标签数
    src_label_num = np.sum(src_labels == 1)
    # 原数据标签占比
    src_label_proportion = src_label_num / src_data_num
    first_time = get_time(start_time, end_time)
    # 显示原始数据信息
    show_line_chart(use_plt, src_timestamps, src_values, 'original csv_data')
    print_text(use_plt, "共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】"
               .format(src_data_num, src_label_num, src_label_proportion, first_time))
    start_time = time.time()
    # 处理时间戳 获得缺失点
    fill_timestamps, src_misses, fill_values, fill_labels = fill_data(src_timestamps, src_labels, src_values)
    end_time = time.time()
    fill_data_num = fill_timestamps.size
    fill_num = fill_data_num - src_data_num
    # 显示填充信息
    show_line_chart(use_plt, fill_timestamps, fill_values, 'fill_data')
    fill_step = fill_timestamps[1] - fill_timestamps[0]
    second_time = get_time(start_time, end_time)
    print_text(use_plt, "填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
               .format(fill_data_num, fill_step, fill_num, second_time))
    # 获得测试与训练数据集
    start_time = time.time()
    train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamps, test_timestamps = \
        get_test_training_data(fill_values, fill_labels, src_misses, fill_timestamps, test_portion)
    end_time = time.time()
    third_time = get_time(start_time, end_time)
    # 显示测试与训练数据集
    show_prepare_data_one \
        (use_plt, src_timestamps, src_values, train_timestamps, train_values, test_timestamps, test_values)
    train_data_num = train_values.size
    train_label_num = np.sum(train_labels == 1)
    train_label_proportion = train_label_num / train_data_num
    test_data_num = test_values.size
    test_label_num = np.sum(test_labels == 1)
    test_label_proportion = test_label_num / test_data_num
    print_text(use_plt, "训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                        "测试数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                        "【填充缺失数据,共用时{}】"
               .format(train_data_num, train_label_num, train_label_proportion,
                       test_data_num, test_label_num, test_label_proportion, third_time))
    # 标准化数据
    start_time = time.time()
    train_values, test_values, train_missing, train_labels, mean, std = \
        standardize_data(train_labels, train_missing, train_values, test_values)
    end_time = time.time()
    forth_time = get_time(start_time, end_time)
    # 显示标准化后的数据
    show_prepare_data_one \
        (use_plt, src_timestamps, src_values, train_timestamps, train_values, test_timestamps, test_values)
    print_text(use_plt, "平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, forth_time))
    # 进行训练，预测，获得重构概率
    start_time = time.time()
    refactor_probability, epoch_list, lr_list, epoch_time, model_time, trainer_time, predictor_time, fit_time, probability_time = \
        train_prediction(use_plt, train_values, train_labels, train_missing, test_values, test_missing, mean, std)
    end_time = time.time()
    fifth_time = get_time(start_time, end_time)
    # 显示退火学习率过程
    show_line_chart(use_plt, epoch_list, lr_list, 'annealing_learning_rate')
    print_text(use_plt, "退火学习率随epoch变化\n【所有epoch共用时：{}\n【训练模型与预测获得测试分数,共用时{}】】".format(epoch_time, fifth_time))
    # 获得重构概率对应分数
    test_scores, zero_num = handle_test_data(refactor_probability, test_values.size)
    # 显示分数
    show_test_score(use_plt, test_timestamps, test_values, test_scores)
    # 根据分数捕获异常 获得阈值
    labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy = \
        catch_label(use_plt, test_labels, test_scores, zero_num, src_threshold_value)
    print_text(use_plt, "默认阈值：{},根据默认阈值获得的异常点数量：{},实际异常标注数量:{}".format(threshold_value, catch_num, labels_num))
    # 如果有标签准确率 显示
    if accuracy is not None:
        print_text(use_plt, "标签准确度:{:.2%}".format(accuracy))
    # 比较异常标注与捕捉的异常的信息
    special_anomaly_index = list(set(catch_index) - set(labels_index))
    special_anomaly_t = test_timestamps[special_anomaly_index]
    special_anomaly_s = test_scores[special_anomaly_index]
    special_anomaly_v = test_values[special_anomaly_index]
    special_anomaly_num = len(special_anomaly_t)
    interval_num, interval_str = get_constant_timestamp(use_plt, special_anomaly_t, fill_step)
    print_text(use_plt, "未标记但超过阈值的点（数量：{}）：\n 共有{}段(处)异常 \n {}".format(special_anomaly_num, interval_num, interval_str))
    for i, t in enumerate(special_anomaly_t):
        print_text(use_plt, "时间戳:{},值:{},分数：{}".format(t, special_anomaly_v[i], special_anomaly_s[i]))
    save_data_cache(use_plt, file_name, test_portion, src_threshold_value,
                    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion,
                    first_time, fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time,
                    third_time, train_data_num, train_label_num, train_label_proportion, test_data_num,
                    test_label_num, test_label_proportion, mean, std, forth_time, epoch_list, lr_list, epoch_time,
                    fifth_time, catch_num, labels_num, accuracy, special_anomaly_num, interval_num, interval_str,
                    special_anomaly_t, special_anomaly_v, special_anomaly_s, test_timestamps, test_values, test_scores,
                    model_time, trainer_time, predictor_time, fit_time, probability_time)
