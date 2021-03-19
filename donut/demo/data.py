# coding=utf-8
import csv
import shelve
import time

import numpy as np
from donut.demo.train_prediction import train_prediction

from donut.utils import get_time

import donut.demo.show_sl as sl
import os
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
    """
    获取数据
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


def get_test_training_data(values, labels, missing, timestamp, test_portion=0.3):
    """
    获得测试与训练数据集
    Args:
        values: 值数据集
        labels: 异常标识数据集
        missing: 缺失点数据集
        timestamp: 时间戳数据集
        test_portion: 测试数据占比

    Returns:

    """
    test_amount = int(len(values) * test_portion)
    train_values, test_values = np.asarray(values[:-test_amount]), np.asarray(values[-test_amount:])
    train_labels, test_labels = labels[:-test_amount], labels[-test_amount:]
    train_missing, test_missing = missing[:-test_amount], missing[-test_amount:]
    train_timestamp, test_timestamp = timestamp[:-test_amount], timestamp[-test_amount:]
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


def label_catch(test_labels, test_score, zero_num, threshold_value):
    labels_index = list(np.where(test_labels == 1)[0])
    labels_index = [ele for ele in labels_index if ele > test_labels[0] + zero_num]
    labels_num = np.size(labels_index)
    accuracy = None
    if threshold_value is not None:
        catch_index = np.where(test_score > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
        accuracy = labels_num / catch_num
        if accuracy <= 0.9:
            sl.text("建议提高阈值或使用【默认阈值】")
        elif accuracy > 1:
            sl.text("建议降低阈值或使用【默认阈值】")
    elif len(labels_index) == 0:
        threshold_value = get_threshold_value(test_score)
        catch_index = np.where(test_score > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
    else:
        labels_score = test_score[labels_index]
        threshold_value, catch_num, catch_index, accuracy = get_threshold_value_label(labels_score,
                                                                                      test_score, labels_num)
        # 准确度
        accuracy = labels_num / catch_num
    return labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy


def get_threshold_value(values):
    """
    默认阈值 至少10个数据，至多20个数据
    Args:
        values: 数据集

    Returns: 默认阈值
    """
    values = np.sort(values)
    num = np.size(values)
    count = round(num * 0.1 / 100)
    if count >= 20:
        return values[num - 20]
    elif count <= 10:
        return values[num - 10]
    else:
        return values[num - count]


def get_constant_timestamp(special_anomaly_t, step):
    if np.size(special_anomaly_t) == 0:
        pass
    else:
        special_anomaly_t = np.sort(special_anomaly_t)
        sl.text("其中时间戳分布为")
        has_dot = 0
        has_head = 0
        interval_str = ''
        last = 0
        interval_num = 0
        for i, t in enumerate(special_anomaly_t):
            if has_head == 0:
                interval_str = interval_str + " " + str(t)
                has_head = 1
                last = t
                interval_num = interval_num + 1
            else:
                if int(t) == int(last) + int(step):
                    if has_dot == 0:
                        interval_str = interval_str + "..."
                    else:
                        last = t
                else:
                    interval_str = interval_str + str(last)
                    last = t
                    has_head = 0
        return interval_num, interval_str


def handle_threshold_value(src_threshold_value):
    if src_threshold_value.isdecimal():
        src_threshold_value = float(src_threshold_value)
    else:
        src_threshold_value = None
    return src_threshold_value


def gain_data_cache(file_name, test_portion, threshold_value):
    # return test_score, epoch_list, lr_list, epoch_time, zero_num
    name = file_name_converter(file_name, test_portion, threshold_value)
    sl.text("读取缓存开始")
    start_time = time.time()
    db = shelve.open(file_name_converter(file_name, test_portion, threshold_value))
    src_timestamps = db["src_timestamps"]
    src_labels = db["src_labels"]
    src_values = db["src_values"]
    src_data_num = db["src_data_num"]
    src_label_num = db["src_label_num"]
    src_label_proportion = db["src_label_proportion"]
    first_time = db["first_time"]
    fill_timestamps = db["fill_timestamps"]
    fill_values = db["fill_values"]
    fill_data_num = db["fill_data_num"]
    fill_step = db["fill_step"]
    fill_num = db["fill_num"]
    second_time = db["second_time"]
    third_time = db["third_time"]
    train_data_num = db["train_data_num"]
    train_label_num = db["train_label_num"]
    train_label_proportion = db["train_label_proportion"]
    test_data_num = db["test_data_num"]
    test_label_num = db["test_label_num"]
    test_label_proportion = db["test_label_proportion"]
    mean = db["mean"]
    std = db["std"]
    forth_time = db["forth_time"]
    epoch_list = db["epoch_list"]
    lr_list = db["lr_list"]
    epoch_time = db["epoch_time"]
    fifth_time = db["fifth_time"]
    catch_num = db["catch_num"]
    labels_num = db["labels_num"]
    accuracy = db["accuracy"]
    special_anomaly_num = db["special_anomaly_num"]
    interval_num = db["interval_num"]
    interval_str = db["interval_str"]
    special_anomaly_t = db["special_anomaly_t"]
    special_anomaly_v = db["special_anomaly_v"]
    special_anomaly_s = db["special_anomaly_s"]
    src_threshold_value = db["src_threshold_value"]
    end_time = time.time()
    sl.text("读取缓存数据结束【共用时：{}】".format(get_time(start_time, end_time)))
    db.close()
    return src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
           fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
           train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
           mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
           accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s


def file_name_converter(file_name, test_portion, threshold_value):
    return "cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)


def is_has_cache(file_name, test_portion, src_threshold_value):
    name = file_name_converter(file_name, test_portion, src_threshold_value)
    return os.path.exists(name+'.db')


def save_data_cache(file_name, test_portion, threshold_value,
                    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion,
                    first_time, fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time,
                    third_time, train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num,
                    test_label_proportion, mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time,
                    catch_num, labels_num, accuracy, special_anomaly_num, interval_num, interval_str,
                    special_anomaly_t, special_anomaly_v, special_anomaly_s):
    sl.text("缓存开始")
    start_time = time.time()
    db = shelve.open(file_name_converter(file_name, test_portion, threshold_value))
    db["src_timestamps"] = src_timestamps
    db["src_labels"] = src_labels
    db["src_values"] = src_values
    db["src_data_num"] = src_data_num
    db["src_label_num"] = src_label_num
    db["src_label_proportion"] = src_label_proportion
    db["first_time"] = first_time
    db["fill_timestamps"] = fill_timestamps
    db["fill_values"] = fill_values
    db["fill_data_num"] = fill_data_num
    db["fill_step"] = fill_step
    db["fill_num"] = fill_num
    db["second_time"] = second_time
    db["third_time"] = third_time
    db["train_data_num"] = train_data_num
    db["train_label_num"] = train_label_num
    db["train_label_proportion"] = train_label_proportion
    db["test_data_num"] = test_data_num
    db["test_label_num"] = test_label_num
    db["test_label_proportion"] = test_label_proportion
    db["mean"] = mean
    db["std"] = std
    db["forth_time"] = forth_time
    db["epoch_list"] = epoch_list
    db["lr_list"] = lr_list
    db["epoch_time"] = epoch_time
    db["fifth_time"] = fifth_time
    db["catch_num"] = catch_num
    db["labels_num"] = labels_num
    db["accuracy"] = accuracy
    db["special_anomaly_num"] = special_anomaly_num
    db["interval_num"] = interval_num
    db["interval_str"] = interval_str
    db["special_anomaly_t"] = special_anomaly_t
    db["special_anomaly_v"] = special_anomaly_v
    db["special_anomaly_s"] = special_anomaly_s
    end_time = time.time()
    sl.text("缓存结束【共用时：{}】".format(get_time(start_time, end_time)))
    db.close()


def show_cache_data(file_name, test_portion, src_threshold_value):
    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
    fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
    train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
    mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
    accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s \
        = gain_data_cache(file_name, test_portion, src_threshold_value)
    sl.line_chart(src_timestamps, src_values, 'original csv_data')
    sl.text("共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】"
            .format(src_data_num, src_label_num, src_label_proportion, first_time))
    sl.line_chart(fill_timestamps, fill_values, 'fill_data')
    sl.text("填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
            .format(fill_data_num, fill_step, fill_num, second_time))
    sl.text("训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
            "测试数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
            "【填充缺失数据,共用时{}】"
            .format(train_data_num, train_label_num, train_label_proportion,
                    test_data_num, test_label_num, test_label_proportion,
                    third_time))
    sl.text("平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, forth_time))
    sl.line_chart(epoch_list, lr_list, 'annealing_learning_rate')
    sl.text("退火学习率随epoch变化")
    sl.text("【所有epoch共用时：{}】".format(epoch_time))
    sl.text("【训练模型与预测获得测试分数,共用时{}】".format(fifth_time))
    sl.text("默认阈值：{},根据默认阈值获得的异常点数量：{},实际异常标注数量:{}".format(src_threshold_value, catch_num, labels_num))
    if accuracy is not None:
        sl.text("标签准确度:{:.2%}".format(accuracy))
    sl.text("未标记但超过阈值的点（数量：{}）：".format(special_anomaly_num))
    sl.text("共有{}段(处)异常".format(interval_num))
    sl.text(interval_str)
    for i, fill_timestamps in enumerate(special_anomaly_t):
        sl.text("时间戳:{},值:{},分数：{}".format(fill_timestamps, special_anomaly_v[i], special_anomaly_s[i]))


def show_new_data(file_name, test_portion, src_threshold_value):
    start_time = time.time()
    src_timestamps, src_labels, src_values = gain_data("sample_data/" + file_name)
    end_time = time.time()
    sl.line_chart(src_timestamps, src_values, 'original csv_data')
    # 原数据数量
    src_data_num = src_timestamps.size
    # 原数据标签数
    src_label_num = np.sum(src_labels == 1)
    # 原数据标签占比
    src_label_proportion = src_label_num / src_data_num
    first_time = get_time(start_time, end_time)
    sl.text("共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】"
            .format(src_data_num, src_label_num, src_label_proportion, first_time))

    start_time = time.time()
    fill_timestamps, src_misses, fill_values, fill_labels = fill_data(src_timestamps, src_labels, src_values)
    end_time = time.time()
    fill_data_num = fill_timestamps.size
    fill_num = fill_data_num - src_data_num
    sl.line_chart(fill_timestamps, fill_values, 'fill_data')
    fill_step = fill_timestamps[1] - fill_timestamps[0]
    second_time = get_time(start_time, end_time)
    sl.text("填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
            .format(fill_data_num, fill_step, fill_num, second_time))

    start_time = time.time()
    train_values, test_values, train_labels, test_labels, train_missing, test_missing, train_timestamp, test_timestamp = \
        get_test_training_data(fill_values, fill_labels, src_misses, fill_timestamps, test_portion)
    end_time = time.time()
    third_time = get_time(start_time, end_time)
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    train_data_num = train_values.size
    train_label_num = np.sum(train_labels == 1)
    train_label_proportion = train_label_num / train_data_num
    test_data_num = test_values.size
    test_label_num = np.sum(test_labels == 1)
    test_label_proportion = test_label_num / test_data_num
    sl.text("训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
            "测试数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
            "【填充缺失数据,共用时{}】"
            .format(train_data_num, train_label_num, train_label_proportion,
                    test_data_num, test_label_num, test_label_proportion,
                    third_time))

    start_time = time.time()
    train_values, test_values, train_missing, train_labels, mean, std = \
        standardize_data(train_labels, train_missing, train_values, test_values)
    end_time = time.time()
    sl.prepare_data_one(train_timestamp, train_values, test_timestamp, test_values)
    forth_time = get_time(start_time, end_time)
    sl.text("平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, forth_time))

    start_time = time.time()
    test_score, epoch_list, lr_list, epoch_time = \
        train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
    end_time = time.time()
    test_score, zero_num = handle_test_data(test_score, test_values.size)
    fifth_time = get_time(start_time, end_time)
    sl.line_chart(epoch_list, lr_list, 'annealing_learning_rate')
    sl.text("退火学习率随epoch变化")
    sl.text("【所有epoch共用时：{}】".format(epoch_time))
    sl.text("【训练模型与预测获得测试分数,共用时{}】".format(fifth_time))
    sl.show_test_score(test_timestamp, test_values, test_score)
    labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy = \
        label_catch(test_labels, test_score, zero_num, src_threshold_value)
    sl.text("默认阈值：{},根据默认阈值获得的异常点数量：{},实际异常标注数量:{}".format(threshold_value, catch_num, labels_num))
    if accuracy is not None:
        sl.text("标签准确度:{:.2%}".format(accuracy))
    special_anomaly_index = list(set(catch_index) - set(labels_index))
    special_anomaly_t = test_timestamp[special_anomaly_index]
    special_anomaly_s = test_score[special_anomaly_index]
    special_anomaly_v = test_values[special_anomaly_index]
    special_anomaly_num = len(special_anomaly_t)
    sl.text("未标记但超过阈值的点（数量：{}）：".format(special_anomaly_num))
    interval_num, interval_str = get_constant_timestamp(special_anomaly_t, fill_step)
    sl.text("共有{}段(处)异常".format(interval_num))
    sl.text(interval_str)
    for i, fill_timestamps in enumerate(special_anomaly_t):
        sl.text("时间戳:{},值:{},分数：{}".format(fill_timestamps, special_anomaly_v[i], special_anomaly_s[i]))
    save_data_cache(file_name, test_portion, src_threshold_value, src_timestamps, src_labels, src_values,
                    src_data_num, src_label_num, src_label_proportion, first_time, fill_timestamps,
                    fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, train_data_num,
                    train_label_num, train_label_proportion, test_data_num, test_label_num,
                    test_label_proportion, mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time,
                    catch_num, labels_num, accuracy, special_anomaly_num, interval_num,
                    interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s)
