import os
import shelve
import time

from donut.demo.out import print_text
from donut.utils import get_time, file_name_converter


def save_data_cache(use_plt, file_name, test_portion, src_threshold_value,
                    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion,
                    first_time, fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time,
                    third_time, train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num,
                    test_label_proportion, mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time,
                    catch_num, labels_num, accuracy, special_anomaly_num, interval_num, interval_str,
                    special_anomaly_t, special_anomaly_v, special_anomaly_s, test_timestamps, test_values, test_scores,
                    model_time, trainer_time, predictor_time, fit_time, probability_time):
    print_text(use_plt, "缓存开始")
    start_time = time.time()
    db = shelve.open(file_name_converter(file_name, test_portion, src_threshold_value))
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
    db["src_threshold_value"] = src_threshold_value
    db["test_timestamps"] = test_timestamps
    db["test_values"] = test_values
    db["test_scores"] = test_scores
    db["model_time"] = model_time
    db["trainer_time"] = trainer_time
    db["predictor_time"] = predictor_time
    db["fit_time"] = fit_time
    db["probability_time"] = probability_time
    end_time = time.time()
    print_text(use_plt, "缓存结束【共用时：{}】".format(get_time(start_time, end_time)))
    db.close()


def gain_data_cache(use_plt, file_name, test_portion, src_threshold_value):
    print_text(use_plt, "读取缓存开始")
    start_time = time.time()
    db = shelve.open(file_name_converter(file_name, test_portion, src_threshold_value))
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
    test_timestamps = db["test_timestamps"]
    test_values = db["test_values"]
    test_scores = db["test_scores"]
    model_time = db["model_time"]
    trainer_time = db["trainer_time"]
    predictor_time = db["predictor_time"]
    fit_time = db["fit_time"]
    probability_time = db["probability_time"]
    end_time = time.time()
    print_text(use_plt, "读取缓存数据结束【共用时：{}】".format(get_time(start_time, end_time)))
    db.close()
    return src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
           fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
           train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
           mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
           accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s, \
           test_timestamps, test_values, test_scores, model_time, trainer_time, predictor_time, fit_time, probability_time


def is_has_cache(file_name, test_portion, src_threshold_value):
    name = file_name_converter(file_name, test_portion, src_threshold_value)
    return os.path.exists(name + '.db')
