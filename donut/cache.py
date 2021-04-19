import os
import shelve

from donut.out import print_text, print_info
from donut.utils import TimeCounter, file_name_converter, format_time


def save_data_cache(use_plt, is_local, file_name, test_portion, src_threshold_value,
                    src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion,
                    first_time, fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time,
                    third_time, train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num,
                    test_label_proportion, train_mean, train_std, forth_time, epoch_list, lr_list, epoch_time,
                    fifth_time, catch_num, labels_num, accuracy, special_anomaly_num, interval_num, interval_str,
                    special_anomaly_t, special_anomaly_v, special_anomaly_s, test_timestamps, test_values, test_scores,
                    model_time, trainer_time, predictor_time, fit_time, probability_time, threshold_value,
                    train_message, train_timestamps, train_values, t_use, t_name, src_train_values, src_test_values):
    """
    保存缓存对象
    Args:
        src_test_values: 标准化前的测试数据
        src_train_values: 标准前的训练数据
        is_local: 展示本地图片
        t_name: 用时排序名称
        t_use: 用时排序用时
        train_values: 训练数据值
        train_timestamps: 训练时间戳
        train_message: 训练信息
        threshold_value: 阈值
        use_plt: 展示方式
        file_name: 文件名
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值
        src_timestamps: 初始时间戳数据
        src_labels: 初始异常标签
        src_values: 初始值
        src_data_num: 初始数据数量
        src_label_num: 初始异常标签数量
        src_label_proportion: 初始异常标签所占比例
        first_time: 第一阶段使用时间
        fill_timestamps: 排序并填充的时间戳
        fill_values: 排序并填充的值
        fill_data_num: 填充后的数据数量
        fill_step: 填充后时间戳步长
        fill_num: 填充的数据的数量
        second_time: 第二阶段的用时
        third_time: 第三阶段用时
        train_data_num: 训练数据数量
        train_label_num: 训练数据标签数量
        train_label_proportion: 训练数据中的异常标签比例
        test_data_num: 测试数据数量
        test_label_num: 测试数据中异常标签数量
        test_label_proportion: 测试数据中异常标签比例
        train_mean: 平均值
        train_std: 标准差
        forth_time: 第四阶段用时
        epoch_list: 迭代遍数
        lr_list: 学习率
        epoch_time: 迭代时间
        fifth_time: 第五阶段用时
        catch_num: 根据阈值捕捉到的数量
        labels_num: 异常标注数量
        accuracy: 异常标注精确率
        special_anomaly_num: 被捕捉到的异常数量
        interval_num: 连续的异常段数量
        interval_str: 连续异常段字符串
        special_anomaly_t: 被捕捉到的异常的时间戳
        special_anomaly_v: 被捕捉到的异常的值
        special_anomaly_s: 被捕捉到的异常的分数
        test_timestamps: 测试数据时间戳
        test_values: 测试数据值
        test_scores: 测试数据分数
        model_time: 构建模型用时
        trainer_time: 构建训练器用时
        predictor_time: 构建预测期用时
        fit_time: 训练时长
        probability_time: 获得重构概率用时
    """
    tc = TimeCounter()
    print_text(use_plt, "缓存开始")
    tc.start()
    name = file_name_converter(file_name, test_portion, src_threshold_value, is_local)
    db = shelve.open(name )
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
    db["train_mean"] = train_mean
    db["train_std"] = train_std
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
    db["threshold_value"] = threshold_value
    db["train_message"] = train_message
    db["train_timestamps"] = train_timestamps
    db["train_values"] = train_values
    db["t_use"] = t_use
    db["t_name"] = t_name
    db["src_train_values"] = src_train_values
    db["src_test_values"] = src_test_values
    tc.end()
    print_info(use_plt, "缓存结束【共用时：{}】".format(tc.get_s()+"秒"))
    db.close()


def gain_data_cache(use_plt, file_name, test_portion, src_threshold_value, is_local):
    """
    获得缓存数据
    Args:
        is_local: 本地照片显示
        use_plt: 显示格式
        file_name: 数据文件名称
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值

    Returns:
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值
        src_timestamps: 初始时间戳数据
        src_labels: 初始异常标签
        src_values: 初始值
        src_data_num: 初始数据数量
        src_label_num: 初始异常标签数量
        src_label_proportion: 初始异常标签所占比例
        first_time: 第一阶段使用时间
        fill_timestamps: 排序并填充的时间戳
        fill_values: 排序并填充的值
        fill_data_num: 填充后的数据数量
        fill_step: 填充后时间戳步长
        fill_num: 填充的数据的数量
        second_time: 第二阶段的用时
        third_time: 第三阶段用时
        train_data_num: 训练数据数量
        train_label_num: 训练数据标签数量
        train_label_proportion: 训练数据中的异常标签比例
        test_data_num: 测试数据数量
        test_label_num: 测试数据中异常标签数量
        test_label_proportion: 测试数据中异常标签比例
        train_mean: 平均值
        train_std: 标准差
        forth_time: 第四阶段用时
        epoch_list: 迭代遍数
        lr_list: 学习率
        epoch_time: 迭代时间
        fifth_time: 第五阶段用时
        catch_num: 根据阈值捕捉到的数量
        labels_num: 异常标注数量
        accuracy: 异常标注精确率
        special_anomaly_num: 被捕捉到的异常数量
        interval_num: 连续的异常段数量
        interval_str: 连续异常段字符串
        special_anomaly_t: 被捕捉到的异常的时间戳
        special_anomaly_v: 被捕捉到的异常的值
        special_anomaly_s: 被捕捉到的异常的分数
        test_timestamps: 测试数据时间戳
        test_values: 测试数据值
        test_scores: 测试数据分数
        model_time: 构建模型用时
        trainer_time: 构建训练器用时
        predictor_time: 构建预测期用时
        fit_time: 训练时长
        probability_time: 获得重构概率用时
        threshold_value: 阈值
    """
    print_text(use_plt, "读取缓存开始")
    tc= TimeCounter()
    tc.start()
    name = file_name_converter(file_name, test_portion, src_threshold_value, is_local)
    db = shelve.open(name)
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
    train_mean = db["train_mean"]
    train_std = db["train_std"]
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
    threshold_value = db["threshold_value"]
    fit_time = db["fit_time"]
    probability_time = db["probability_time"]
    train_message = db["train_message"]
    train_timestamps = db["train_timestamps"]
    train_values = db["train_values"]
    t_use = db["t_use"]
    t_name = db["t_name"]
    src_train_values = db["src_train_values"]
    src_test_values = db["src_test_values"]
    tc.end()
    print_info(use_plt, "读取缓存数据结束【共用时：{}】".format(tc.get_s()+"秒"))
    db.close()
    return src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
           fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
           train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
           train_mean, train_std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, \
           labels_num, accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, \
           special_anomaly_s, test_timestamps, test_values, test_scores, model_time, trainer_time, predictor_time, \
           fit_time, probability_time, threshold_value, train_message, train_timestamps, train_values, t_use, t_name, \
           src_train_values, src_test_values


def is_has_cache(file_name, test_portion, src_threshold_value, is_local):
    """
    是否有对应缓存+缓存时间
    Args:
        is_local: 本地图片显示
        file_name: 文件名称
        test_portion: 测试数据比例
        src_threshold_value: 初始阈值

    Returns:
        是否存在缓存
        缓存文件信息
    """
    name = file_name_converter(file_name, test_portion, src_threshold_value, is_local)
    cache_name = name + '.db'
    exist = os.path.exists(cache_name)
    if exist:
        file = os.stat(cache_name)
        return exist, "该配置有缓存,建议使用缓存数据\n" \
                      "文件名：{},文件大小：{} 字节\n" \
                      "最后一次访问时间:{},最后一次修改时间：{}" \
            .format(cache_name, file.st_size, format_time(file.st_atime), format_time(file.st_mtime))
    else:
        return exist, "该配置无缓存,默认缓存数据"
