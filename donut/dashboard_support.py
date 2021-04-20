import numpy as np

from donut.assessment import Assessment
from donut.data import get_info_from_file, merge_data, standardize_data_v2, handle_refactor_probability_v2
from donut.train_prediction import train_prediction_v2
from donut.util.out.out import print_warn, print_info, print_text, show_line_chart, show_test_score
from donut.util.time_util import TimeCounter, get_constant_timestamp


def new_data(use_plt=True, train_file="4096_14.21.csv", test_file="4096_1.88.csv", is_local=True, is_upload=False,
             src_threshold_value=None, a=1):
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
    #  获得重构概率对应分数
    epoch_list, lr_list, epoch_time, \
    model_time, trainer_time, predictor_time, fit_time, train_message, \
    test_refactor_probability, test_probability_time \
        = train_prediction_v2(use_plt,
                              std_train_values, fill_train_labels, train_missing,
                              std_test_values, fill_test_labels, test_missing,
                              mean, std, test_data_num)
    tc.end()
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
    assessment = Assessment(src_threshold_value, test_scores, real_test_labels, real_test_missing, a, use_plt)
    threshold_value, f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall \
        = assessment.get_assessment()
    catch_timestamps = real_test_timestamps[catch_index]
    catch_interval_num, catch_interval_str = get_constant_timestamp(catch_timestamps, fill_step)
    print_text(use_plt, "捕捉到异常（数量：{}）：\n 共有{}段连续 \n 具体为{}".format(catch_num, catch_interval_num, catch_interval_str))
    print_text(use_plt, "默认阈值：{}，最佳F分值：{}，精度:{}，召回率：{}".format(round(threshold_value, 7), f_score, precision, recall))
    tp_interval_num, tp_interval_str = get_constant_timestamp(tp_index, fill_step)
    print_text(use_plt, "【TP】成功监测出的异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}".format(tp_num, tp_interval_num, tp_interval_str))
    fp_interval_num, fp_interval_str = get_constant_timestamp(fp_index, fill_step)
    print_text(use_plt, "【FP】未标记但超过阈值的点（数量：{}）：\n 共有{}段连续 \n 具体为{}".format(fp_num, fp_interval_num, fp_interval_str))
    fn_interval_num, fn_interval_str = get_constant_timestamp(fn_index, fill_step)
    print_text(use_plt, "【FN】漏报异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}".format(fn_num, fn_interval_num, fn_interval_str))

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
