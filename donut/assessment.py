import numpy as np


def get_fp(catch_index, real_test_labels_index):
    """
    FP 未标记但超过阈值 实际为正常点单倍误判为异常点
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点
    Returns:
        实际为正常点单倍误判为异常点相关
    """
    fp_index = list(set(catch_index) - set(real_test_labels_index))
    fp_num = np.size(fp_index)
    return fp_index, fp_num


def get_tp(catch_index, real_test_labels_index):
    tp_index = set(catch_index).intersection(set(real_test_labels_index))
    tp_num = np.size(tp_index)
    return tp_index, tp_num


def get_precision(tp_num, fp_num):
    return tp_num / (tp_num + fp_num)


def get_F_score(test_scores, threshold_value, real_test_labels_index):
    """

    Args:
        test_scores: 测试分数集
        threshold_value: 阈值
        real_test_labels_index: 测试数据中真实异常标识的索引

    Returns:

    """
    catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
    catch_num = np.size(catch_index)
    # FP 未标记但超过阈值 实际为正常点单倍误判为异常点
    fp_index, fp_num = get_fp(catch_index, real_test_labels_index)
    # TP 成功检测出的异常
    tp_index, tp_num = get_tp(catch_index, real_test_labels_index)
    # Precision精度
    precision = get_precision(tp_num, fp_num)
