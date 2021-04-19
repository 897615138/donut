import numpy as np

from donut.out import print_warn


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
    """
    TP 成功检测出的异常
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点

    Returns:

    """
    tp_index = set(catch_index).intersection(set(real_test_labels_index))
    tp_num = np.size(tp_index)
    return tp_index, tp_num


def get_precision(tp_num, fp_num):
    if tp_num + fp_num == 0:
        return None
    return tp_num / (tp_num + fp_num)


def get_fn(catch_index, real_test_labels_index):
    """
    漏报的异常
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点

    Returns:
        漏报的异常
    """
    fn_index = list(set(real_test_labels_index) - set(catch_index))
    fn_num = np.size(fn_index)
    return fn_index, fn_num


def get_recall(tp_num, fn_num):
    if tp_num + fn_num == 0:
        return None
    return tp_num / (tp_num + fn_num)


def compute_f_score(precision, recall, a=1):
    if (a == 0) or a is None or precision + recall == 0:
        return None
    return (a * a + 1) * precision * recall / (a * a * (precision + recall))


def get_F_score(use_plt, test_scores, threshold_value, real_test_labels_index, a=1):
    """

    Args:
        use_plt: 展示方式
        a: 系数 a>1召回率主导
        test_scores: 测试分数集
        threshold_value: 阈值
        real_test_labels_index: 测试数据中真实异常标识的索引

    Returns:
        f分数
    """
    if (a == 0) or a is None:
        print_warn(use_plt, "系数为空或0")
    # 测试数据无异常
    if np.size(real_test_labels_index) == 0:
        print_warn(use_plt, "测试数据无异常,请更换验证用测试数据")
        return None
    catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
    catch_num = np.size(catch_index)
    if catch_num == 0:
        return None
        # print_warn(use_plt, "该阈值分数没有捕捉到任何异常")
    # FP 未标记但超过阈值 实际为正常点单倍误判为异常点
    fp_index, fp_num = get_fp(catch_index, real_test_labels_index)
    # TP 成功检测出的异常
    tp_index, tp_num = get_tp(catch_index, real_test_labels_index)
    # FN  漏报
    fn_index, fn_num = get_fn(catch_index, real_test_labels_index)
    # Precision精度
    precision = get_precision(tp_num, fp_num)
    if precision is None:
        # print_warn(use_plt, "该分数没有捕捉到任何异常")
        return None
    recall = get_recall(tp_num, fn_num)
    if recall is None:
        # print_warn(use_plt, "测试数据无异常,请更换验证用测试数据")
        return None
    f_score = compute_f_score(precision, recall, a)
    return f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall
