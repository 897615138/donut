import numpy as np

from donut.util.out.out import print_warn


def get_fp(catch_index, real_test_labels_index, real_test_missing):
    """
    FP 未标记但超过阈值 实际为正常点单倍误判为异常点 去除延迟点
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点
    Returns:
        实际为正常点单倍误判为异常点相关
    """
    fp_index = list(set(catch_index) - set(real_test_labels_index))
    for fp in fp_index:
        if fp - 1 in real_test_labels_index or fp - 2 in real_test_labels_index or fp - 3 in real_test_labels_index \
                or fp - 4 in real_test_labels_index or fp - 5 in real_test_labels_index or fp - 6 in real_test_labels_index \
                or fp + 1 in real_test_labels_index or fp + 2 in real_test_labels_index or fp + 3 in real_test_labels_index \
                or fp + 4 in real_test_labels_index or fp + 5 in real_test_labels_index or fp + 6 in real_test_labels_index \
                or fp - 7 in real_test_labels_index or fp + 7 in real_test_labels_index or fp - 8 in real_test_labels_index \
                or fp + 8 in real_test_labels_index or fp + 9 in real_test_labels_index or fp - 9 in real_test_labels_index \
                or fp + 10 in real_test_labels_index or fp - 10 in real_test_labels_index:
            fp_index.remove(fp)
        elif fp - 1 in real_test_missing or fp - 2 in real_test_missing or fp - 3 in real_test_missing \
                or fp - 4 in real_test_missing or fp - 5 in real_test_missing or fp - 6 in real_test_missing \
                or fp + 1 in real_test_missing or fp + 2 in real_test_missing or fp + 3 in real_test_missing \
                or fp + 4 in real_test_missing or fp + 5 in real_test_missing or fp + 6 in real_test_missing \
                or fp - 7 in real_test_missing or fp + 7 in real_test_missing or fp - 8 in real_test_missing \
                or fp + 8 in real_test_missing or fp + 9 in real_test_missing or fp - 9 in real_test_missing \
                or fp + 10 in real_test_missing or fp - 10 in real_test_missing:
            fp_index.remove(fp)
    fp_num = np.size(fp_index)
    return fp_index, fp_num


def get_tp(catch_index, real_test_labels_index, real_test_missing):
    """
    TP 成功检测出的异常
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点

    Returns:
        成功检测出的异常
    """
    tp_index = list(set(catch_index).intersection(set(real_test_labels_index)))
    for tp in tp_index:
        if tp - 1 in real_test_labels_index or tp - 2 in real_test_labels_index or tp - 3 in real_test_labels_index \
                or tp - 4 in real_test_labels_index or tp - 5 in real_test_labels_index or tp - 6 in real_test_labels_index \
                or tp + 1 in real_test_labels_index or tp + 2 in real_test_labels_index or tp + 3 in real_test_labels_index \
                or tp + 4 in real_test_labels_index or tp + 5 in real_test_labels_index or tp + 6 in real_test_labels_index \
                or tp - 7 in real_test_labels_index or tp + 7 in real_test_labels_index or tp - 8 in real_test_labels_index \
                or tp + 8 in real_test_labels_index or tp + 9 in real_test_labels_index or tp - 9 in real_test_labels_index \
                or tp + 10 in real_test_labels_index or tp - 10 in real_test_labels_index:
            tp_index.append(tp)
        elif tp - 1 in real_test_missing or tp - 2 in real_test_missing or tp - 3 in real_test_missing \
                or tp - 4 in real_test_missing or tp - 5 in real_test_missing or tp - 6 in real_test_missing \
                or tp + 1 in real_test_missing or tp + 2 in real_test_missing or tp + 3 in real_test_missing \
                or tp + 4 in real_test_missing or tp + 5 in real_test_missing or tp + 6 in real_test_missing \
                or tp - 7 in real_test_missing or tp + 7 in real_test_missing or tp - 8 in real_test_missing \
                or tp + 8 in real_test_missing or tp + 9 in real_test_missing or tp - 9 in real_test_missing \
                or tp + 10 in real_test_missing or tp - 10 in real_test_missing:
            tp_index.append(tp)
    tp_index = set(tp_index)
    tp_index = list(tp_index)
    tp_num = np.size(tp_index)
    return tp_index, tp_num


def get_precision(tp_num, fp_num):
    """
    精度
    Args:
        tp_num: TP
        fp_num: FP

    Returns:
        精度
    """
    if tp_num + fp_num == 0:
        return None
    return tp_num / (tp_num + fp_num)


def get_fn(catch_index, real_test_labels_index, real_test_missing):
    """
    漏报的异常
    Args:
        catch_index: 认为是异常点
        real_test_labels_index: 实际是异常点

    Returns:
        漏报的异常
    """
    fn_index = list(set(real_test_labels_index) - set(catch_index))
    for fn in fn_index:
        if fn - 1 in real_test_labels_index or fn - 2 in real_test_labels_index or fn - 3 in real_test_labels_index \
                or fn - 4 in real_test_labels_index or fn - 5 in real_test_labels_index or fn - 6 in real_test_labels_index \
                or fn + 1 in real_test_labels_index or fn + 2 in real_test_labels_index or fn + 3 in real_test_labels_index \
                or fn + 4 in real_test_labels_index or fn + 5 in real_test_labels_index or fn + 6 in real_test_labels_index \
                or fn - 7 in real_test_labels_index or fn + 7 in real_test_labels_index or fn - 8 in real_test_labels_index \
                or fn + 8 in real_test_labels_index or fn + 9 in real_test_labels_index or fn - 9 in real_test_labels_index \
                or fn + 10 in real_test_labels_index or fn - 10 in real_test_labels_index:
            fn_index.remove(fn)
        elif fn - 1 in real_test_missing or fn - 2 in real_test_missing or fn - 3 in real_test_missing \
                or fn - 4 in real_test_missing or fn - 5 in real_test_missing or fn - 6 in real_test_missing \
                or fn + 1 in real_test_missing or fn + 2 in real_test_missing or fn + 3 in real_test_missing \
                or fn + 4 in real_test_missing or fn + 5 in real_test_missing or fn + 6 in real_test_missing \
                or fn - 7 in real_test_missing or fn + 7 in real_test_missing or fn - 8 in real_test_missing \
                or fn + 8 in real_test_missing or fn + 9 in real_test_missing or fn - 9 in real_test_missing \
                or fn + 10 in real_test_missing or fn - 10 in real_test_missing:
            fn_index.remove(fn)
    fn_num = np.size(fn_index)
    return fn_index, fn_num


def get_recall(tp_num, fn_num):
    """
    召回率
    Args:
        tp_num: TP
        fn_num: FN

    Returns:
        召回率
    """
    if tp_num + fn_num == 0:
        return None
    return tp_num / (tp_num + fn_num)


def compute_f_score(precision, recall, a=1):
    """
    f-score
    Args:
        precision:精度
        recall: 召回率
        a: 系数

    Returns:
        f-score

    """
    if (a == 0) or a is None or precision + recall == 0:
        return None
    return (a * a + 1) * precision * recall / (a * a * (precision + recall))


def get_F_score(use_plt, test_scores, threshold_value, real_test_labels_index, real_test_missing, a=1):
    """

    Args:
        use_plt: 展示方式
        a: 系数 a>1召回率主导
        test_scores: 测试分数集
        threshold_value: 阈值
        real_test_labels_index: 测试数据中真实异常标识的索引

    Returns:
        最佳f分数相关
    """
    if (a == 0) or a is None:
        print_warn(use_plt, "系数为空或0")
    # 测试数据无异常
    if np.size(real_test_labels_index) == 0:
        print_warn(use_plt, "测试数据无异常,请更换验证用测试数据")
        return None, None, None, None, None, None, None, None, None, None, None
    catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
    catch_num = np.size(catch_index)
    if catch_num == 0:
        return None, None, None, None, None, None, None, None, None, None, None
        # print_warn(use_plt, "该阈值分数没有捕捉到任何异常")
    # FP 未标记但超过阈值 实际为正常点单倍误判为异常点
    fp_index, fp_num = get_fp(catch_index, real_test_labels_index, real_test_missing)
    # TP 成功检测出的异常
    tp_index, tp_num = get_tp(catch_index, real_test_labels_index, real_test_missing)
    # FN  漏报
    fn_index, fn_num = get_fn(catch_index, real_test_labels_index, real_test_missing)
    # Precision精度
    precision = get_precision(tp_num, fp_num)
    if precision is None:
        # print_warn(use_plt, "该分数没有捕捉到任何异常")
        return None, None, None, None, None, None, None, None, None, None, None
    recall = get_recall(tp_num, fn_num)
    if recall is None:
        # print_warn(use_plt, "测试数据无异常,请更换验证用测试数据")
        return None, None, None, None, None, None, None, None, None, None, None
    f_score = compute_f_score(precision, recall, a)
    return f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall
