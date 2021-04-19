import numpy as np

from donut.assessment import get_F_score
from donut.util.out.out import print_warn, print_info


def catch_label_v1(use_plt, test_labels, test_scores, zero_num, threshold_value):
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
    # 有人为设置阈值
    if threshold_value is not None:
        catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
        if catch_num is 0:
            print_warn(use_plt, "当前阈值无异常，请确认")
        else:
            accuracy = labels_num / catch_num
            if accuracy <= 0.9:
                print_warn(use_plt, "建议提高阈值或使用【默认阈值】")
            elif accuracy > 1:
                print_warn(use_plt, "建议降低阈值或使用【默认阈值】")
    # 默认阈值
    # 无异常标签
    elif len(labels_index) == 0:
        threshold_value = compute_default_threshold_value_v1(test_scores)
        catch_index = np.where(test_scores > float(threshold_value))[0].tolist()
        catch_num = np.size(catch_index)
    else:
        labels_score = test_scores[labels_index]
        threshold_value, catch_num, catch_index, accuracy = \
            get_threshold_value_label(use_plt, labels_score, test_scores, labels_num)
        # 准确度
        if catch_num is not 0:
            accuracy = labels_num / catch_num
    return labels_num, catch_num, catch_index, labels_index, threshold_value, accuracy


def catch_label_v2(use_plt, threshold_value, train_scores, real_train_labels, test_scores, real_test_labels):
    """
    根据阈值捕获异常点
    Args:
        use_plt: 使用plt
        threshold_value: 已有的阈值
        real_test_labels: 测试异常标签
        train_scores: 训练分数
        real_train_labels: 训练异常标签
        test_scores: 测试数据分数

    Returns:
        捕捉到的异常信息，阈值信息
    """
    real_test_labels_index = list(np.where(real_test_labels == 1)[0])
    real_train_labels_index = list(np.where(real_train_labels == 1)[0])
    real_train_label_scores = train_scores[real_train_labels_index]
    real_test_label_scores = test_scores[real_test_labels_index]
    # 有人为设置的阈值
    if threshold_value is not None:
        f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall = \
            get_F_score(use_plt, test_scores, threshold_value, real_test_labels_index, 1)
        if f_score is None:
            print_info(use_plt, "当前阈值无异常，请确认")
        else:
            if f_score < 0.7:
                print_warn(use_plt, "建议调整阈值分数或使用【默认阈值】以获得更好的效果（F—score）")
    # 默认阈值
    else:
        threshold_value, catch_num, catch_index, f_score, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall = \
            compute_default_label_threshold_value(use_plt, test_scores, real_test_labels_index, real_train_label_scores)
    return threshold_value, catch_num, catch_index, f_score, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall


def compute_default_label_threshold_value(use_plt, test_scores, real_test_labels_index, real_train_label_scores):
    print_info(use_plt, "开始计算默认阈值")
    # 降序训练数据中的异常标签对应分值
    sorted_label_scores = np.asarray(real_train_label_scores)
    sorted_label_scores = sorted_label_scores[np.argsort(-sorted_label_scores)]
    lis = []
    for i, score in enumerate(sorted_label_scores):
        f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall = get_F_score(
            use_plt, test_scores, score, real_test_labels_index, 1)
        if f_score is not None:
            catch = {"score": score, "num": catch_num, "index": catch_index, "f": f_score,
                     "fpi": fp_index, "fpn": fp_num, "tpi": tp_index, "tpn": tp_num, "fni": fn_index, "fnn": fn_num,
                     "p": precision, "r": recall}
            lis.append(catch)
    # 字典按照生序排序 取最大的准确度
    if len(lis) > 0:
        lis = sorted(lis, key=lambda dict_catch: (dict_catch['f'], dict_catch['score']))
        catch = lis[- 1]
        # 最优F-score
        return catch.get("score"), catch.get("num"), catch.get("index"), catch.get("f"), \
               catch.get("fpi"), catch.get("fpn"), catch.get("tpi"), catch.get("tpn"), catch.get("fni"), catch.get(
            "fnn"), catch.get("p"), catch.get("r")
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None


def compute_default_label_threshold_value_v1(
        labels_score, test_score, test_labels_num_vo, test_actual_num, test_labels_index):
    """
    计算默认阈值 【训练数据有异常标注】
    Args:
        test_labels_index: 异常索引
        test_actual_num: 测试数据实际有效分数数据数量
        labels_score: 训练异常标注分数
        test_score: 测试分数
        test_labels_num_vo: 测试异常标注数量

    Returns:
        阈值分数，捕捉到的异常数量，捕捉到的异常索引，准确率
    """
    # 降序
    merge_score = np.asarray(labels_score)
    # merge_score = np.asarray(set(train_labels_score).intersection(set(test_labels_score)))
    merge_score = merge_score[np.argsort(-merge_score)]
    lis = []
    for i, score in enumerate(merge_score):
        catch_index = np.where(test_score > float(score))[0].tolist()
        catch_num = np.size(catch_index)
        accuracy = test_labels_num_vo / catch_num
        if 0.9 < accuracy <= 1:
            # 存在就存储
            catch = {"score": score, "num": catch_num, "index": catch_index, "accuracy": accuracy}
            lis.append(catch)
    # 字典按照生序排序 取最大的准确度
    if len(lis) > 0:
        sorted(lis, key=lambda dict_catch: (dict_catch['accuracy'], dict_catch['score']))
        catch = lis[- 1]
        return catch.get("score"), catch.get("num"), catch.get("index"), catch.get("accuracy")
    # 没有满足0.9标准的
    score = np.min(merge_score)
    catch_index = np.where(test_score >= float(score)).tolist()
    catch_num = np.size(catch_index)
    accuracy = None
    if catch_num is not 0:
        accuracy = test_labels_num_vo / catch_num
    return score, catch_num, catch_index, accuracy


def compute_default_threshold_value_v1(values):
    """
    默认阈值 至多10个数据
    Args:
        values: 数据集
    Returns: 默认阈值
    """
    values = np.sort(values)
    num = np.size(values)
    count = round(num * 0.01 / 100)
    if count >= 10:
        return values[num - 10]
    elif count <= 0:
        return 2 * values[num - 1] - values[num - 2]
    else:
        return values[num - count]


def get_threshold_value_label(use_plt, labels_score, test_score, labels_num):
    """
    带异常标签的默认阈值
    Args:
        use_plt: 展示方式
        labels_num: 标签数量
        test_score: 所有分值
        labels_score: 异常标签对应分数

    Returns:
        默认阈值
    """
    # 降序
    labels_score = labels_score[np.argsort(-labels_score)]
    lis = []
    for i, score in enumerate(labels_score):
        catch_index = np.where(test_score > float(score))[0].tolist()
        catch_num = np.size(catch_index)
        if catch_num == 0:
            continue
        accuracy = labels_num / catch_num
        if 0.9 < accuracy <= 1:
            # 存在就存储
            catch = {"score": score, "num": catch_num, "index": catch_index, "accuracy": accuracy}
            lis.append(catch)
    # print_text(use_plt, lis)
    # 字典按照生序排序 取最大的准确度
    if len(lis) > 0:
        sorted(lis, key=lambda dict_catch: (dict_catch['accuracy'], dict_catch['score']))
        catch = lis[- 1]
        # print_text(use_plt, catch)
        return catch.get("score"), catch.get("num"), catch.get("index"), catch.get("accuracy")
    # 没有满足0.9标准的
    score = np.min(labels_score)
    print_warn(use_plt, "请注意异常标注的准确性")
    catch_index = np.where(test_score >= float(score)).tolist()
    catch_num = np.size(catch_index)
    accuracy = None
    if catch_num is not 0:
        accuracy = labels_num / catch_num
    return score, catch_num, catch_index, accuracy


def handle_src_threshold_value(src_threshold_value):
    """
    处理初始阈值
    Args:
        src_threshold_value: 初始阈值
    Returns:
        初始阈值
    """
    if src_threshold_value.isdecimal():
        src_threshold_value = float(src_threshold_value)
    else:
        src_threshold_value = None
    return src_threshold_value
