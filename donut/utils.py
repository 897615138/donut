import csv
import time

import numpy as np

from donut.data import get_threshold_value_label
from donut.out import print_warn, print_info


class TimeUse:
    def __init__(self, use, name):
        self.use = use
        self.name = name

    def __str__(self):
        return self.name + "【用时：{}】".format(self.use)

    def __eq__(self, other):
        return self.use == other.use

    def __lt__(self, other):
        return self.use < other.use

    def __gt__(self, other):
        return self.use > other.use

    def __cmp__(self, other):
        if self.use < other.use:
            return -1
        elif self.use == other.use:
            return 0
        else:
            return 1


def format_time(atime):
    """
    格式化时间
    Args:
        atime: 时间戳

    Returns:
        格式化后的时间

    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(atime))


def file_name_converter(file_name, test_portion, threshold_value, is_local):
    """
    获得缓存路径
    Args:
        is_local: 本地照片展示
        file_name: 文件名
        test_portion: 测试数据比例
        threshold_value: 阈值
    Returns:
        缓存文件路径
    """
    if is_local:
        return "../cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)
    else:
        return "cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)


def get_constant_timestamp(timestamps, step):
    """
    获得连续的时间戳区间字符串
    Args:
        timestamps: 时间戳
        step: 时间戳步长
    Returns:
        时间戳区间字符串

    """
    if np.size(timestamps) == 0:
        return 0, None
    else:
        timestamps = np.sort(timestamps)
        # print(timestamps)
        has_dot = False
        has_head = False
        time_str = "其中时间戳分布为\n"
        last = 0
        interval_num = 0
        start = 0
        count = 0
        for i, t in enumerate(timestamps):
            if not has_head:
                time_str = time_str + "\t  " + str(t)
                has_head = True
                last = t
                start = t
            else:
                if int(t) == (int(last) + int(step)):
                    if not has_dot:
                        time_str = time_str + "..."
                        has_dot = True
                        interval_num = interval_num + 1
                    last = t
                else:
                    if has_dot and last != start:
                        time_str = time_str + str(last) + ","
                    else:
                        time_str = time_str + ","
                    count = count + 1
                    last = t
                    has_head = False
                    has_dot = False
        return interval_num, time_str


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


def catch_label_v2(use_plt, src_threshold_value,
                   train_scores, train_zero_num, real_train_labels,
                   test_scores, test_zero_num, real_test_labels):
    """
    根据阈值捕获异常点
    Args:
        train_scores: 训练分数
        real_train_labels: 训练异常标签
        use_plt: 使用plt
        test_scores: 测试数据分数
        train_zero_num: 训练数据补齐的0点数量
        test_zero_num: 测试数据补齐的0点数量
        src_threshold_value: 已有的阈值
    Returns:
        捕捉到的异常信息，阈值信息
    """
    test_labels_index = list(np.where(real_test_labels == 1)[0])
    # test_labels_index = [ele for ele in test_labels_index if ele > fill_test_labels[0] + test_zero_num]
    # test_actual_num = np.size(test_scores) - test_zero_num
    real_test_label_num = np.size(real_test_labels)
    test_labels_score = test_scores[test_labels_index]
    accuracy = None
    # 有人为设置的阈值
    if src_threshold_value is not None:
        catch_index_set = np.where(test_scores > float(src_threshold_value))[0].tolist()
        catch_num_set = np.size(catch_index_set)
        if catch_num_set is 0:
            print_info(use_plt, "当前阈值无异常，请确认")
        else:
            accuracy = real_test_label_num / catch_num_set
            if accuracy <= 0.9:
                print_warn(use_plt, "建议提高阈值或使用【默认阈值】")
            elif accuracy > 1:
                print_warn(use_plt, "建议降低阈值或使用【默认阈值】")
    # 默认阈值
    else:
        train_labels_index = list(np.where(real_train_labels == 1)[0])
        train_labels_num_vo = np.size(train_labels_index)
        # 训练数据有异常标签
        if train_labels_num_vo > 0:
            # 去前面的0
            train_labels_index = [ele for ele in train_labels_index if ele > real_train_labels[0] + train_zero_num]
            train_labels_num_vo = np.size(train_labels_index)
            train_labels_score = train_scores[train_labels_index]
            score, catch_num, catch_index, accuracy \
                = compute_default_label_threshold_value(train_labels_score, test_scores, real_test_label_num)
            if accuracy < 0.9:
                print_warn(use_plt, "请注意训练数据异常标注的准确性,或者手动调整阈值")


def compute_default_label_threshold_value(
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
        # FP 未标记但超过阈值 实际为正常点单倍误判为异常点
        fp_index = list(set(catch_index) - set(test_labels_index))
        # special_anomaly_t = test_timestamps[special_anomaly_index]
        # special_anomaly_s = test_scores[special_anomaly_index]
        # special_anomaly_v = test_values[special_anomaly_index]
        # special_anomaly_num = len(special_anomaly_t)
        # interval_num, interval_str = get_constant_timestamp(special_anomaly_t, fill_step)
        # print_text(use_plt, "未标记但超过阈值的点（数量：{}）：\n 共有{}段连续异常 \n ".format(special_anomaly_num, interval_num))
        # 精度
        # precision =
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


def mini_batch_slices_iterator(length, batch_size,
                               ignore_incomplete_batch=False):
    """
    遍历所有小切片。

    Args:
        length (int): 一个epoch中数据的总长度。
        batch_size (int): 每个小切片的尺寸。
        ignore_incomplete_batch (bool): 如果为:obj:`True`, 如果最后一批中包含的项目数量小于' batch_size '，则丢弃该批。
            (default :obj:`False`)

    Yields
        slice: 每一块小切片。最后一个小切片可能包含比' batch_size '更少的索引。
    """
    start = 0
    # 可能会有最后一个切片不完整的情况
    stop = (length // batch_size) * batch_size
    while start < stop:
        yield slice(start, start + batch_size, 1)
        # 下一个切片
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class TimeCounter(object):
    def __init__(self):
        self._start = time.time()
        self._end = time.time()

    def start(self):
        self._start = time.time()

    def end(self):
        self._end = time.time()

    def get(self):
        return self._end - self._start

    def get_s(self):
        return str(self.get())


class BatchSlidingWindow(object):
    """
    用于获取滑动窗口的小切片的迭代器类。

    每个小切片都有“batch_size”窗口。如果最终的切片包含小于' batch_size '窗口，
    即如果`ignore_incomplete_batch`为:obj:`True`，则该批处理将被丢弃。

    Args:
        array_size (int): 迭代数组的长度，至少和窗口数一样
        window_size (int): 窗口的大小，至少为1
        batch_size (int): 每个小切片的大小
        excludes (np.ndarray):
            一维布尔数组，标识是否完全包含一个点。如果一个点被包含，那么包含该点的窗口也包含。
            (default :obj:`None`, 没有点被完全包含)
        shuffle (bool):
            如果为 :obj:`True`, 窗口将按照打乱的顺序进行迭代。
            (default :obj:`False`)
        ignore_incomplete_batch (bool):
            如果为 :obj:`True`, 如果最后的小切片包含的窗口数小于' batch_size '，则丢弃它。
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None, shuffle=False,
                 ignore_incomplete_batch=False):
        # 校验参数
        if window_size < 1:
            raise ValueError('`window_size` 至少为1')
        if array_size < window_size:
            raise ValueError('`array_size` 至少和`window_size`一样大')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('`excludes`的形状应该是{},但是现在是{}'.format(expected_shape, excludes.shape))
        # 计算哪些点不被计算在内
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        # 只要一个窗口里面有一个点包含，就整个窗口都是包含在内的
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            # 不能超出数组范围
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # 生成窗口的结束索引
        indices = np.arange(array_size)[mask]
        # 一列
        self._indices = indices.reshape([-1, 1])

        # 生成窗口的偏移数组
        self._offsets = np.arange(-window_size + 1, 1)

        # 记住参数
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        在“arrays”中迭代每个数组的滑动窗口。
        这个方法是不可重入的，也就是说，调用 :meth:`get_iterator`将使之前获得的任何迭代器失效。

        Args:
            arrays (Iterable[np.ndarray]): 要被迭代的一维数组。

        Yields:
            tuple[np.ndarray]: 每个小切片数组的窗口
        """
        # 校验参数
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` 必须不为空')
        expected_shape = (self._array_size,)
        for i, a in enumerate(arrays):
            if a.shape != expected_shape:
                raise ValueError('`arrays[{}]`的形状应该是{},但是现在为{}'.
                                 format(i, expected_shape, a.shape))
        # 如果需要随机
        if self._shuffle:
            np.random.shuffle(self._indices)

        # 通过小切片迭代
        for s in mini_batch_slices_iterator(length=len(self._indices), batch_size=self._batch_size,
                                            ignore_incomplete_batch=self._ignore_incomplete_batch):
            # 索引加上偏移量
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] for a in arrays)


def split_csv(file_name, begin, num):
    header = ["timestamp", "value", "label", "KPI ID"]
    rows = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        count = 1
        for i in reader:
            if count in range(begin, begin + num):
                rows.append([int(i[0]), float(i[1]), int(i[2]), str(i[3])])
            count = count + 1
    with open(file_name + 'new.csv', 'w', newline='')as f:
        ff = csv.writer(f)
        ff.writerow(header)
        ff.writerows(rows)

# split_csv('../sample_data/real.csv',1,65536)
