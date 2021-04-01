import time

import numpy as np

__all__ = ['get_time', 'get_constant_timestamp', 'file_name_converter', 'mini_batch_slices_iterator',
           'BatchSlidingWindow', 'handle_src_threshold_value', 'compute_default_threshold_value']

from donut.demo.out import print_text


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


def get_time(start_time, end_time):
    """
    秒表
    Args:
        start_time: 开始时间
        end_time: 结束时间

    Returns:
        使用时间 秒

    """
    return str(end_time - start_time) + "秒"


def file_name_converter(file_name, test_portion, threshold_value):
    """
    获得缓存路径
    Args:
        file_name: 文件名
        test_portion: 测试数据比例
        threshold_value: 阈值
    Returns:
        缓存文件路径
    """
    return "cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)


def get_constant_timestamp(use_plt, timestamps, step):
    """
    获得连续的时间戳区间字符串
    Args:
        use_plt: 输出格式是否为plt
        timestamps: 时间戳
        step: 时间戳步长
    Returns:
        时间戳区间字符串

    """
    if np.size(timestamps) == 0:
        return None
    else:
        timestamps = np.sort(timestamps)
        print_text(use_plt,timestamps)
        print_text(use_plt, "其中时间戳分布为")
        has_dot = False
        has_head = False
        interval_str = ''
        last = 0
        interval_num = 0

        for i, t in enumerate(timestamps):
            if not has_head:
                interval_str = interval_str + str(t) + " "
                has_head = True
                last = t
                interval_num = interval_num + 1
            else:
                if int(t) == int(last) + int(step):
                    if not has_dot:
                        interval_str = interval_str + "..."
                        has_dot = True
                    else:
                        last = t
                else:
                    interval_str = interval_str + str(last) + " "
                    last = t
                    has_head = False
                    has_dot = False
        return interval_num, interval_str

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


def compute_default_threshold_value(values):
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
        length (int): 一个epoch中csv_data的总长度。
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
            arrays (Iterable[np.ndarray]): 要被迭代的一维数组

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
