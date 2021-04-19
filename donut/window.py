import numpy as np


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
