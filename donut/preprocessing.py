# coding=utf-8
import numpy as np

__all__ = ['complete_timestamp', 'standardize_kpi']


def complete_timestamp(timestamp, src_arrays=None):
    """
    1.完成时间戳，使时间间隔是齐次的。
    2.在缺失点上注入0。
    3.同样，将返回一个指示符数组来指示缺失点。

    Args:
        timestamp (np.ndarray):  时间戳 一维64位整数数组 可以无序 \
            1-D int64 src_array, the timestamp values.It can be unsorted.
        src_arrays (Iterable[np.ndarray]): (values,labels) (数值，标签) 与时间戳相关的一维数组 \
            The 1-D arrays to be filled with zeros according to `timestamp`.

    Returns:
        np.ndarray: 一维64位整型数组 补充完整的时间戳 \
            A 1-D int64 src_array, the completed timestamp.
        np.ndarray: 一维32位整型数组 标注时间戳对应数据是否为缺失数据 \
            A 1-D int32 src_array, indicating whether the data corresponding to the timestamp is missing.
        list[np.ndarray]: 已经进行缺失值补0的时间戳对应值数组\
            The arrays, missing points filled with zeros.(optional, return only if `arrays` is specified)
    """
    # 1.检验数据合法性
    # np src_array-> src_array
    timestamp = np.asarray(timestamp, np.int64)
    # 一维数组检验
    if len(timestamp.shape) != 1:
        raise ValueError('`timestamp` must be a 1-D src_array')
    # 数组是否为空
    has_array = src_arrays is not None
    # np arrays-> arrays
    src_arrays = [np.asarray(src_array) for src_array in src_arrays]
    # 相同维度
    for i, src_array in enumerate(src_arrays):
        if src_array.shape != timestamp.shape:
            raise ValueError(
                'component of src_array must has same shape with timestamp ,shape :timestamp - src_array {} - {}'
                ',src_array index {}'.format(timestamp.shape, src_array.shape, i))
    # 2.检验时间戳数据 补充为有序等间隔时间戳数组
    # 时间戳排序 获得对数组排序后的原数组的对应索引以及有序数组
    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    # 沿给定轴计算离散差分 即获得所有存在的时间戳间隔
    intervals = np.unique(np.diff(timestamp_sorted))
    # 最小的间隔数
    interval = np.min(intervals)
    # 有重复值抛异常 数据有误
    if interval == 0:
        raise ValueError('Duplicated values in `timestamp`')
    # 所有间隔数是否与最小间隔为整除关系
    for i in intervals:
        if i % interval != 0:
            raise ValueError('Not all intervals in `timestamp` are multiples of the minimum interval')
    # 最终时间戳数量为 时间跨度/时间间隔+1
    amount = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    # 重构时间戳数组
    dst_timestamp = np.arange(timestamp_sorted[0], timestamp_sorted[-1] + interval, interval, dtype=np.int64)
    # 初始化缺失点数组与数值与标注数组
    dst_missing = np.ones([amount], dtype=np.int32)
    dst_arrays = [np.zeros([amount], dtype=src_array.dtype) for src_array in src_arrays]
    # 3.填充数值
    # 获得与初始时间戳的差值数组
    diff_with_first = (timestamp_sorted - timestamp_sorted[0])
    # 获得与初始时间戳相差的最小间隔数 即应处的索引值
    diff_intervals_with_first = diff_with_first // interval
    dst_index = np.asarray(diff_intervals_with_first, dtype=np.int)
    # 标记有原值的时间戳为非缺失点
    dst_missing[dst_index] = 0
    if not has_array:
        return dst_timestamp, dst_missing
    # 分别组合
    zip_array = zip(dst_arrays, src_arrays)
    for dst_array, src_array in zip_array:
        dst_array[dst_index] = src_array[src_index]
    return dst_timestamp, dst_missing, dst_arrays


def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if (mean is None) != (std is None):
        raise ValueError('`mean` and `std` must be both None or not None')
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('The shape of `excludes` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(excludes.shape, values.shape))

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()

    return (values - mean) / std, mean, std
