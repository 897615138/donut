import numpy as np

__all__ = ['complete_timestamp', 'standardize_kpi']


def complete_timestamp(timestamp, arrays=None):
    """
    1.补齐时间戳，使时间间隔是齐次的。
    2.标记非缺失点。

    Args:
        timestamp (np.ndarray):
            时间戳 一维64位整数数组 可以无序
        arrays (Iterable[np.ndarray]): (values,labels)
            (数值，标签) 与时间戳相关的一维数组

    Returns:
        np.ndarray:
            一维64位整型数组 补充完整的时间戳
        np.ndarray:
            一维32位整型数组 标注时间戳对应数据是否为缺失数据
        list[np.ndarray]:
            已经进行缺失值补0的时间戳对应值数组
            (可选，仅当指定arrays时返回)
    """
    # 1.检验数据合法性
    # np src_array-> src_array
    timestamp = np.asarray(timestamp, np.int64)
    # 一维数组检验
    if len(timestamp.shape) != 1:
        raise ValueError('`timestamp`必须为一维数组')
    # 数组是否为空
    has_array = arrays is not None
    # np arrays-> arrays
    src_arrays = [np.asarray(src_array) for src_array in arrays]
    # 相同维度
    for i, src_array in enumerate(src_arrays):
        if src_array.shape != timestamp.shape:
            raise ValueError('`timestamp` 的形状必须与`src_array`的形状相同 ({} vs {}) src_array index {}'
                             .format(timestamp.shape, src_array.shape, i))
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
        r_timestamp = timestamp - np.unique(timestamp)
        s_timestamp = set(r_timestamp)
        for t in s_timestamp:
            raise ValueError('`timestamp`中有重复值')
    # 所有间隔数是否与最小间隔为整除关系
    for i in intervals:
        if i % interval != 0:
            raise ValueError('并不是“timestamp”中的所有时间间隔都是最小时间间隔的倍数')
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
    标准化 Standardize
    Args:
        values (np.ndarray):
            一维浮点数组，KPI数据
        mean (float):
            如果不为None，将使用平均值来标准化values。
            默认为None。
            注意:mean和std同时为None或不为None。
        std (float):
            标准差， 与mean类似。
        excludes (np.ndarray):
            可选，一维布尔或者32位整型数组,指示是否应该排除某个点计算mean和std。如果mean和std不是None，则忽略。
            默认为None。

    Returns:
        np.ndarray:
            标准化数据。
        float:
            计算得出的平均值或提供的平均值。
        float:
            算得出的标准差或提供的标准差。
    """
    # 1.转化数组格式 校验数据类型
    values = np.asarray(values, dtype=np.float32)
    # 一维数组检验
    if len(values.shape) != 1:
        raise ValueError('values 必须为一维数组')
    # mean 和 std 同时为None或非None
    if (mean is None) != (std is None):
        raise ValueError('mean和std 必须同为None或者同不为None')
    # 排除点维数必须与数值维数相同
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('`excludes` 的形状必须与`values`的形状相同 ({} vs {})'.format(excludes.shape, values.shape))
    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()
    return (values - mean) / std, mean, std
