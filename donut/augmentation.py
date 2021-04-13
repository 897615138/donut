import numpy as np
from tfsnippet.utils import DocInherit

__all__ = ['DataAugmentation', 'MissingDataInjection']


@DocInherit
class DataAugmentation(object):
    """
    训练过程中的数据增强基础类

    Args:
        mean (float): 训练数据的平均值
        std (float): 训练数据的标准差
    """

    def __init__(self, mean, std):
        if std <= 0.:
            raise ValueError('`std` 必须为正数')
        self._mean = mean
        self._std = std

    def augment(self, values, labels, missing):
        """
        数据增强
        Args:
            values (np.ndarray): 一维32位浮点数组，形状为`(data_length,)`,规则化过得KPI数据
            labels (np.ndarray): 一维32位整数数组，形状为`(data_length,)`,`values`的异常标签
            missing (np.ndarray): 一维32位整型数组，形状为`(data_length,)`,指出缺失点
        Returns:
            np.ndarray: 增强过的KPI值
            np.ndarray: 增强过的异常标签
            np.ndarray: 增强过的缺失值指示器
        """
        if len(values.shape) != 1:
            raise ValueError('`values`必须为一维数组')
        if labels.shape != values.shape:
            raise ValueError('`labels` 的形状必须与`values`的形状相同 ({} vs {})'.format(labels.shape, values.shape))
        if missing.shape != values.shape:
            raise ValueError('`missing` 的形状必须与`values`的形状相同 ({} vs {})'.format(missing.shape, values.shape))
        return self._augment(values, labels, missing)

    def _augment(self, values, labels, missing):
        """
        派生类覆盖本方法实际实现数据增强算法。
        """
        raise NotImplementedError()

    @property
    def mean(self):
        """获得训练数据数据的平均值."""
        return self._mean

    @property
    def std(self):
        """获得训练数据的标准差"""
        return self._std


class MissingDataInjection(DataAugmentation):
    """
    缺失数据注入
    Args:
        train_mean (float): 训练数据的平均值
        train_std (float): 训练数据的标准差
        train_missing_rate (float): 训练数据的缺失值指示
    """

    def __init__(self, train_mean, train_std, train_missing_rate):
        super(MissingDataInjection, self).__init__(train_mean, train_std)
        self._missing_rate = train_missing_rate

    @property
    def missing_rate(self):
        """获得缺失点的比例"""
        return self._missing_rate

    def _augment(self, values, labels, missing):
        inject_y = np.random.binomial(1, self.missing_rate, size=values.shape)
        inject_idx = np.where(inject_y.astype(np.bool))[0]
        values = np.copy(values)
        values[inject_idx] = -self.mean / self.std
        missing = np.copy(missing)
        missing[inject_idx] = 1
        return values, labels, missing
