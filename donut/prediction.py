import numpy as np
import six
import tensorflow as tf
from tfsnippet.utils import (VarScopeObject, get_default_session_or_error,
                             reopen_variable_scope)

from donut.model import Donut

__all__ = ['DonutPredictor']

from donut.util.time_util import TimeCounter

from donut.window import BatchSlidingWindow


class DonutPredictor(VarScopeObject):
    """
    Donut 预报器

    Args:
        model (Donut): :class:`Donut` 模型实例
        n_z (int or None): 每个x的z样本数量
            (default 1024，如果为 :obj:`None`,则是一个没有明确抽样维度的样本。)
        mcmc_iteration: (int or tf.Tensor): MCMC缺失点注入的迭代计数。
            (default 10)
        batch_size (int): 用于预测的每个小切片的大小。
            (default 32)
        feed_dict (dict[tf.Tensor, any]): 用户提供的用于预测的提要字典。
            (default :obj:`None`)
        last_point_only (bool): 是否只获取每个窗口最后一点的重构概率
            (default :obj:`True`)
        name (str): Optional name of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): 这个预测器的可选范围
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, n_z=1024, mcmc_iteration=10, batch_size=32,
                 feed_dict=None, last_point_only=True, name=None, scope=None):
        super(DonutPredictor, self).__init__(name=name, scope=scope)
        self._model = model
        self._n_z = n_z
        self._mcmc_iteration = mcmc_iteration
        self._batch_size = batch_size
        # 有提要字典
        if feed_dict is not None:
            # Tensor字典->字典迭代器->字典
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        self._last_point_only = last_point_only

        # 重新打开指定的变量作用域及其原始名称作用域。
        with reopen_variable_scope(self.variable_scope):
            # 输入占位符
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.x_dims], name='input_x')
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.x_dims], name='input_y')

            # 感兴趣的输出
            self.__refactor_probability = self._refactor_probability_without_y = None

    def _get_refactor_probability(self):
        """
        获取重构概率
        Returns:重构概率
        """
        if self.__refactor_probability is None:
            with reopen_variable_scope(self.variable_scope), tf.name_scope('score'):
                self.__refactor_probability = self.model.get_refactor_probability(
                    window=self._input_x,
                    missing=self._input_y,
                    n_z=self._n_z,
                    mcmc_iteration=self._mcmc_iteration,
                    last_point_only=self._last_point_only
                )
        return self.__refactor_probability

    def _get_refactor_probability_without_y(self):
        """
        没有y时获取重构概率
        Returns:没有y时获取的重构概率
        """
        if self._refactor_probability_without_y is None:
            with reopen_variable_scope(self.variable_scope), \
                    tf.name_scope('score_without_y'):
                self._refactor_probability_without_y = self.model.get_refactor_probability(
                    window=self._input_x,
                    n_z=self._n_z,
                    last_point_only=self._last_point_only
                )
        return self._refactor_probability_without_y

    @property
    def model(self):
        """
       获得 :class:`Donut` 模型实例

        Returns:
            Donut: :class:`Donut` 模型实例
        """
        return self._model

    def get_refactor_probability(self, values, missing=None):
        """
        获取指定KPI监测数据的“重构概率”。

        “重建概率”越大，异常点的可能性就越小。如果想要直接表明异常的严重程度，可以取这个分数的负值。

        Args:
            values (np.ndarray): 一维32位浮点数数组，KPI监测数据
            missing (np.ndarray): 一维32位整型数组，指明缺失点
                (default :obj:`None`，如果为 :obj:`None`, 不会进行缺失点注入 )

        Returns:
            np.ndarray: 重构概率，`last_point_only`如果是 :obj:`True`，就是一维数组，
                `last_point_only`如果是 :obj:`False`，就是二维数组
        """
        tc = TimeCounter()
        tc.start()
        with tf.name_scope('DonutPredictor.get_refactor_probability'):
            sess = get_default_session_or_error()
            collector = []
            # 校验参数
            values = np.asarray(values, dtype=np.float32)
            if len(values.shape) != 1:
                raise ValueError('`values` 必须为一维数组')
            # 对每个小切片进行预测
            # 滑动窗口
            sliding_window = BatchSlidingWindow(
                array_size=len(values),
                window_size=self.model.x_dims,
                batch_size=self._batch_size)
            # 有缺失点
            if missing is not None:
                missing = np.asarray(missing, dtype=np.int32)
                # 缺失点shape必须与values的shape相同
                if missing.shape != values.shape:
                    raise ValueError('`missing` 的形状必须与`values`的形状相同 ({} vs {})'.format(missing.shape, values.shape))
                for b_x, b_y in sliding_window.get_iterator([values, missing]):
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._input_x] = b_x
                    feed_dict[self._input_y] = b_y
                    b_r = sess.run(self._get_refactor_probability(), feed_dict=feed_dict)
                    collector.append(b_r)
            else:
                for b_x, in sliding_window.get_iterator([values]):
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._input_x] = b_x
                    b_r = sess.run(self._get_refactor_probability_without_y(),
                                   feed_dict=feed_dict)
                    collector.append(b_r)
            # 合并小切片的数据
            tc.end()
            test_probability_time = tc.get_s() + "秒"
            return np.concatenate(collector, axis=0), test_probability_time
