import os

import numpy as np
import six
import tensorflow as tf
from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import (VarScopeObject,
                             reopen_variable_scope,
                             get_default_session_or_error,
                             ensure_variables_initialized,
                             get_variables_as_dict)

from donut.augmentation import MissingDataInjection
from donut.model import Donut
from donut.util.out.out import print_text
from donut.util.time_util import TimeCounter

__all__ = ['DonutTrainer']

from donut.window import BatchSlidingWindow

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DonutTrainer(VarScopeObject):
    """
    Donut训练器

    Args:
        model (Donut)::class:`Donut` 模型实例
        model_vs (str or tf.compat.v1.VariableScope):如果指定，将只从这个范围收集可训练变量。(default :obj:`None`，如果为 :obj:`None`，将收集所有在当前的图表的可训练的变量。)
        n_z (int or None):每个x要取的z样本数。(default :obj:`None`, 即一个没有明确抽样维度的样本)
        feed_dict (dict[tf.Tensor, any]):用户提供的训练用种子字典。(default :obj:`None`, 表示没有输送)
        valid_feed_dict (dict[tf.Tensor, any]):用户提供的验证用的提要字典。(default :obj:`None` ，即使用`feed_dict`)
        missing_data_injection_rate (float):缺失数据注入的比率。(default 0.01)
        use_regularization_loss (bool):是否在训练损失中添加`tf.GraphKeys.REGULARIZATION_LOSSES`。(default :obj:`True`)
        max_epoch (int or None): 最大的运行遍数。(default 256,如果为:obj:`None`，不会在任何特定的遍数停止，必须至少指定 `max_epoch`和`max_step`中的一个。)
        max_step (int or None):运行最大步长.(default :obj:`None`，如果为 :obj:`None`，将不会在任 何特定的步骤停止。必须至少指定 `max_epoch`和`max_step`中的一个。)
        batch_size (int):训练用的小批数量。(default 256)
        valid_batch_size (int): 验证用的小切片数量。(default 1024)
        valid_step_freq (int):在每个“valid_step_freq”数量的训练步骤之后进行验证。(default 100)
        initial_lr (float):初始学习速率。(default 0.001)
        lr_anneal_epochs (int):在每一个“lr_anneal_epoch”的遍数之后退火学习率。(default 10)
        lr_anneal_factor (float):用这个折现因子来计算学习率，即 learning_rate = learning_rate * lr_anneal_factor。(default 0.75)
        optimizer (Type[tf.train.Optimizer]):TensorFlow优化器的类.(default :class:`tf.train.AdamOptimizer`)
        optimizer_params (dict[str, any] or None):用于构造优化器的命名参数。(default :obj:`None`)
        grad_clip_norm (float or None):根据这个标准渐变裁剪。(default 10.0 ，如果为:obj:`None`按标准禁用渐变裁剪)
        check_numerics (bool):是否在数值问题中添加TensorFlow断言(default :obj:`True`)
        name (str):可选训练器的名称(argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str):可选的训练范围(argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, model_vs=None, n_z=None,
                 feed_dict=None, valid_feed_dict=None,
                 missing_data_injection_rate=0.01,
                 use_regularization_loss=True,
                 max_epoch=256, max_step=None, batch_size=256,
                 valid_batch_size=1024, valid_step_freq=100,
                 initial_lr=0.001, lr_anneal_epochs=10, lr_anneal_factor=0.75,
                 optimizer=tf.train.AdamOptimizer, optimizer_params=None,
                 grad_clip_norm=10.0, check_numerics=True,
                 name=None, scope=None):
        super(DonutTrainer, self).__init__(name=name, scope=scope)
        # 记忆参数
        self._model = model
        self._n_z = n_z
        if feed_dict is not None:
            # 迭代器->字典
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        if valid_feed_dict is not None:
            self._valid_feed_dict = dict(six.iteritems(valid_feed_dict))
        else:
            # 为空使用feed_dict
            self._valid_feed_dict = self._feed_dict
        self._missing_data_injection_rate = missing_data_injection_rate
        # 必须有最大限制
        if max_epoch is None and max_step is None:
            raise ValueError('`max_epoch`和`max_step`至少有一个被指定')
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._initial_lr = initial_lr
        self._lr_anneal_epochs = lr_anneal_epochs
        self._lr_anneal_factor = lr_anneal_factor

        # 构建训练器
        with reopen_variable_scope(self.variable_scope):
            # 输入占位符 x，y输入列都为x维数，学习率为一维
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.x_dims], name='input_x')
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.x_dims], name='input_y')
            self._learning_rate = tf.placeholder(
                dtype=tf.float32, shape=(), name='learning_rate')
            # 弥补训练损失
            with tf.name_scope('loss'):
                loss = model.get_training_loss(
                    x=self._input_x, y=self._input_y, n_z=n_z)
                if use_regularization_loss:
                    loss += tf.losses.get_regularization_loss()
                self._loss = loss

            # 获得训练变量
            train_params = get_variables_as_dict(scope=model_vs, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_params = train_params

            # 创建训练器
            if optimizer_params is None:
                optimizer_params = {}
            else:
                optimizer_params = dict(six.iteritems(optimizer_params))
            optimizer_params['learning_rate'] = self._learning_rate
            # 默认 实现Adam算法的优化器。
            self._optimizer = optimizer(**optimizer_params)

            # 推导训练梯度 对var_list中的变量计算loss的梯度
            # 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
            origin_grad_vars = self._optimizer.compute_gradients(
                self._loss, list(six.itervalues(self._train_params))
            )
            grad_vars = []
            for grad, var in origin_grad_vars:
                if grad is not None and var is not None:
                    if grad_clip_norm:
                        # 剪辑张量值到最大l2范数。
                        grad = tf.clip_by_norm(grad, grad_clip_norm)
                    if check_numerics:
                        # 检查一个张量中的NaN和Inf值。
                        grad = tf.check_numerics(
                            grad,
                            'gradient for {} has numeric issue'.format(var.name)
                        )
                    grad_vars.append((grad, var))

            # 构建训练操作
            # 模型的全局步长 常量初始化
            self._global_step = tf.get_variable(
                dtype=tf.int64, name='global_step', trainable=False,
                initializer=tf.constant(0, dtype=tf.int64)
            )
            # 保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行。需要在训练操作之前完成的操作。
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
                self._train_op = self._optimizer.apply_gradients(
                    grad_vars, global_step=self._global_step)

            # 如果指定了`summary_dir`，则为训练摘要
            with tf.name_scope('summary'):
                self._summary_op = tf.summary.merge([
                    tf.summary.histogram(v.name.rsplit(':', 1)[0], v)
                    for v in six.itervalues(self._train_params)
                ])
            # 变量的初始化
            self._trainer_initializer = tf.variables_initializer(
                list(six.itervalues(self.get_variables_as_dict()))
            )

    @property
    def model(self):
        """
        获得 :class:`Donut` 模型实例

        Returns:
            Donut: :class:`Donut`模型实例
        """
        return self._model

    def fit(self, use_plt, train_values, train_labels, train_missing,
            test_values, test_labels, test_missing,
            train_mean, train_std, valid_num, excludes=None, summary_dir=None):
        """
        根据所给数据训练:class:`Donut`模型

        Args:
            use_plt: 展示方式
            valid_num: 测试数据数量
            test_missing: 测试数据缺失值
            test_labels: 测试数据异常标注
            test_values: 测试数据缺失值
            train_values (np.ndarray):一维32位浮点数组，标准化的KPI数据
            train_labels (np.ndarray):一维32位整型数组，异常标签
            train_missing (np.ndarray):一维32位数组，指出缺失点
            train_mean (float):标准化之前的平均值
            train_std (float):标准化之前的标准差
            excludes (np.ndarray):一维布尔数组，表明是否包含该点，如果包含，任何包含该点的窗口都包含在内(default :obj:`None`,没有点包含)
            summary_dir (str)::class:`tf.summary.FileWriter`的可选的概要目录。(default :obj:`None`,无目录)
        """
        # 获得默认session
        sess = get_default_session_or_error()
        # 转化训练集
        train_values = np.asarray(train_values, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.int32)
        train_missing = np.asarray(train_missing, dtype=np.int32)
        # 一维数组检验
        if len(train_values.shape) != 1:
            raise ValueError('values必须是一维数组')
        # 标注维数必须与数值维数相同
        if train_labels.shape != train_values.shape:
            raise ValueError('`labels` 的形状必须与`values`的形状相同 ({} vs {})'.format(train_labels.shape, train_values.shape))
        # 缺失点维数必须与数值维数相同
        if train_missing.shape != train_values.shape:
            raise ValueError('`missing` 的形状必须与`values`的形状相同 ({} vs {})'.format(train_missing.shape, train_values.shape))
        v_y = np.logical_or(test_labels, test_missing).astype(np.int32)
        if excludes is None:
            train_excludes, valid_excludes = None, None
        else:
            train_excludes, valid_excludes = excludes[:-valid_num], excludes[-valid_num:]
        # 数据扩展对象和滑动窗口迭代器
        aug = MissingDataInjection(train_mean, train_std, self._missing_data_injection_rate)
        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.x_dims,
            batch_size=self._batch_size,
            excludes=train_excludes,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(test_values),
            window_size=self.model.x_dims,
            batch_size=self._valid_batch_size,
            excludes=valid_excludes,
        )

        # 初始化训练器变量并检验
        sess.run(self._trainer_initializer)
        ensure_variables_initialized(self._train_params)
        # 循环训练
        lr = self._initial_lr
        epoch_list = []
        lr_list = []
        train_message = []
        tc = TimeCounter()
        with TrainLoop(
                param_vars=self._train_params,
                early_stopping=True,
                summary_dir=summary_dir,
                max_epoch=self._max_epoch,
                max_step=self._max_step) as loop:  # type: TrainLoop
            loop.print_training_summary()
            tc.start()
            for epoch in loop.iter_epochs():
                aug_values, aug_labels, aug_missing = aug.augment(train_values, train_labels, train_missing)
                label_or_missing = np.logical_or(aug_labels, aug_missing).astype(np.int32)
                train_iterator = train_sliding_window.get_iterator([aug_values, label_or_missing])
                for step, (batch_x, batch_y) in loop.iter_steps(train_iterator):
                    # 一次训练
                    feed_dict_train = dict(six.iteritems(self._feed_dict))
                    feed_dict_train[self._learning_rate] = lr
                    feed_dict_train[self._input_x] = batch_x
                    feed_dict_train[self._input_y] = batch_y
                    loss, _ = sess.run([self._loss, self._train_op], feed_dict=feed_dict_train)
                    loop.collect_metrics({'loss': loss})
                    if step % self._valid_step_freq == 0:
                        # 收集变量目录
                        if summary_dir is not None:
                            loop.add_summary(sess.run(self._summary_op))
                        # 批量进行验证
                        with loop.timeit('valid_time'), loop.metric_collector('valid_loss') as mc:
                            it = valid_sliding_window.get_iterator([test_values, v_y])
                            for b_x, b_y in it:
                                feed_dict_train = dict(six.iteritems(self._valid_feed_dict))
                                feed_dict_train[self._input_x] = b_x
                                feed_dict_train[self._input_y] = b_y
                                loss = sess.run(self._loss, feed_dict=feed_dict_train)
                                mc.collect(loss, weight=len(b_x))
                        # 打印最近步骤的日志
                        suffix, message = loop.print_logs(False)
                        train_message.append(suffix)
                        train_message.append(message)
                        print_text(use_plt, suffix)
                        print_text(use_plt, message)
                # 退火学习率
                if self._lr_anneal_epochs and epoch % self._lr_anneal_epochs == 0:
                    lr *= self._lr_anneal_factor
                    loop.println('Learning rate decreased to {}'.format(lr), with_tag=True)
                    epoch_list.append(epoch)
                    lr_list.append(lr)
            tc.end()
        return epoch_list, lr_list, tc.get_s() + "秒", train_message
