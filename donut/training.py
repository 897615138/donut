import six
import numpy as np
import tensorflow as tf
from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import (VarScopeObject,
                             reopen_variable_scope,
                             get_default_session_or_error,
                             ensure_variables_initialized,
                             get_variables_as_dict)

from .augmentation import MissingDataInjection
from .model import Donut
from .utils import BatchSlidingWindow

__all__ = ['DonutTrainer']


class DonutTrainer(VarScopeObject):
    """
    Donut训练器

    Args:
        model (Donut):
            :class:`Donut` 模型实例
        model_vs (str or tf.compat.v1.VariableScope):
            如果指定，将只从这个范围收集可训练变量。
            (default :obj:`None`，如果为 :obj:`None`，将收集所有在当前的图表的可训练的变量。)
        x_z_n (int or None):
            每个x要取的z样本数。
            (default :obj:`None`, 即一个没有明确抽样维度的样本)
        feed_dict (dict[tf.Tensor, any]):
            用户提供的训练用种子字典。
            (default :obj:`None`, 表示没有输送)
        valid_feed_dict (dict[tf.Tensor, any]):
            用户提供的提要字典用于验证。
            (default :obj:`None` ，即使用`feed_dict`)
        missing_data_injection_rate (float):
            缺失数据注入的比率。
            (default 0.01)
        use_regularization_loss (bool):
            是否在训练损失中添加`tf.GraphKeys.REGULARIZATION_LOSSES`。
            (default :obj:`True`)
        max_epoch (int or None):
            最大的运行遍数。
            (default 256,如果为:obj:`None`，不会在任何特定的遍数停止，必须至少指定 `max_epoch`和`max_step`中的一个。)
        max_step (int or None):
            运行最大步长.
            (default :obj:`None`，如果为 :obj:`None`，将不会在任何特定的步骤停止。必须至少指定 `max_epoch`和`max_step`中的一个。)
        batch_size (int):
            训练用的小批数量。
            (default 256)
        valid_batch_size (int):
            验证用的小批数量。
            (default 1024)
        valid_step_freq (int):
            在每个‘valid_step_freq’数量的训练步骤之后进行验证。
            (default 100)
        lr_initial (float):
            初始学习速率.
            (default 0.001)
        lr_anneal_epochs (int):
            在每一个‘lr_anneal_epoch’的遍数之后退火学习率。
            (default 10)
        lr_anneal_factor (float):
            用这个折现因子来计算学习率，即 learning_rate = learning_rate * lr_anneal_factor。
            (default 0.75)
        optimizer (Type[tf.train.Optimizer]):
            TensorFlow优化器的类.
            (default :class:`tf.train.AdamOptimizer`)
        optimizer_params (dict[str, any] or None):
            用于构造优化器的命名参数。
            (default :obj:`None`)
        grad_clip_norm (float or None):
            根据这个标准渐变裁剪。
            (default 10.0 ，如果为:obj:`None`按标准禁用渐变裁剪)
        check_numerics (bool):
            是否在数值问题中添加TensorFlow断言
            (default :obj:`True`)
        name (str):
            可选训练器的名称
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str):
            可选的训练范围
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, model_vs=None, x_z_n=None,
                 feed_dict=None, valid_feed_dict=None,
                 missing_data_injection_rate=0.01,
                 use_regularization_loss=True,
                 max_epoch=256, max_step=None, batch_size=256,
                 valid_batch_size=1024, valid_step_freq=100,
                 lr_initial=0.001, lr_anneal_epochs=10, lr_anneal_factor=0.75,
                 optimizer=tf.train.AdamOptimizer, optimizer_params=None,
                 grad_clip_norm=10.0, check_numerics=True,
                 name=None, scope=None):
        super(DonutTrainer, self).__init__(name=name, scope=scope)

        # 记忆参数
        self._model = model
        self._n_z = x_z_n
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
            raise ValueError('At least one of `max_epoch` and `max_step` should be specified')
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._lr_initial = lr_initial
        self._lr_anneal_epochs = lr_anneal_epochs
        self._lr_anneal_factor = lr_anneal_factor

        # 构建训练器
        with reopen_variable_scope(self.variable_scope):
            # 模型的全局步长
            self._global_step = tf.get_variable(
                dtype=tf.int64, name='global_step', trainable=False,
                initializer=tf.constant(0, dtype=tf.int64)
            )
            # 输入占位符
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.x_dims], name='input_x')
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.x_dims], name='input_y')
            self._learning_rate = tf.placeholder(
                dtype=tf.float32, shape=(), name='learning_rate')

            # compose the training loss
            with tf.name_scope('loss'):
                loss = model.get_training_loss(
                    x=self._input_x, y=self._input_y, n_z=x_z_n)
                if use_regularization_loss:
                    loss += tf.losses.get_regularization_loss()
                self._loss = loss

            # get the training variables
            train_params = get_variables_as_dict(
                scope=model_vs, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_params = train_params

            # create the trainer
            if optimizer_params is None:
                optimizer_params = {}
            else:
                optimizer_params = dict(six.iteritems(optimizer_params))
            optimizer_params['learning_rate'] = self._learning_rate
            self._optimizer = optimizer(**optimizer_params)

            # derive the training gradient
            origin_grad_vars = self._optimizer.compute_gradients(
                self._loss, list(six.itervalues(self._train_params))
            )
            grad_vars = []
            for grad, var in origin_grad_vars:
                if grad is not None and var is not None:
                    if grad_clip_norm:
                        grad = tf.clip_by_norm(grad, grad_clip_norm)
                    if check_numerics:
                        grad = tf.check_numerics(
                            grad,
                            'gradient for {} has numeric issue'.format(var.name)
                        )
                    grad_vars.append((grad, var))

            # build the training op
            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self._train_op = self._optimizer.apply_gradients(
                    grad_vars, global_step=self._global_step)

            # the training summary in case `summary_dir` is specified
            with tf.name_scope('summary'):
                self._summary_op = tf.summary.merge([
                    tf.summary.histogram(v.name.rsplit(':', 1)[0], v)
                    for v in six.itervalues(self._train_params)
                ])

            # initializer for the variables
            self._trainer_initializer = tf.variables_initializer(
                list(six.itervalues(self.get_variables_as_dict()))
            )

    @property
    def model(self):
        """
        Get the :class:`Donut` model instance.

        Returns:
            Donut: The :class:`Donut` model instance.
        """
        return self._model

    def fit(self, values, labels, missing, mean, std, excludes=None,
            valid_portion=0.3, summary_dir=None):
        """
        Train the :class:`Donut` model with given data.

        Args:
            values (np.ndarray): 1-D `float32` array, the standardized
                KPI observations.
            labels (np.ndarray): 1-D `int32` array, the anomaly labels.
            missing (np.ndarray): 1-D `int32` array, the indicator of
                missing points.
            mean (float): The mean of KPI observations before standardization.
            std (float): The standard deviation of KPI observations before
                standardization.
            excludes (np.ndarray): 1-D `bool` array, indicators of whether
                or not to totally exclude a point.  If a point is excluded,
                any window which contains that point is excluded.
                (default :obj:`None`, no point is totally excluded)
            valid_portion (float): Ratio of validation data out of all the
                specified training data. (default 0.3)
            summary_dir (str): Optional summary directory for
                :class:`tf.summary.FileWriter`. (default :obj:`None`,
                summary is disabled)
        """
        sess = get_default_session_or_error()

        # split the training & validation set
        values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        missing = np.asarray(missing, dtype=np.int32)
        if len(values.shape) != 1:
            raise ValueError('`values` must be a 1-D array')
        if labels.shape != values.shape:
            raise ValueError('The shape of `labels` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(labels.shape, values.shape))
        if missing.shape != values.shape:
            raise ValueError('The shape of `missing` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(missing.shape, values.shape))

        n = int(len(values) * valid_portion)
        train_values, v_x = values[:-n], values[-n:]
        train_labels, valid_labels = labels[:-n], labels[-n:]
        train_missing, valid_missing = missing[:-n], missing[-n:]
        v_y = np.logical_or(valid_labels, valid_missing).astype(np.int32)
        if excludes is None:
            train_excludes, valid_excludes = None, None
        else:
            train_excludes, valid_excludes = excludes[:-n], excludes[-n:]

        # data augmentation object and the sliding window iterator
        aug = MissingDataInjection(mean, std, self._missing_data_injection_rate)
        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.x_dims,
            batch_size=self._batch_size,
            excludes=train_excludes,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(v_x),
            window_size=self.model.x_dims,
            batch_size=self._valid_batch_size,
            excludes=valid_excludes,
        )

        # initialize the variables of the trainer, and the model
        sess.run(self._trainer_initializer)
        ensure_variables_initialized(self._train_params)

        # training loop
        lr = self._lr_initial
        with TrainLoop(
                param_vars=self._train_params,
                early_stopping=True,
                summary_dir=summary_dir,
                max_epoch=self._max_epoch,
                max_step=self._max_step) as loop:  # type: TrainLoop
            loop.print_training_summary()

            for epoch in loop.iter_epochs():
                x, y1, y2 = aug.augment(
                    train_values, train_labels, train_missing)
                y = np.logical_or(y1, y2).astype(np.int32)

                train_iterator = train_sliding_window.get_iterator([x, y])
                for step, (batch_x, batch_y) in loop.iter_steps(train_iterator):
                    # run a training step
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._learning_rate] = lr
                    feed_dict[self._input_x] = batch_x
                    feed_dict[self._input_y] = batch_y
                    loss, _ = sess.run(
                        [self._loss, self._train_op], feed_dict=feed_dict)
                    loop.collect_metrics({'loss': loss})

                    if step % self._valid_step_freq == 0:
                        # collect variable summaries
                        if summary_dir is not None:
                            loop.add_summary(sess.run(self._summary_op))

                        # do validation in batches
                        with loop.timeit('valid_time'), \
                                loop.metric_collector('valid_loss') as mc:
                            v_it = valid_sliding_window.get_iterator([v_x, v_y])
                            for b_v_x, b_v_y in v_it:
                                feed_dict = dict(
                                    six.iteritems(self._valid_feed_dict))
                                feed_dict[self._input_x] = b_v_x
                                feed_dict[self._input_y] = b_v_y
                                loss = sess.run(self._loss, feed_dict=feed_dict)
                                mc.collect(loss, weight=len(b_v_x))

                        # print the logs of recent steps
                        loop.print_logs()

                # anneal the learning rate
                if self._lr_anneal_epochs and \
                        epoch % self._lr_anneal_epochs == 0:
                    lr *= self._lr_anneal_factor
                    loop.println('Learning rate decreased to {}'.format(lr),
                                 with_tag=True)
