from functools import partial

import tensorflow as tf
from tfsnippet.distributions import Normal
from tfsnippet.modules import VAE, Lambda, Module
from tfsnippet.stochastic import validate_n_samples
from tfsnippet.utils import (VarScopeObject, reopen_variable_scope, is_integer)
from tfsnippet.variational import VariationalInference

from .reconstruction import iterative_masked_reconstruct

__all__ = ['Donut']


def softplus_std(inputs, units, epsilon, name):
    return tf.nn.softplus(tf.layers.dense(inputs, units, name=name)) + epsilon


def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    with tf.variable_scope('hidden'):
        h = h_for_dist(inputs)
    return {
        'mean': mean_layer(h),
        'std': std_layer(h),
    }


class Donut(VarScopeObject):
    """
    构造Donut

    `get_training_loss`方法 得出训练损失 :class:`tf.Tensor`
    `get_score`方法 获取重构概率 :class:`tf.Tensor`.

    Note:
        :class:`Donut` 实例在`get_training_loss`方法或`get_score`方法被调用的时候才会构建。
        在保存或恢复模型参数之前构建:class:`donut.DonutTrainer`或者 :class:`donut.DonutPredictor`

    Args:
        hidden_net_p_x_z (Module or (tf.Tensor) -> tf.Tensor):
            :math:`p(x|z)`的隐藏网络
        hidden_net_q_z_x (Module or (tf.Tensor) -> tf.Tensor):
            :math:`q(z|x)`的隐藏网络
        x_dims (int):
            x维的数量
            必须为正整数
        z_dims (int):
            z维的数量
            必须为正整数
        std_epsilon (float):
            x和z的标准差最小值。
        name (str):
            可选模块名
            (:class:`tfsnippet.utils.VarScopeObject`参数)
        scope (str):
            此模块的可选范围
            (:class:`tfsnippet.utils.VarScopeObject`参数)
    """

    def __init__(self, hidden_net_p_x_z, hidden_net_q_z_x, x_dims, z_dims, std_epsilon=1e-4,
                 name=None, scope=None):
        if not is_integer(x_dims) or x_dims <= 0:
            raise ValueError('`x_dims`需要为正整数')
        if not is_integer(z_dims) or z_dims <= 0:
            raise ValueError('`z_dims`需要为正整数')

        super(Donut, self).__init__(name=name, scope=scope)
        with reopen_variable_scope(self.variable_scope):
            self._vae = VAE(
                p_z=Normal(mean=tf.zeros([z_dims]), std=tf.ones([z_dims])),
                p_x_given_z=Normal,
                q_z_given_x=Normal,
                h_for_p_x=Lambda(
                    partial(
                        wrap_params_net,
                        h_for_dist=hidden_net_p_x_z,
                        mean_layer=partial(
                            tf.layers.dense, units=x_dims, name='x_mean'
                        ),
                        std_layer=partial(
                            softplus_std, units=x_dims, epsilon=std_epsilon,
                            name='x_std'
                        )
                    ),
                    name='p_x_given_z'
                ),
                h_for_q_z=Lambda(
                    partial(
                        wrap_params_net,
                        h_for_dist=hidden_net_q_z_x,
                        mean_layer=partial(
                            tf.layers.dense, units=z_dims, name='z_mean'
                        ),
                        std_layer=partial(
                            softplus_std, units=z_dims, epsilon=std_epsilon,
                            name='z_std'
                        )
                    ),
                    name='q_z_given_x'
                )
            )
        self._x_dims = x_dims
        self._z_dims = z_dims

    @property
    def x_dims(self):
        """Get the number of `x` dimensions."""
        return self._x_dims

    @property
    def z_dims(self):
        """Get the number of `z` dimensions."""
        return self._z_dims

    @property
    def vae(self):
        """
        获得 :class:`Donut`模型的VAE对象。

        Returns:
            该:class:`Donut`模型的VAE对象。
        """
        return self._vae

    def get_training_loss(self, x, y, n_z=None):
        """
        Get the training loss for `x` and `y`.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            y (tf.Tensor): 2-D `int32` :class:`tf.Tensor`, the windows of
                ``(label | missing)`` in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)

        Returns:
            tf.Tensor: 0-d tensor, the training loss, which can be optimized
                by gradient descent algorithms.
        """
        with tf.name_scope('Donut.training_loss'):
            chain = self.vae.chain(x, n_z=n_z)
            x_log_prob = chain.model['x'].log_prob(group_ndims=0)
            alpha = tf.cast(1 - y, dtype=tf.float32)
            beta = tf.reduce_mean(alpha, axis=-1)
            log_joint = (
                    tf.reduce_sum(alpha * x_log_prob, axis=-1) +
                    beta * chain.model['z'].log_prob()
            )
            vi = VariationalInference(
                log_joint=log_joint,
                latent_log_probs=chain.vi.latent_log_probs,
                axis=chain.vi.axis
            )
            loss = tf.reduce_mean(vi.training.sgvb())
            return loss

    # def get_training_objective(self, *args, **kwargs):  # pragma: no cover
    #     warnings.warn('`get_training_objective` is deprecated, use`get_training_loss` instead.', DeprecationWarning)
    #     return self.get_training_loss(*args, **kwargs)

    def get_score(self, x, y=None, n_z=None, mcmc_iteration=None,
                  last_point_only=True):
        """
        Get the reconstruction probability for `x` and `y`.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            y (tf.Tensor): 2-D `int32` :class:`tf.Tensor`, the windows of
                missing point indicators in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)
            mcmc_iteration (int or tf.Tensor): Iteration count for MCMC
                missing csv_data imputation. (default :obj:`None`, no iteration)
            last_point_only (bool): Whether to obtain the reconstruction
                probability of only the last point in each window?
                (default :obj:`True`)

        Returns:
            tf.Tensor: The reconstruction probability, with the shape
                ``(len(x) - self.x_dims + 1,)`` if `last_point_only` is
                :obj:`True`, or ``(len(x) - self.x_dims + 1, self.x_dims)``
                if `last_point_only` is :obj:`False`.  This is because the
                first ``self.x_dims - 1`` points are not the last point of
                any window.
        """
        with tf.name_scope('Donut.get_score'):
            # MCMC missing csv_data imputation
            if y is not None and mcmc_iteration:
                x_r = iterative_masked_reconstruct(
                    reconstruct=self.vae.reconstruct,
                    x=x,
                    mask=y,
                    iter_count=mcmc_iteration,
                    back_prop=False,
                )
            else:
                x_r = x

            # get the reconstruction probability
            q_net = self.vae.variational(x=x_r, n_z=n_z)  # notice: x=x_r
            p_net = self.vae.model(z=q_net['z'], x=x, n_z=n_z)  # notice: x=x
            r_prob = p_net['x'].log_prob(group_ndims=0)
            if n_z is not None:
                n_z = validate_n_samples(n_z, 'n_z')
                assert_shape_op = tf.assert_equal(
                    tf.shape(r_prob),
                    tf.stack([n_z, tf.shape(x)[0], self.x_dims]),
                    message='Unexpected shape of reconstruction prob'
                )
                with tf.control_dependencies([assert_shape_op]):
                    r_prob = tf.reduce_mean(r_prob, axis=0)
            if last_point_only:
                r_prob = r_prob[:, -1]
            return r_prob
