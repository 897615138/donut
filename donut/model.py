from functools import partial

import tensorflow as tf
from tfsnippet.distributions import Normal
from tfsnippet.modules import VAE, Lambda
from tfsnippet.stochastic import validate_n_samples
from tfsnippet.utils import (VarScopeObject, reopen_variable_scope, is_integer)
from tfsnippet.variational import VariationalInference

from .reconstruction import iterative_masked_reconstruct

__all__ = ['Donut']


def softplus_std(inputs, units, epsilon, name):
    """
    计算softplus: `log(exp(features) + 1)`再加上偏移量epsilon。
    Args:
        inputs: 输入
        units: 维数
        epsilon: x和z的标准差最小值
        name: 模块名

    Returns:
        softplus: `log(exp(features) + 1)`再加上偏移量epsilon
    """
    return tf.nn.softplus(tf.layers.dense(inputs, units, name=name)) + epsilon


def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    """
    包装网络
    Args:
        inputs: 输入
        h_for_dist: (Module or (tf.Tensor) -> tf.Tensor):
            隐藏网络 `p(x|z)`或者`q(z|x)`
        mean_layer: 输出空间维度为维数量的层
        std_layer: 输出空间维度为维数量的层的softplus再加上偏移量的层
    Returns：
        平均值，标准差
    """
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
    `get_refactor_probability`方法 获取重构概率 :class:`tf.Tensor`.

    Note:
        :class:`Donut` 实例在`get_training_loss`方法或`get_refactor_probability`方法被调用的时候才会构建。
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
            raise ValueError('`x_dims`必须为正整数')
        if not is_integer(z_dims) or z_dims <= 0:
            raise ValueError('`z_dims`必须为正整数')

        super(Donut, self).__init__(name=name, scope=scope)
        with reopen_variable_scope(self.variable_scope):
            # 基于VAE构造
            self._vae = VAE(
                # p(z)：均值和标准差都为z维数量大小的全零数组的一元正态分布
                p_z=Normal(mean=tf.zeros([z_dims]), std=tf.ones([z_dims])),
                # p(x|h(z))：一元正态分布
                p_x_given_z=Normal,
                # q(z|h(x))：一元正态分布
                q_z_given_x=Normal,
                # p(x|h(z))的隐藏网络：mean、std，由p(x|z)隐藏网络输入获得
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
                # q(z|h(x))的隐藏网络：mean、std，由q(z|x)隐藏网络输入获得
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
        """获取“x”维数"""
        return self._x_dims

    @property
    def z_dims(self):
        """获取“z”维数"""
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
        得到x和y的训练损失。

        Args:
            x (tf.Tensor):
                二维32位浮点 :class:`tf.Tensor`，小切片KPI观察窗口。
            y (tf.Tensor):
                二维32位整型 :class:`tf.Tensor`,在一个小切片中有“(标签|缺失点)”的窗口。
            n_z (int or None):
                每个x需要取的z样本数量。
                (default :obj:`None`, 没有显式抽样维数的样本)

        Returns:
            tf.Tensor:
                0维tensor, 训练损失（可以通过梯度下降算法进行优化）。
        """
        with tf.name_scope('Donut.training_loss'):
            # 将变分网q(z|h(x))和模型网p(x, z|h(x))与特定的观测x连接在一起，通常用于推导VAE的训练目标。也可以用来计算x的重构概率。
            chain = self.vae.chain(x, n_z=n_z)
            # 计算x模型网张量的对数密度。
            x_log_prob = chain.model['x'].log_prob(group_ndims=0)
            z_log_prob = chain.model['z'].log_prob()
            alpha = tf.cast(1 - y, dtype=tf.float32)
            # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的平均值，主要用作降维或者计算tensor（图像）的平均值。
            beta = tf.reduce_mean(alpha, axis=-1)
            log_joint = (
                    tf.reduce_sum(alpha * x_log_prob, axis=-1) + beta * z_log_prob
            )
            vi = VariationalInference(
                # 对数节点。
                log_joint=log_joint,
                # 得到隐变量的对数密度。
                latent_log_probs=chain.vi.latent_log_probs,
                # 坐标轴或坐标轴被认为是隐变量的采样维度。指定的轴将在变化的下界或训练目标中求和。
                axis=chain.vi.axis
            )
            loss = tf.reduce_mean(vi.training.sgvb())
            return loss

    def get_refactor_probability(self, window, missing=None, n_z=None, mcmc_iteration=None,
                                 last_point_only=True):
        """
        获得x，y的重构概率
        “重建概率”越大，异常点的可能性就越小。如果想要直接表明异常的严重程度，可以取这个分数的负值。

        Args:
            window (tf.Tensor): 二维32位浮点数:class:`tf.Tensor`, KPI观测的小切片窗口。
            missing (tf.Tensor): 二维32位整型 :class:`tf.Tensor`, 每个小切片中是否有有缺失点的窗口。
            n_z (int or None): 每个“x”要取的“z”样本数。(default :obj:`None`, 一个没有明确抽样维度的样本)
            mcmc_iteration (int or tf.Tensor): 缺失点注入的迭代次数(default :obj:`None`, 不迭代)
            last_point_only (bool): 是否获得窗口最后一个点的重构概率(default :obj:`True`)

        Returns:
            tf.Tensor: 重构概率,
                如果`last_point_only` 是:obj:`True`,shape为 ``(len(x) - self.x_dims + 1,)``
                反之，则为``(len(x) - self.x_dims + 1, self.x_dims)``
                这是因为第一个``self.x_dims - 1`` 点不是任何窗口的最后一个点。
        """
        with tf.name_scope('Donut.get_refactor_probability'):
            # MCMC缺失点注入
            # 如果没有缺失且需要迭代重构
            if missing is not None and mcmc_iteration:
                x_r = iterative_masked_reconstruct(
                    reconstruct=self.vae.reconstruct,
                    x=window,
                    mask=missing,
                    iter_count=mcmc_iteration,
                    back_prop=False,
                )
            # 使用原数据
            else:
                x_r = window

            # 获得重构概率
            # 派生一个:math:`q(z|h(x))`实例，变分网。如果未观察到z，则对每个x取z的样本数。
            q_net = self.vae.variational(x=x_r, n_z=n_z)  # notice: x=x_r
            # 派生一个:math:`p(x|h(z))`实例，模型网。
            p_net = self.vae.model(z=q_net['z'], x=window, n_z=n_z)  # notice: x=x
            # 计算:class:`StochasticTensor`的对数密度。覆盖已配置的'group_ndimm'为0。
            r_prob = p_net['x'].log_prob(group_ndims=0)
            # 如果未观察到z，则对每个x取z的样本数。样本数不为None
            if n_z is not None:
                # 验证' n_samples '参数。用装饰器，定义支持with语句上下文管理器的工厂函数
                n_z = validate_n_samples(n_z, 'n_z')
                assert_shape_op = tf.assert_equal(
                    tf.shape(r_prob),
                    tf.stack([n_z, tf.shape(window)[0], self.x_dims]),
                    message='Unexpected shape of reconstruction prob'
                )
                # 控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在[assert_shape_op]执行。'
                # graph. control_dependencies() '的包装器，使用默认的图形。
                with tf.control_dependencies([assert_shape_op]):
                    # 计算张量的维数中元素的均值。
                    # 沿着给定的维数0减少r_prob。在0中的每一项张量的秩都会减少1。
                    r_prob = tf.reduce_mean(r_prob, axis=0)
            # 获得窗口最后一个点的重构概率
            if last_point_only:
                r_prob = r_prob[:, -1]
            return r_prob
