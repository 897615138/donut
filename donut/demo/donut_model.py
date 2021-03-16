import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential

__all__ = ["get_model"]


def get_model():
    return model, model_vs


# 我们在“model_vs”的范围内构建整个模型，
# 它准确地保存“模型”的所有变量，包括Keras图层创建的变量。
with tf.variable_scope('model') as model_vs:
    model = Donut(
        # 构建`p(x|z)`的隐藏网络
        hidden_net_p_x_z=Sequential([
            # units：该层的输出维度。
            # kernel_regularizer：施加在权重上的正则项。L2正则化 权重尽可能小 惩罚力度不大
            # activation：激活函数 ReLU
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        hidden_net_q_z_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )
