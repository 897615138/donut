import tensorflow as tf
from donut import Donut
from tensorflow import keras as k
from tfsnippet.modules import Sequential

# 我们在“model_vs”的范围内构建整个模型，
# 它应该准确地保存“模型”的所有变量，包括Keras图层创建的变量。
with tf.variable_scope('model') as model_vs:
    model = Donut(
        hidden_net_p_x_z=Sequential([
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        hidden_net_q_z_x=Sequential([
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )
