import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from donut import Donut, DonutTrainer, DonutPredictor


def train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std):
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
        trainer = DonutTrainer(model=model, model_vs=model_vs)
        predictor = DonutPredictor(model)
        with tf.Session().as_default():
            epoch_list, lr_list, epoch_time = trainer.fit(values=train_values, labels=train_labels,
                                                          missing=train_missing, mean=mean, std=std)
            test_score = predictor.get_score(test_values, test_missing)
            return test_score, epoch_list, lr_list, epoch_time
