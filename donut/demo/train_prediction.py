import time

import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from donut import Donut, DonutTrainer, DonutPredictor, get_time
from donut.demo.out import print_text


def train_prediction(use_plt, train_values, train_labels, train_missing, test_values, test_missing, test_labels,
                     train_mean, train_std, valid_num):
    """
    训练与预测
    Args:
        test_labels: 测试数据异常标签
        valid_num: 测试数据数量
        use_plt: 使用plt输出
        train_values: 训练数据值
        train_labels: 训练数据异常标签
        train_missing: 训练数据缺失点
        test_values: 测试数据
        test_missing: 测试数据缺失点
        train_mean: 平均值
        train_std: 标准差

    Returns:
        refactor_probability:  重构概率
        epoch_list: 遍数列表
        lr_list: 学习率变化列表
        epoch_time: 遍数
        model_time: 构建Donut模型时间
        trainer_time: 构建训练器时间
        predictor_time: 构建预测期时间
        fit_time: 训练时间
        probability_time: 获得重构概率时间
    """
    # 1.构造模型
    start_time = time.time()
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            # 构建`p(x|z)`的隐藏网络
            hidden_net_p_x_z=Sequential([
                # units：该层的输出维度。
                # kernel_regularizer：施加在权重上的正则项。L2正则化 使权重尽可能小 惩罚力度不大
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
            # x维的数量
            x_dims=120,
            # z维的数量
            z_dims=5,
        )
        end_time = time.time()
        model_time = get_time(start_time, end_time)
        print_text(use_plt, "构建Donut模型【共用时{}】".format(model_time))
        # 2.构造训练器
        start_time = time.time()
        trainer = DonutTrainer(model=model, model_vs=model_vs)
        end_time = time.time()
        trainer_time = get_time(start_time, end_time)
        print_text(use_plt, "构造训练器【共用时{}】".format(trainer_time))
        # 3.构造预测器
        start_time = time.time()
        predictor = DonutPredictor(model)
        end_time = time.time()
        predictor_time = get_time(start_time, end_time)
        print_text(use_plt, "构造预测器【共用时{}】".format(predictor_time))
        with tf.Session().as_default():
            # 4.训练器训练模型
            start_time = time.time()
            epoch_list, lr_list, epoch_time = \
                trainer.fit(train_values, train_labels, train_missing, test_values, test_labels, test_missing,
                            train_mean, train_std, valid_num, )
            end_time = time.time()
            fit_time = get_time(start_time, end_time)
            print_text(use_plt, "训练器训练模型【共用时{}】".format(fit_time))
            # 5.预测器获取重构概率
            start_time = time.time()
            refactor_probability = predictor.get_score(test_values, test_missing)
            end_time = time.time()
            probability_time = get_time(start_time, end_time)
            print_text(use_plt, "预测器获取重构概率【共用时{}】".format(probability_time))
            return refactor_probability, epoch_list, lr_list, epoch_time, model_time, trainer_time, predictor_time, fit_time, probability_time
