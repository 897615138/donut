import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from donut.model import Donut
from donut.out import print_info, print_text, show_line_chart
from donut.prediction import DonutPredictor
from donut.time_util import TimeCounter
from donut.training import DonutTrainer


def train_prediction_v1(use_plt, train_values, train_labels, train_missing, test_values, test_missing, test_labels,
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
    tc = TimeCounter()
    tc.start()
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
        tc.end()
        model_time = tc.get_s() + "秒"
        print_info(use_plt, "5.构建Donut模型【共用时{}】".format(model_time))
        # 2.构造训练器
        tc.start()
        trainer = DonutTrainer(model=model, model_vs=model_vs)
        tc.end()
        trainer_time = tc.get_s() + "秒"
        print_info(use_plt, "6.构造训练器【共用时{}】".format(trainer_time))
        # 3.构造预测器
        tc.start()
        predictor = DonutPredictor(model)
        tc.end()
        predictor_time = tc.get_s() + "秒"
        print_info(use_plt, "7.构造预测器【共用时{}】".format(predictor_time))
        with tf.Session().as_default():
            # 4.训练器训练模型
            tc.start()
            epoch_list, lr_list, epoch_time, train_message = \
                trainer.fit(use_plt, train_values, train_labels, train_missing, test_values, test_labels, test_missing,
                            train_mean, train_std, valid_num)
            tc.end()
            fit_time = tc.get_s() + "秒"
            print_info(use_plt, "8.训练器训练模型【共用时{}】".format(fit_time))
            print_text(use_plt, "所有epoch【共用时：{}】".format(epoch_time))
            print_text(use_plt, "退火学习率 学习率随epoch变化")
            show_line_chart(use_plt, epoch_list, lr_list, 'annealing learning rate')
            # 5.预测器获取重构概率
            tc.start()
            refactor_probability = predictor.get_refactor_probability(test_values, test_missing)
            tc.end()
            probability_time = tc.get_s() + "秒"
            print_info(use_plt, "9.预测器获取重构概率【共用时{}】".format(probability_time))
            return refactor_probability, epoch_list, lr_list, epoch_time, model_time, trainer_time, predictor_time, fit_time, probability_time, train_message


def train_prediction_v2(use_plt, src_threshold_value,
                        std_train_values, fill_train_labels, train_missing,
                        std_test_values, fill_test_labels, test_missing,
                        mean, std, test_data_num):
    """
    训练与预测
    Args:
        std: 标准差
        mean: 平均值
        fill_test_labels: 测试数据异常标签
        test_data_num: 测试数据数量
        use_plt: 使用plt输出
        std_train_values: 训练数据值
        fill_train_labels: 训练数据异常标签
        train_missing: 训练数据缺失点
        std_test_values: 测试数据
        test_missing: 测试数据缺失点
        src_threshold_value: 初始阈值

    Returns:
        test_refactor_probability:  重构概率
        epoch_list: 遍数列表
        lr_list: 学习率变化列表
        epoch_time: 遍数
        model_time: 构建Donut模型时间
        trainer_time: 构建训练器时间
        predictor_time: 构建预测期时间
        fit_train_time: 训练时间
        test_probability_time: 获得重构概率时间
    """
    # 1.构造模型
    tc = TimeCounter()
    tc.start()
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
        tc.end()
        model_time = tc.get_s() + "秒"
        print_info(use_plt, "4.构建Donut模型【共用时{}】".format(model_time))
        # 2.构造训练器
        tc.start()
        trainer = DonutTrainer(model=model, model_vs=model_vs)
        tc.end()
        trainer_time = tc.get_s() + "秒"
        print_info(use_plt, "5.构造训练器【共用时{}】".format(trainer_time))
        # 3.构造预测器
        tc.start()
        predictor = DonutPredictor(model)
        tc.end()
        predictor_time = tc.get_s() + "秒"
        print_info(use_plt, "6.构造预测器【共用时{}】".format(predictor_time))
        with tf.Session().as_default():
            # 4.训练器训练模型
            tc.start()
            epoch_list, lr_list, epoch_time, train_message = \
                trainer.fit(use_plt,
                            std_train_values, fill_train_labels, train_missing,
                            std_test_values, fill_test_labels, test_missing,
                            mean, std, test_data_num)
            tc.end()
            fit_time = tc.get_s() + "秒"
            print_info(use_plt, "7.训练器训练模型【共用时{}】".format(fit_time))
            print_text(use_plt, "所有epoch【共用时：{}】".format(epoch_time))
            print_text(use_plt, "退火学习率 学习率随epoch变化")
            show_line_chart(use_plt, epoch_list, lr_list, 'annealing learning rate')
            # 5.预测器获取重构概率
            # 有默认阈值
            if src_threshold_value is not None:
                test_refactor_probability, test_probability_time \
                    = predictor.get_refactor_probability(std_test_values, test_missing)
                test_refactor_probability = None
                test_probability_time = None
            else:
                test_refactor_probability, test_probability_time \
                    = predictor.get_refactor_probability(std_test_values, test_missing)
                train_refactor_probability, train_probability_time \
                    = predictor.get_refactor_probability(std_train_values, train_missing)
            print_info(use_plt, "8.预测器获取重构概率【共用时{}】".format(test_probability_time))
            return epoch_list, lr_list, epoch_time, \
                   model_time, trainer_time, predictor_time, fit_time, train_message, \
                   train_refactor_probability, train_probability_time, \
                   test_refactor_probability, test_probability_time
