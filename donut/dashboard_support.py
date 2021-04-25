import os
import shelve
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.modules import Sequential

import donut.util.out.show_plt as plt
import donut.util.out.show_sl as sl
from donut.assessment import Assessment
from donut.data import gain_sl_cache_data, gain_data
from donut.model import Donut
from donut.prediction import DonutPredictor
from donut.preprocessing import standardize_kpi
from donut.training import DonutTrainer
from donut.util.time_util import TimeCounter, get_constant_timestamp, TimeUse, format_time


class Dashboard(object):

    def __init__(self, use_plt=True, train_file="4096_14.21.csv", test_file="4096_1.88.csv", is_local=True,
                 is_upload=False, src_threshold_value=None, a=1, use_cache_probability=True, use_cache_result=True):
        self._use_plt = use_plt
        self._train_file = train_file
        self._test_file = test_file
        self._is_local = is_local
        self._is_upload = is_upload
        self._src_threshold_value = src_threshold_value
        self._has_error = False
        self._error_str = ""
        self._a = a
        self._tc_1 = TimeCounter()
        self._tc_2 = TimeCounter()
        self._use_cache_probability = use_cache_probability
        if use_cache_result:
            exist, suggest = self.check_file("result")
            if exist:
                self.print_text(suggest)
                try:
                    self.read_all()
                    self.show_cache()
                except:
                    self.use_cache_pro()
            else:
                self.print_text(suggest)
                self.use_cache_pro()
        else:
            self.use_cache_pro()

    def use_cache_pro(self):
        if self._use_cache_probability:
            try:
                self.use_cache_p()
            except:
                self.print_warn("缓存文件损坏")
                self.do_all()
        else:
            exist, suggest = self.check_file("probability")
            self.print_text(suggest)
            self.do_all()

    def do_all(self):
        self.before_save_probability()
        # 5.存储重构概率等数据
        self.save_probability()
        self.after_save_probability()

    def save_probability(self):
        self._tc_1.start()
        name = self.file_name_converter("probability")
        db = shelve.open(name)
        db["_test_scores"] = self._test_scores
        db["_real_test_labels"] = self._real_test_labels
        db["_get_file_time"] = self._get_file_time
        db["_real_test_missing"] = self._real_test_missing
        db["_has_error"] = self._has_error
        db["_error_str"] = self._error_str
        db["_src_train_timestamps"]=self._src_train_timestamps
        db["_get_train_file_time"] = self._get_train_file_time
        db["_src_train_num"] = self._src_train_num
        db["_src_train_label_num"] = self._src_train_label_num
        db["_src_train_label_proportion"] = self._src_train_label_proportion
        db["_src_train_values"] = self._src_train_values
        db["_get_test_file_time"] = self._get_test_file_time
        db["_src_test_num"] = self._src_test_num
        db["_src_test_label_num"] = self._src_test_label_num
        db["_src_test_label_proportion"] = self._src_test_label_proportion
        db["_src_test_timestamps"] = self._src_test_timestamps
        db["_src_test_values"] = self._src_test_values
        db["_handel_data_time"] = self._handel_data_time
        db["_check_m_s_time"] = self._check_m_s_time
        db["_miss_insert_re_time"] = self._miss_insert_re_time
        db["_fill_step"] = self._fill_step
        db["_mean"] = self._mean
        db["_std"] = self._std
        db["_train_data_num"] = self._train_data_num
        db["_fill_train_label_num"] = self._fill_train_label_num
        db["_fill_train_label_proportion"] = self._fill_train_label_proportion
        db["_fill_train_num"] = self._fill_train_num
        db["_train_missing_interval_num"] = self._train_missing_interval_num
        db["_train_missing_str"] = self._train_missing_str
        db["_fill_train_timestamps"] = self._fill_train_timestamps
        db["_fill_train_values"] = self._fill_train_values
        db["_test_data_num"] = self._test_data_num
        db["_fill_test_label_num"] = self._fill_test_label_num
        db["_fill_test_label_proportion"] = self._fill_test_label_proportion
        db["_fill_test_num"] = self._fill_test_num
        db["_test_missing_interval_num"] = self._test_missing_interval_num
        db["_test_missing_str"] = self._test_missing_str
        db["_fill_test_timestamps"] = self._fill_test_timestamps
        db["_fill_test_values"] = self._fill_test_values
        db["_std_time"] = self._std_time
        db["_fill_train_timestamps"] = self._fill_train_timestamps
        db["_std_train_values"] = self._std_train_values
        db["_fill_test_timestamps"] = self._fill_test_timestamps
        db["_std_test_values"] = self._std_test_values
        db["_model_time"] = self._model_time
        db["_trainer_time"] = self._trainer_time
        db["_predictor_time"] = self._predictor_time
        db["_fit_time"] = self._fit_time
        db["_epoch_time"] = self._epoch_time
        db["_epoch_list"] = self._epoch_list
        db["_lr_list"] = self._lr_list
        db["_test_probability_time"] = self._test_probability_time
        db["_train_predict_compute_time"] = self._train_predict_compute_time
        db["_handle_refactor_probability_time"] = self._handle_refactor_probability_time
        db["_real_test_data_num"] = self._real_test_data_num
        db["_real_test_label_num"] = self._real_test_label_num
        db["_real_test_missing_num"] = self._real_test_missing_num
        db["_real_test_label_proportion"] = self._real_test_label_proportion
        db["_real_test_timestamps"] = self._real_test_timestamps
        db["_real_test_values"] = self._real_test_values
        db["_test_scores"] = self._test_scores
        db["_real_test_labels"] = self._real_test_labels
        db["_get_file_time"] = self._get_file_time
        self._save_probability_time = self._tc_1.get_s() + '秒'
        db["_save_probability_time"] = self._save_probability_time
        db.close()
        self.print_info("5.存储重构概率等数据【共用时{}】".format(self._save_probability_time))

    def use_cache_p(self):
        exist, suggest = self.check_file("probability")
        if exist:
            self.print_text(suggest)
            self.read_probability()
            self.show_cache_probability()
            self.after_save_probability()
        else:
            self.print_text(suggest)
            self.do_all()

    def after_save_probability(self):
        # 6.评估
        self.get_assessment()
        # 7.存储所有信息
        self.save_all()
        # 8.时间排序
        self.time_sort()

    def before_save_probability(self):
        # 1.获取数据
        self.get_data()
        # 2.处理数据
        self.handle_data()
        # 3.训练与预测，获得重构概率
        self.train_predict_compute()
        # 4.处理重构概率
        self.handle_probability()

    def show_test_score(self):
        """
        展示测试数据
        """
        if self._use_plt:
            plt.show_test_score(self._real_test_timestamps, self._real_test_values, self._test_scores)
        else:
            sl.show_test_score(self._real_test_timestamps, self._real_test_values, self._test_scores)

    def handle_probability(self):
        """
        处理测试数据分数结果
        1.得出默认为正常数据的点数
        2.截取实际测试数据 相关
        3.重构概率负数，获得分数
        """
        # 因为对于每个窗口的检测实际返回的是最后一个窗口的重建概率，
        # 也就是说第一个窗口的前面一部分的点都没有检测，默认为正常数据。
        # 因此需要在检测结果前面补零或者测试数据的真实 label。
        self._tc_1.start()
        self._zero_num = self._test_data_num - self._test_refactor_probability.size
        self._real_test_timestamps = self._fill_test_timestamps[self._zero_num:np.size(self._fill_test_timestamps)]
        self._real_test_values = self._std_test_values[self._zero_num:np.size(self._std_test_values)]
        self._real_test_labels = self._fill_test_labels[self._zero_num:np.size(self._fill_test_labels)]
        self._real_test_missing = self._test_missing[self._zero_num:np.size(self._test_missing)]
        self._test_scores = 0 - self._test_refactor_probability
        self._real_test_data_num = np.size(self._real_test_timestamps)
        self._real_test_label_num = np.sum(self._real_test_labels == 1)
        self._real_test_missing_num = np.sum(self._real_test_missing == 1)
        self._real_test_label_proportion = self._real_test_label_num / self._real_test_data_num
        self._handle_refactor_probability_time = self._tc_1.get_s() + '秒'
        self.print_info("4.处理重构概率，获得真实测试数据集【共用时{}】".format(self._handle_refactor_probability_time))
        self.print_text("实际测试数据集")
        self.show_test_score()
        self.print_text("共{}条数据,有{}个标注，有{}个缺失数据，标签比例约为{:.2%}"
                        .format(self._real_test_data_num, self._real_test_label_num, self._real_test_missing_num,
                                self._real_test_label_proportion))

    def train_predict_compute(self):
        """
        训练与预测
        """
        self.print_info("训练与预测，获得重构概率")
        self._tc_2.start()
        # 1.构造模型
        self._tc_1.start()
        with tf.variable_scope('model') as model_vs:
            # 1.构造模型
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
            self._model = model
            self._model_time = self._tc_1.get_s() + "秒"
            self.print_text("构建Donut模型【共用时{}】".format(self._model_time))
            # 2.构造训练器
            self._tc_1.start()
            trainer = DonutTrainer(model=model, model_vs=model_vs)
            self._trainer_time = self._tc_1.get_s() + "秒"
            self.print_text("构造训练器【共用时{}】".format(self._trainer_time))
            # 3.构造预测器
            self._tc_1.start()
            predictor = DonutPredictor(model)
            self._predictor = predictor
            self._predictor_time = self._tc_1.get_s() + "秒"
            self.print_text("构造预测器【共用时{}】".format(self._predictor_time))
            with tf.Session().as_default():
                # 4.训练器训练模型
                self._tc_1.start()
                self._epoch_list, self._lr_list, self._epoch_time, self._train_message = \
                    trainer.fit(self._use_plt,
                                self._std_train_values, self._fill_train_labels, self._train_missing,
                                self._std_test_values, self._fill_test_labels, self._test_missing,
                                self._mean, self._std, self._test_data_num)
                self._fit_time = self._tc_1.get_s() + "秒"
                self.print_text("训练器训练模型【共用时{}】".format(self._fit_time))
                self.print_text("所有epoch【共用时：{}】".format(self._epoch_time))
                self.print_text("退火学习率 学习率随epoch变化")
                self.show_line_chart(self._epoch_list, self._lr_list, 'annealing learning rate')
                # 5.预测器获取重构概率
                self._test_refactor_probability, self._test_probability_time \
                    = predictor.get_refactor_probability(self._std_test_values, self._test_missing)
                self.print_text("预测器获取重构概率【共用时{}】".format(self._test_probability_time))
                self._train_predict_compute_time = self._tc_2.get_s() + "秒"
                self.print_text("【共用时{}】".format(self._train_predict_compute_time))

    def print_warn(self, content):
        if not self._use_plt:
            sl.warning(content)
        else:
            print(content)

    def print_info(self, content):
        """
           展示文字
           Args:
               content: 文字内容
           """
        if self._use_plt:
            print(content)
        else:
            sl.info(content)

    def print_text(self, content):
        """
        展示文字
        Args:
            content: 文字内容
        """
        if self._use_plt:
            print(content)
        else:
            sl.text(content)

    def show_line_chart(self, x, y, name):
        """
        展示折线图
        Args:
            x: x轴
            y: y轴
            name: 名称
        """
        if self._use_plt:
            plt.line_chart(x, y, name)
        else:
            sl.line_chart(x, y, name)

    def get_info_from_file(self, file, suffix="sample_data"):
        """
        读取数据文件
        Args:
            file: 文件名
            suffix: 文件包名前缀

        Returns:
            数据与后续可处理文件名
        """
        try:
            if self._is_upload:
                src_timestamps, src_labels, src_values = gain_sl_cache_data(file)
            elif self._is_local:
                src_timestamps, src_labels, src_values = gain_data("../" + suffix + "/" + file)
                a = file.split("/")
                file = a[len(a) - 1]
            else:
                src_timestamps, src_labels, src_values = gain_data(suffix + "/" + file)
            return src_timestamps, src_labels, src_values, file, True
        except Exception:
            return None, None, None, None, False

    def get_data(self):
        self._tc_2.start()
        self._tc_1.start()
        self._src_train_timestamps, self._src_train_labels, self._src_train_values, self._train_file, success \
            = self.get_info_from_file(self._train_file)
        if not success:
            self.print_warn("找不到数据文件，请检查文件名与路径")
            sys.exit()
        self._get_train_file_time = self._tc_1.get_s() + "秒"
        self._tc_1.start()
        self._src_test_timestamps, self._src_test_labels, self._src_test_values, self._test_file, success \
            = self.get_info_from_file(self._test_file)
        if not success:
            self.print_warn("找不到数据文件，请检查文件名与路径")
            sys.exit()
        self._get_test_file_time = self._tc_1.get_s() + "秒"
        # 原训练数据数量
        self._src_train_num = self._src_train_timestamps.size
        # 原训练数据标签数
        self._src_train_label_num = np.sum(self._src_train_labels == 1)
        # 原训练数据标签占比
        self._src_train_label_proportion = self._src_train_label_num / self._src_train_num
        # 原测试数据数量
        self._src_test_num = self._src_test_timestamps.size
        # 原测试数据标签数
        self._src_test_label_num = np.sum(self._src_test_labels == 1)
        # 原测试数据标签占比
        self._src_test_label_proportion = self._src_test_label_num / self._src_test_num

        self._get_file_time = self._tc_2.get_s() + "秒"
        self.print_info("1.获取数据【共用时{}】".format(self._get_file_time))
        self.print_text("获取训练数据【共用时{}】".format(self._get_train_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_train_num, self._src_train_label_num, self._src_train_label_proportion))
        self.show_line_chart(self._src_train_timestamps, self._src_train_values, 'original csv train data')
        self.print_text("获取测试数据【共用时{}】".format(self._get_test_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_test_num, self._src_test_label_num, self._src_test_label_proportion))
        self.show_line_chart(self._src_test_timestamps, self._src_test_values, 'original csv test data')

    def check_timestamp(self):
        # np src_array-> src_array
        self._train_timestamp = np.asarray(self._src_train_timestamps, np.int64)
        self._test_timestamp = np.asarray(self._src_test_timestamps, np.int64)
        # 一维数组检验
        if len(self._train_timestamp.shape) != 1 or len(self._test_timestamp.shape) != 1:
            self.print_warn('`self._train_timestamp`必须为一维数组')
            return
        train_arrays = (self._src_train_values, self._src_train_labels)
        # np arrays-> arrays
        src_train_arrays = [np.asarray(src_array) for src_array in train_arrays]
        # 相同维度
        for i, src_array in enumerate(src_train_arrays):
            if src_array.shape != self._train_timestamp.shape:
                self.print_warn('`self._train_timestamp` 的形状必须与`src_array`的形状相同 ({} vs {}) src_array index {}'
                                .format(self._train_timestamp.shape, src_array.shape, i))
                return
        # np arrays-> arrays
        test_arrays = (self._src_test_values, self._src_test_labels)
        src_test_arrays = [np.asarray(src_array) for src_array in test_arrays]
        # 相同维度
        for i, src_array in enumerate(src_test_arrays):
            if src_array.shape != self._test_timestamp.shape:
                self.print_warn('`self._test_timestamp` 的形状必须与`src_array`的形状相同 ({} vs {}) src_array index {}'
                                .format(self._test_timestamp.shape, src_array.shape, i))

    def handle_data(self):
        self._tc_2.start()
        self._tc_1.start()
        # 1.检验数据并计算综合平均值与标准差
        self.check_timestamp()
        # 2.检验时间戳数据 补充为有序等间隔时间戳数组
        # 时间戳排序 获得对数组排序后的原数组的对应索引以及有序数组
        self._src_train_index = np.argsort(self._train_timestamp)
        self._src_test_index = np.argsort(self._test_timestamp)
        self._train_timestamp_sorted = self._train_timestamp[self._src_train_index]
        self._test_timestamp_sorted = self._test_timestamp[self._src_test_index]
        self._train_values_sorted = self._src_train_values[self._src_train_index]
        self._test_values_sorted = self._src_test_values[self._src_test_index]
        self._train_labels_sorted = self._src_train_labels[self._src_train_index]
        self._test_labels_sorted = self._src_test_labels[self._src_test_index]
        # 沿给定轴计算离散差分 即获得所有存在的时间戳间隔
        train_intervals = np.unique(np.diff(self._train_timestamp_sorted))
        test_intervals = np.unique(np.diff(self._test_timestamp_sorted))
        # 最小的间隔数
        self._train_min_interval = np.min(train_intervals)
        self._test_min_interval = np.min(test_intervals)
        self._error_str = ""
        # 单独有重复
        if self._train_min_interval <= 0:
            self._has_error = True
            self._repeat_timestamp = set(self._train_timestamp_sorted - np.unique(self._train_timestamp_sorted))
            for t in self._repeat_timestamp:
                self._error_str = self._error_str + " " + str(t)
            self._error_str = '训练数据重复时间戳:' + self._error_str + "\n"
            self.print_warn(self._error_str)
            sys.exit()
        if self._test_min_interval <= 0:
            self._has_error = True
            self._repeat_timestamp = set(self._test_timestamp_sorted - np.unique(self._test_timestamp_sorted))
            self._error_str = ""
            for t in self._repeat_timestamp:
                self._error_str = self._error_str + " " + str(t)
            self._error_str = '测试数据重复时间戳:' + self._error_str + "\n"
            self.print_warn(self._error_str)
            sys.exit()
        # 合并有重复时检查
        merge_set = set(self._train_timestamp_sorted).intersection(set(self._test_timestamp_sorted))
        if len(merge_set) != 0:
            for i in merge_set:
                # 寻找指定数值的索引
                train_index = np.where(self._train_timestamp_sorted == i)
                test_index = np.where(self._test_timestamp_sorted == i)
                if self._train_values_sorted[train_index] != self._test_values_sorted[test_index]:
                    self._has_error = True
                    self._error_str = "训练与测试数据中相同时间戳有不同KPI值，时间戳：{}，训练数据：{}，测试数据：{}" \
                        .format(i, self._train_values_sorted[train_index], self._test_values_sorted[test_index])
                    self.print_warn(self._error_str)
                    sys.exit()
                if self._train_labels_sorted[train_index] != self._test_labels_sorted[test_index]:
                    self._error_str = "训练与测试数据中相同时间戳有不同标签，时间戳：{}，训练数据：{}，测试数据：{}" \
                        .format(i, self._train_labels_sorted[train_index], self._test_labels_sorted[test_index])
                    self.print_warn(self._error_str)
                    self._has_error = True
                    sys.exit()
            #  取并集
            union_timestamps = list(set(self._train_timestamp_sorted).union(set(self._test_timestamp_sorted)))
            union_amount = len(union_timestamps)
            union_values = np.zeros([union_amount], dtype=self._train_values_sorted.dtype)
            union_labels = np.zeros([union_amount], dtype=self._train_labels_sorted.dtype)
            for i, t in enumerate(union_timestamps):
                # 寻找指定数值的索引
                train_index = np.where(self._train_timestamp_sorted == t)
                test_index = np.where(self._test_timestamp_sorted == t)
                if np.size(train_index) != 0:
                    union_values[i] = self._train_values_sorted[train_index][0]
                    union_labels[i] = self._train_labels_sorted[train_index][0]
                elif np.size(test_index) != 0:
                    union_values[i] = self._test_values_sorted[test_index][0]
                    union_labels[i] = self._test_labels_sorted[test_index][0]
                else:
                    self._has_error = True
                    self._error_str = "找不到指定时间戳的索引:{}".format(str(t))
                    self.print_warn(self._error_str)
            self._mean = np.mean(union_values)
            self._std = np.std(union_values, ddof=1)
        else:
            self._n = len(self._train_values_sorted)
            self._m = len(self._test_values_sorted)
            self._m1 = np.mean(self._train_values_sorted)
            self._m2 = np.mean(self._test_values_sorted)
            self._s1 = np.std(self._train_values_sorted)
            self._s2 = np.std(self._train_values_sorted)
            self._mean = (self._n * self._m1 + self._m * self._m2) / (self._m + self._n)
            self._std = np.sqrt((self._n * self._s1 * self._s1 + self._m * self._s2 * self._s2 +
                                 (self._m * self._n * (self._m1 - self._m2) * (self._m1 - self._m2)) /
                                 (self._m + self._n))
                                / (self._m + self._n))
        if self._train_min_interval != self._test_min_interval:
            self._has_error = True
            self._error_str = '最小间隔数不同训练数据最小间隔：{},测试数据最小间隔：{}' \
                .format(self._train_min_interval, self._test_min_interval)
            self.print_warn(self._error_str)
            sys.exit()
        for i in train_intervals:
            if i % self._train_min_interval != 0 or i % self._test_min_interval != 0:
                self._has_error = True
                self._error_str = '并不是所有时间间隔都是最小时间间隔的倍数,最小间隔：{},异常间隔：{}' \
                    .format(self._train_min_interval, i)
                self.print_warn(self._error_str)
                sys.exit()
        for i in test_intervals:
            if i % self._train_min_interval != 0 or i % self._test_min_interval != 0:
                self._has_error = True
                self._error_str = '并不是所有时间间隔都是最小时间间隔的倍数,最小间隔：{},异常间隔：{}' \
                    .format(self._test_min_interval, i)
                self.print_warn(self._error_str)
                sys.exit()

        self._check_m_s_time = self._tc_1.get_s() + "秒"

        # 2.定位缺失点并填充重构数据集
        self._tc_1.start()
        self._train_amount = (self._train_timestamp_sorted[-1] - self._train_timestamp_sorted[
            0]) // self._train_min_interval + 1
        self._test_amount = (self._test_timestamp_sorted[-1] - self._test_timestamp_sorted[
            0]) // self._test_min_interval + 1
        # 初始化
        self._fill_train_values = np.zeros([self._train_amount], dtype=self._train_values_sorted.dtype)
        self._fill_test_values = np.zeros([self._test_amount], dtype=self._test_values_sorted.dtype)
        self._fill_train_labels = np.zeros([self._train_amount], dtype=self._train_labels_sorted.dtype)
        self._fill_test_labels = np.zeros([self._test_amount], dtype=self._test_labels_sorted.dtype)
        # 重构时间戳数组
        self._fill_train_timestamps = np.arange(self._train_timestamp_sorted[0],
                                                self._train_timestamp_sorted[-1] + self._train_min_interval,
                                                self._train_min_interval, dtype=np.int64)
        self._fill_test_timestamps = np.arange(self._test_timestamp_sorted[0],
                                               self._test_timestamp_sorted[-1] + self._test_min_interval,
                                               self._test_min_interval, dtype=np.int64)
        # 初始化缺失点数组与数值与标注数组
        self._train_missing = np.ones([self._train_amount], dtype=np.int32)
        self._test_missing = np.ones([self._test_amount], dtype=np.int32)
        # 获得与初始时间戳的差值数组
        train_diff_with_first = (self._train_timestamp_sorted - self._train_timestamp_sorted[0])
        test_diff_with_first = (self._test_timestamp_sorted - self._test_timestamp_sorted[0])
        # 获得与初始时间戳相差的最小间隔数 即应处的索引值
        diff_train_intervals_with_first = train_diff_with_first // self._train_min_interval
        dst_train_index = np.asarray(diff_train_intervals_with_first, dtype=np.int)
        diff_test_intervals_with_first = test_diff_with_first // self._test_min_interval
        dst_test_index = np.asarray(diff_test_intervals_with_first, dtype=np.int)
        # 标记有原值的时间戳为非缺失点
        self._train_missing[dst_train_index] = 0
        self._test_missing[dst_test_index] = 0
        # 分别组合
        self._fill_train_values[dst_train_index] = self._src_train_values[self._src_train_index]
        self._fill_test_values[dst_test_index] = self._src_test_values[self._src_test_index]
        self._fill_train_labels[dst_train_index] = self._src_train_labels[self._src_train_index]
        self._fill_test_labels[dst_test_index] = self._src_test_labels[self._src_test_index]
        self._train_data_num = self._fill_train_timestamps.size
        self._test_data_num = self._fill_test_timestamps.size
        self._fill_train_num = self._train_data_num - self._src_train_num
        self._fill_test_num = self._test_data_num - self._src_test_num
        self._fill_train_label_num = np.sum(self._fill_train_labels == 1)
        self._fill_test_label_num = np.sum(self._fill_test_labels == 1)
        self._fill_train_label_proportion = self._fill_train_label_num / self._train_data_num
        self._fill_test_label_proportion = self._fill_test_label_num / self._test_data_num
        self._fill_step = self._fill_train_timestamps[1] - self._fill_train_timestamps[0]
        self._train_missing_index = np.where(self._train_missing == 1)
        self._train_missing_timestamps = self._fill_train_timestamps[self._train_missing_index]
        self._train_missing_interval_num, self._train_missing_str = get_constant_timestamp(
            self._train_missing_timestamps, self._fill_step)
        self._test_missing_index = np.where(self._test_missing == 1)
        self._test_missing_timestamps = self._fill_test_timestamps[self._test_missing_index]
        self._test_missing_interval_num, self._test_missing_str = get_constant_timestamp(
            self._test_missing_timestamps, self._fill_step)

        self._miss_insert_re_time = self._tc_1.get_s() + "秒"
        # 标准化数据
        self._tc_1.start()
        self.standardize_data()

        self._std_time = self._tc_1.get_s() + "秒"

        self._handel_data_time = self._tc_2.get_s() + "秒"
        self.print_info("2.数据处理【共用时{}】".format(self._handel_data_time))
        self.print_text("检验数据并计算综合平均值与标准差【共用时:{}】".format(self._check_m_s_time))
        self.print_text("定位缺失点并填充重构数据集【共用时:{}】".format(self._miss_insert_re_time))
        self.print_text("时间戳步长:{}".format(self._fill_step))
        self.print_text("平均值：{}，标准差：{}".format(self._mean, self._std))
        self.print_text("训练数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._train_data_num, self._fill_train_label_num, self._fill_train_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_train_num, self._train_missing_interval_num, self._train_missing_str))
        self.show_line_chart(self._fill_train_timestamps, self._fill_train_values, 'filled train data')
        self.print_text("测试数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._test_data_num, self._fill_test_label_num, self._fill_test_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_test_num, self._test_missing_interval_num, self._test_missing_str))
        self.show_line_chart(self._fill_test_timestamps, self._fill_test_values, 'filled test data')
        self.print_info("标准化训练数据【共用时{}】".format(self._std_time))
        # 显示标准化后的训练数据
        self.show_line_chart(self._fill_train_timestamps, self._std_train_values, 'standardized train data')
        self.show_line_chart(self._fill_test_timestamps, self._std_test_values, 'standardized test data')

    def standardize_data(self):
        """
        标准化数据
        """
        exclude_array = np.logical_or(self._fill_train_labels, self._train_missing)
        self._std_train_values, _, _ = standardize_kpi(self._fill_train_values, mean=self._mean, std=self._std,
                                                       excludes=np.asarray(exclude_array, dtype='bool'))
        self._std_test_values, _, _ = standardize_kpi(self._fill_test_values, mean=self._mean, std=self._std)

    def get_assessment(self):
        self._tc_1.start()
        # 根据分数捕获异常 获得阈值
        self._assessment = Assessment(self._src_threshold_value, self._test_scores, self._real_test_labels,
                                      self._real_test_missing, self._a, self._use_plt)
        self._threshold_value, self._f_score, self._catch_num, self._catch_index, self._fp_index, self._fp_num, \
        self._tp_index, self._tp_num, self._fn_index, self._fn_num, self._precision, self._recall, self._lis \
            = self._assessment.get_assessment()
        self._catch_timestamps = self._real_test_timestamps[self._catch_index]
        self._catch_interval_num, self._catch_interval_str = get_constant_timestamp(self._catch_timestamps,
                                                                                    self._fill_step)
        self._assessment_time = self._tc_1.get_s() + '秒'
        self._tp_interval_num, self._tp_interval_str = get_constant_timestamp(self._tp_index, self._fill_step)
        self._fp_interval_num, self._fp_interval_str = get_constant_timestamp(self._fp_index, self._fill_step)
        self._fn_interval_num, self._fn_interval_str = get_constant_timestamp(self._fn_index, self._fill_step)
        self._lis_str = ""
        index = 1
        if len(self._lis) > 10:
            for l in self._lis[::-1]:
                if index < 10:
                    self._lis_str = self._lis_str + '{} 阈值：{}，分数：{}\n'.format(index, l.get("threshold"), l.get("f"))
                    index = index + 1
                else:
                    break
        else:
            for l in self._lis[::-1]:
                self._lis_str = self._lis_str + '{} 阈值：{}，分数：{}\n'.format(index, l.get("threshold"), l.get("f"))
                index = index + 1
        self.print_info("6.评估【共用时{}】".format(self._assessment_time))
        self.print_text(
            "捕捉到异常（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._catch_num, self._catch_interval_num, self._catch_interval_str))
        self.print_text(
            "默认阈值：{}，最佳F分值：{}，精度:{}，召回率：{}"
                .format(round(self._threshold_value, 7), self._f_score, self._precision, self._recall))
        self.print_text("F-分数简易排名：{}".format(self._lis_str))
        self.print_text(
            "【TP】成功监测出的异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._tp_num, self._tp_interval_num, self._tp_interval_str))
        self.print_text(
            "【FP】未标记但超过阈值的点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._fp_num, self._fp_interval_num, self._fp_interval_str))
        self.print_text(
            "【FN】漏报异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._fn_num, self._fn_interval_num, self._fn_interval_str))

    def file_name_converter(self, suffix):
        """
        获得缓存路径
        """
        if self._is_local:
            return "../cache/{}/{}_{}".format(suffix, self._train_file, self._test_file)
        else:
            return "cache/{}/{}_{}".format(suffix, self._train_file, self._test_file)

    def read_probability(self):
        self._tc_1.start()
        name = self.file_name_converter("probability")
        db = shelve.open(name)
        self._has_error = db["_has_error"]
        self._error_str = db["_error_str"]
        self._get_train_file_time = db["_get_train_file_time"]
        self._src_train_num = db["_src_train_num"]
        self._src_train_label_num = db["_src_train_label_num"]
        self._src_train_label_proportion = db["_src_train_label_proportion"]
        self._src_train_timestamps = db["_src_train_timestamps"]
        self._src_train_values = db["_src_train_values"]
        self._get_test_file_time = db["_get_test_file_time"]
        self._src_test_num = db["_src_test_num"]
        self._src_test_label_num = db["_src_test_label_num"]
        self._src_test_label_proportion = db["_src_test_label_proportion"]
        self._src_test_timestamps = db["_src_test_timestamps"]
        self._src_test_values = db["_src_test_values"]
        self._handel_data_time = db["_handel_data_time"]
        self._check_m_s_time = db["_check_m_s_time"]
        self._miss_insert_re_time = db["_miss_insert_re_time"]
        self._fill_step = db["_fill_step"]
        self._mean = db["_mean"]
        self._std = db["_std"]
        self._train_data_num = db["_train_data_num"]
        self._fill_train_label_num = db["_fill_train_label_num"]
        self._fill_train_label_proportion = db["_fill_train_label_proportion"]
        self._fill_train_num = db["_fill_train_num"]
        self._train_missing_interval_num = db["_train_missing_interval_num"]
        self._train_missing_str = db["_train_missing_str"]
        self._fill_train_timestamps = db["_fill_train_timestamps"]
        self._fill_train_values = db["_fill_train_values"]
        self._test_data_num = db["_test_data_num"]
        self._fill_test_label_num = db["_fill_test_label_num"]
        self._fill_test_label_proportion = db["_fill_test_label_proportion"]
        self._fill_test_num = db["_fill_test_num"]
        self._test_missing_interval_num = db["_test_missing_interval_num"]
        self._test_missing_str = db["_test_missing_str"]
        self._fill_test_timestamps = db["_fill_test_timestamps"]
        self._fill_test_values = db["_fill_test_values"]
        self._std_time = db["_std_time"]
        self._fill_train_timestamps = db["_fill_train_timestamps"]
        self._std_train_values = db["_std_train_values"]
        self._fill_test_timestamps = db["_fill_test_timestamps"]
        self._std_test_values = db["_std_test_values"]
        self._model_time = db["_model_time"]
        self._trainer_time = db["_trainer_time"]
        self._predictor_time = db["_predictor_time"]
        self._fit_time = db["_fit_time"]
        self._epoch_time = db["_epoch_time"]
        self._epoch_list = db["_epoch_list"]
        self._lr_list = db["_lr_list"]
        self._test_probability_time = db["_test_probability_time"]
        self._train_predict_compute_time = db["_train_predict_compute_time"]
        self._handle_refactor_probability_time = db["_handle_refactor_probability_time"]
        self._real_test_data_num = db["_real_test_data_num"]
        self._real_test_label_num = db["_real_test_label_num"]
        self._real_test_missing = db["_real_test_missing"]
        self._real_test_missing_num = db["_real_test_missing_num"]
        self._real_test_label_proportion = db["_real_test_label_proportion"]
        self._save_probability_time = db["_save_probability_time"]
        self._real_test_data_num = db["_real_test_data_num"]
        self._real_test_label_num = db["_real_test_label_num"]
        self._real_test_label_proportion = db["_real_test_label_proportion"]
        self._real_test_timestamps = db["_real_test_timestamps"]
        self._real_test_values = db["_real_test_values"]
        self._test_scores = db["_test_scores"]
        self._real_test_labels = db["_real_test_labels"]
        self._get_file_time = db["_get_file_time"]
        db.close()
        self._read_probability_time = self._tc_1.get_s() + '秒'
        self.print_info("读取重构概率等数据【共用时{}】".format(self._read_probability_time))

    def time_sort(self):
        self._time_list_1 = [TimeUse(self._get_train_file_time, "1.获取训练数据集"),
                             TimeUse(self._get_test_file_time, "2.获取测试数据集"),
                             TimeUse(self._check_m_s_time, "3.检验数据并计算综合平均值与标准差"),
                             TimeUse(self._miss_insert_re_time, "4.定位缺失点并填充重构数据集"),
                             TimeUse(self._std_time, "5.标准化训练数据"),
                             TimeUse(self._model_time, "6.构建Donut模型"),
                             TimeUse(self._trainer_time, "7.构造训练器"),
                             TimeUse(self._predictor_time, "8.构造预测器"),
                             TimeUse(self._fit_time, "9.训练器训练模型"),
                             TimeUse(self._test_probability_time, "10.预测器获取重构概率"),
                             TimeUse(self._handle_refactor_probability_time, "11.处理重构概率，获得真实测试数据集"),
                             TimeUse(self._save_probability_time, "12.存储重构概率等数据"),
                             TimeUse(self._assessment_time, "13.评估"),
                             TimeUse(self._save_all_time, "14.存储所有数据")
                             ]
        self._time_list_2 = [TimeUse(self._get_file_time, "1.获取训练与测试数据集"),
                             TimeUse(self._handel_data_time, "2.数据处理"),
                             TimeUse(self._train_predict_compute_time, "3.训练与预测，获得重构概率"),
                             TimeUse(self._handle_refactor_probability_time, "4.处理重构概率，获得真实测试数据集"),
                             TimeUse(self._save_probability_time, "5.存储重构概率等数据"),
                             TimeUse(self._assessment_time, "6.评估"),
                             TimeUse(self._save_all_time, "7.存储所有数据")
                             ]
        self._time_list_1 = np.array(self._time_list_1)
        self._time_list_2 = np.array(self._time_list_2)
        self._sorted_time_list_1 = sorted(self._time_list_1)
        self._sorted_time_list_2 = sorted(self._time_list_2)
        self._t_use_1 = []
        self._t_name_1 = []
        self.print_info("用时排名正序")
        for i, t in enumerate(self._sorted_time_list_1):
            self.print_text("第{}：{}用时{}".format(i + 1, t.name, t.use))
            self._t_use_1.append(t.use)
            self._t_name_1.append(t.name)
        self._t_use_2 = []
        self._t_name_2 = []
        self.print_info("用时排名正序")
        for i, t in enumerate(self._sorted_time_list_1):
            self.print_text("第{}：{}用时{}".format(i + 1, t.name, t.use))
            self._t_use_2.append(t.use)
            self._t_name_2.append(t.name)

    def save_all(self):
        self._tc_1.start()
        name = self.file_name_converter("result")
        db = shelve.open(name)
        db["_test_scores"] = self._test_scores
        db["_real_test_labels"] = self._real_test_labels
        db["_get_file_time"] = self._get_file_time
        db["_real_test_missing"] = self._real_test_missing
        db["_has_error"] = self._has_error
        db["_error_str"] = self._error_str
        db["_get_train_file_time"] = self._get_train_file_time
        db["_src_train_num"] = self._src_train_num
        db["_src_train_label_num"] = self._src_train_label_num
        db["_src_train_label_proportion"] = self._src_train_label_proportion
        db["_src_train_timestamps"] = self._src_train_timestamps
        db["_src_train_values"] = self._src_train_values
        db["_get_test_file_time"] = self._get_test_file_time
        db["_src_test_num"] = self._src_test_num
        db["_src_test_label_num"] = self._src_test_label_num
        db["_src_test_label_proportion"] = self._src_test_label_proportion
        db["_src_test_timestamps"] = self._src_test_timestamps
        db["_src_test_values"] = self._src_test_values
        db["_handel_data_time"] = self._handel_data_time
        db["_check_m_s_time"] = self._check_m_s_time
        db["_miss_insert_re_time"] = self._miss_insert_re_time
        db["_fill_step"] = self._fill_step
        db["_mean"] = self._mean
        db["_std"] = self._std
        db["_train_data_num"] = self._train_data_num
        db["_fill_train_label_num"] = self._fill_train_label_num
        db["_fill_train_label_proportion"] = self._fill_train_label_proportion
        db["_fill_train_num"] = self._fill_train_num
        db["_train_missing_interval_num"] = self._train_missing_interval_num
        db["_train_missing_str"] = self._train_missing_str
        db["_fill_train_timestamps"] = self._fill_train_timestamps
        db["_fill_train_values"] = self._fill_train_values
        db["_test_data_num"] = self._test_data_num
        db["_fill_test_label_num"] = self._fill_test_label_num
        db["_fill_test_label_proportion"] = self._fill_test_label_proportion
        db["_fill_test_num"] = self._fill_test_num
        db["_test_missing_interval_num"] = self._test_missing_interval_num
        db["_test_missing_str"] = self._test_missing_str
        db["_fill_test_timestamps"] = self._fill_test_timestamps
        db["_fill_test_values"] = self._fill_test_values
        db["_std_time"] = self._std_time
        db["_fill_train_timestamps"] = self._fill_train_timestamps
        db["_std_train_values"] = self._std_train_values
        db["_fill_test_timestamps"] = self._fill_test_timestamps
        db["_std_test_values"] = self._std_test_values
        db["_model_time"] = self._model_time
        db["_trainer_time"] = self._trainer_time
        db["_predictor_time"] = self._predictor_time
        db["_fit_time"] = self._fit_time
        db["_epoch_time"] = self._epoch_time
        db["_epoch_list"] = self._epoch_list
        db["_lr_list"] = self._lr_list
        db["_test_probability_time"] = self._test_probability_time
        db["_train_predict_compute_time"] = self._train_predict_compute_time
        db["_handle_refactor_probability_time"] = self._handle_refactor_probability_time
        db["_real_test_data_num"] = self._real_test_data_num
        db["_real_test_label_num"] = self._real_test_label_num
        db["_real_test_missing_num"] = self._real_test_missing_num
        db["_real_test_label_proportion"] = self._real_test_label_proportion
        db["_save_probability_time"] = self._save_probability_time
        db["_assessment_time"] = self._assessment_time
        db["_catch_num"] = self._catch_num
        db["_catch_interval_num"] = self._catch_interval_num
        db["_catch_interval_str"] = self._catch_interval_str
        db["_threshold_value"] = self._threshold_value
        db["_f_score"] = self._f_score
        db["_precision"] = self._precision
        db["_recall"] = self._recall
        db["_tp_num"] = self._tp_num
        db["_tp_interval_num"] = self._tp_interval_num
        db["_tp_interval_str"] = self._tp_interval_str
        db["_fp_num"] = self._fp_num
        db["_fp_interval_num"] = self._fp_interval_num
        db["_fp_interval_str"] = self._fp_interval_str
        db["_fn_num"] = self._fn_num
        db["_fn_interval_num"] = self._fn_interval_num
        db["_fn_interval_str"] = self._fn_interval_str
        db["_real_test_timestamps"] = self._real_test_timestamps
        db["_real_test_values"] = self._real_test_values
        db["_test_scores"] = self._test_scores
        db["_real_test_labels"] = self._real_test_labels
        db["_get_file_time"] = self._get_file_time
        self._save_all_time = self._tc_1.get_s() + "秒"
        db["_save_all_time"] = self._save_all_time
        db["_lis"] = self._lis
        db["_lis_str"] = self._lis_str
        db.close()
        self.print_info("7.存储所有数据【共用时{}】".format(self._save_all_time))

    def read_all(self):
        self._tc_1.start()
        name = self.file_name_converter("result")
        db = shelve.open(name)
        self._has_error = db["_has_error"]
        self._error_str = db["_error_str"]
        self._get_train_file_time = db["_get_train_file_time"]
        self._src_train_num = db["_src_train_num"]
        self._src_train_label_num = db["_src_train_label_num"]
        self._src_train_label_proportion = db["_src_train_label_proportion"]
        self._src_train_timestamps = db["_src_train_timestamps"]
        self._src_train_values = db["_src_train_values"]
        self._get_test_file_time = db["_get_test_file_time"]
        self._src_test_num = db["_src_test_num"]
        self._src_test_label_num = db["_src_test_label_num"]
        self._src_test_label_proportion = db["_src_test_label_proportion"]
        self._src_test_timestamps = db["_src_test_timestamps"]
        self._src_test_values = db["_src_test_values"]
        self._handel_data_time = db["_handel_data_time"]
        self._check_m_s_time = db["_check_m_s_time"]
        self._miss_insert_re_time = db["_miss_insert_re_time"]
        self._fill_step = db["_fill_step"]
        self._mean = db["_mean"]
        self._std = db["_std"]
        self._train_data_num = db["_train_data_num"]
        self._fill_train_label_num = db["_fill_train_label_num"]
        self._fill_train_label_proportion = db["_fill_train_label_proportion"]
        self._fill_train_num = db["_fill_train_num"]
        self._train_missing_interval_num = db["_train_missing_interval_num"]
        self._train_missing_str = db["_train_missing_str"]
        self._fill_train_timestamps = db["_fill_train_timestamps"]
        self._fill_train_values = db["_fill_train_values"]
        self._test_data_num = db["_test_data_num"]
        self._fill_test_label_num = db["_fill_test_label_num"]
        self._fill_test_label_proportion = db["_fill_test_label_proportion"]
        self._fill_test_num = db["_fill_test_num"]
        self._test_missing_interval_num = db["_test_missing_interval_num"]
        self._test_missing_str = db["_test_missing_str"]
        self._fill_test_timestamps = db["_fill_test_timestamps"]
        self._fill_test_values = db["_fill_test_values"]
        self._std_time = db["_std_time"]
        self._fill_train_timestamps = db["_fill_train_timestamps"]
        self._std_train_values = db["_std_train_values"]
        self._fill_test_timestamps = db["_fill_test_timestamps"]
        self._std_test_values = db["_std_test_values"]
        self._model_time = db["_model_time"]
        self._trainer_time = db["_trainer_time"]
        self._predictor_time = db["_predictor_time"]
        self._fit_time = db["_fit_time"]
        self._epoch_time = db["_epoch_time"]
        self._epoch_list = db["_epoch_list"]
        self._lr_list = db["_lr_list"]
        self._test_probability_time = db["_test_probability_time"]
        self._train_predict_compute_time = db["_train_predict_compute_time"]
        self._handle_refactor_probability_time = db["_handle_refactor_probability_time"]
        self._real_test_data_num = db["_real_test_data_num"]
        self._real_test_label_num = db["_real_test_label_num"]
        self._real_test_missing_num = db["_real_test_missing_num"]
        self._real_test_label_proportion = db["_real_test_label_proportion"]
        self._save_probability_time = db["_save_probability_time"]
        self._assessment_time = db["_assessment_time"]
        self._catch_num = db["_catch_num"]
        self._catch_interval_num = db["_catch_interval_num"]
        self._catch_interval_str = db["_catch_interval_str"]
        self._threshold_value = db["_threshold_value"]
        self._f_score = db["_f_score"]
        self._precision = db["_precision"]
        self._recall = db["_recall"]
        self._tp_num = db["_tp_num"]
        self._tp_interval_num = db["_tp_interval_num"]
        self._tp_interval_str = db["_tp_interval_str"]
        self._fp_num = db["_fp_num"]
        self._fp_interval_num = db["_fp_interval_num"]
        self._fp_interval_str = db["_fp_interval_str"]
        self._fn_num = db["_fn_num"]
        self._fn_interval_num = db["_fn_interval_num"]
        self._fn_interval_str = db["_fn_interval_str"]
        self._save_all_time = db["_save_all_time"]
        self._real_test_data_num = db["_real_test_data_num"]
        self._real_test_label_num = db["_real_test_label_num"]
        self._real_test_missing_num = db["_real_test_missing_num"]
        self._real_test_label_proportion = db["_real_test_label_proportion"]
        self._real_test_timestamps = db["_real_test_timestamps"]
        self._real_test_values = db["_real_test_values"]
        self._test_scores = db["_test_scores"]
        self._real_test_labels = db["_real_test_labels"]
        self._get_file_time = db["_get_file_time"]
        self._lis = db["_lis"]
        self._lis_str = db["_lis_str"]
        db.close()
        self._read_all_time = self._tc_1.get_s() + '秒'
        self.print_info("读取重构概率等数据【共用时{}】".format(self._read_all_time))

    def check_file(self, param):
        """
        是否有对应缓存+缓存时间
        Returns:
            是否存在缓存
            缓存文件信息
        """
        name = self.file_name_converter(param)
        cache_name = name + '.db'
        exist = os.path.exists(cache_name)
        if exist:
            file = os.stat(cache_name)
            return exist, "文件名：{},文件大小：{} 字节\n" \
                          "最后一次访问时间:{},最后一次修改时间：{}" \
                .format(cache_name, file.st_size, format_time(file.st_atime), format_time(file.st_mtime))
        else:
            return exist, "该配置无缓存,默认缓存数据"

    def show_cache_probability(self):
        if self._has_error:
            self.print_warn(self._error_str)
            return
        self.print_text("获取训练数据【共用时{}】".format(self._get_train_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_train_num, self._src_train_label_num, self._src_train_label_proportion))
        self.show_line_chart(self._src_train_timestamps, self._src_train_values, 'original csv train data')
        self.print_text("获取测试数据【共用时{}】".format(self._get_test_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_test_num, self._src_test_label_num, self._src_test_label_proportion))
        self.show_line_chart(self._src_test_timestamps, self._src_test_values, 'original csv test data')
        self.print_info("2.数据处理【共用时{}】".format(self._handel_data_time))
        self.print_text("检验数据并计算综合平均值与标准差【共用时:{}】".format(self._check_m_s_time))
        self.print_text("定位缺失点并填充重构数据集【共用时:{}】".format(self._miss_insert_re_time))
        self.print_text("时间戳步长:{}".format(self._fill_step))
        self.print_text("平均值：{}，标准差：{}".format(self._mean, self._std))
        self.print_text("训练数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._train_data_num, self._fill_train_label_num, self._fill_train_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_train_num, self._train_missing_interval_num, self._train_missing_str))
        self.show_line_chart(self._fill_train_timestamps, self._fill_train_values, 'filled train data')
        self.print_text("测试数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._test_data_num, self._fill_test_label_num, self._fill_test_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_test_num, self._test_missing_interval_num, self._test_missing_str))
        self.show_line_chart(self._fill_test_timestamps, self._fill_test_values, 'filled test data')
        self.print_info("标准化训练数据【共用时{}】".format(self._std_time))
        # 显示标准化后的训练数据
        self.show_line_chart(self._fill_train_timestamps, self._std_train_values, 'standardized train data')
        self.show_line_chart(self._fill_test_timestamps, self._std_test_values, 'standardized test data')
        self.print_info("训练与预测，获得重构概率")
        self.print_text("构建Donut模型【共用时{}】".format(self._model_time))
        self.print_text("构造训练器【共用时{}】".format(self._trainer_time))
        self.print_text("构造预测器【共用时{}】".format(self._predictor_time))
        self.print_text("训练器训练模型【共用时{}】".format(self._fit_time))
        self.print_text("所有epoch【共用时：{}】".format(self._epoch_time))
        self.print_text("退火学习率 学习率随epoch变化")
        self.show_line_chart(self._epoch_list, self._lr_list, 'annealing learning rate')
        self.print_text("预测器获取重构概率【共用时{}】".format(self._test_probability_time))
        self.print_text("【共用时{}】".format(self._train_predict_compute_time))
        self.print_info("4.处理重构概率，获得真实测试数据集【共用时{}】".format(self._handle_refactor_probability_time))
        self.print_text("实际测试数据集")
        self.show_test_score()
        self.print_text("共{}条数据,有{}个标注，有{}个缺失数据，标签比例约为{:.2%}"
                        .format(self._real_test_data_num, self._real_test_label_num, self._real_test_missing_num,
                                self._real_test_label_proportion))
        self.print_info("5.存储重构概率等数据【共用时{}】".format(self._save_probability_time))

    def show_cache(self):
        if self._has_error:
            self.print_warn(self._error_str)
            return
        self.print_text("获取训练数据【共用时{}】".format(self._get_train_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_train_num, self._src_train_label_num, self._src_train_label_proportion))
        self.show_line_chart(self._src_train_timestamps, self._src_train_values, 'original csv train data')
        self.print_text("获取测试数据【共用时{}】".format(self._get_test_file_time))
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._src_test_num, self._src_test_label_num, self._src_test_label_proportion))
        self.show_line_chart(self._src_test_timestamps, self._src_test_values, 'original csv test data')
        self.print_info("2.数据处理【共用时{}】".format(self._handel_data_time))
        self.print_text("检验数据并计算综合平均值与标准差【共用时:{}】".format(self._check_m_s_time))
        self.print_text("定位缺失点并填充重构数据集【共用时:{}】".format(self._miss_insert_re_time))
        self.print_text("时间戳步长:{}".format(self._fill_step))
        self.print_text("平均值：{}，标准差：{}".format(self._mean, self._std))
        self.print_text("训练数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._train_data_num, self._fill_train_label_num, self._fill_train_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_train_num, self._train_missing_interval_num, self._train_missing_str))
        self.show_line_chart(self._fill_train_timestamps, self._fill_train_values, 'filled train data')
        self.print_text("测试数据")
        self.print_text("共{}条数据,有{}个标注，标签比例约为{:.2%}"
                        .format(self._test_data_num, self._fill_test_label_num, self._fill_test_label_proportion))
        self.print_text("补充{}个时间戳数据,共有{}段连续缺失 \n {}"
                        .format(self._fill_test_num, self._test_missing_interval_num, self._test_missing_str))
        self.show_line_chart(self._fill_test_timestamps, self._fill_test_values, 'filled test data')
        self.print_info("标准化训练数据【共用时{}】".format(self._std_time))
        # 显示标准化后的训练数据
        self.show_line_chart(self._fill_train_timestamps, self._std_train_values, 'standardized train data')
        self.show_line_chart(self._fill_test_timestamps, self._std_test_values, 'standardized test data')
        self.print_info("训练与预测，获得重构概率")
        self.print_text("构建Donut模型【共用时{}】".format(self._model_time))
        self.print_text("构造训练器【共用时{}】".format(self._trainer_time))
        self.print_text("构造预测器【共用时{}】".format(self._predictor_time))
        self.print_text("训练器训练模型【共用时{}】".format(self._fit_time))
        self.print_text("所有epoch【共用时：{}】".format(self._epoch_time))
        self.print_text("退火学习率 学习率随epoch变化")
        self.show_line_chart(self._epoch_list, self._lr_list, 'annealing learning rate')
        self.print_text("预测器获取重构概率【共用时{}】".format(self._test_probability_time))
        self.print_text("【共用时{}】".format(self._train_predict_compute_time))
        self.print_info("4.处理重构概率，获得真实测试数据集【共用时{}】".format(self._handle_refactor_probability_time))
        self.print_text("实际测试数据集")
        self.show_test_score()
        self.print_text("共{}条数据,有{}个标注，有{}个缺失数据，标签比例约为{:.2%}"
                        .format(self._real_test_data_num, self._real_test_label_num, self._real_test_missing_num,
                                self._real_test_label_proportion))
        self.print_info("5.存储重构概率等数据【共用时{}】".format(self._save_probability_time))
        self.print_info("6.评估【共用时{}】".format(self._assessment_time))
        self.print_text(
            "捕捉到异常（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._catch_num, self._catch_interval_num, self._catch_interval_str))
        self.print_text(
            "默认阈值：{}，最佳F分值：{}，精度:{}，召回率：{}"
                .format(round(self._threshold_value, 7), self._f_score, self._precision, self._recall))
        self.print_text("F-分数简易排名：{}".format(self._lis_str))
        self.print_text(
            "【TP】成功监测出的异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._tp_num, self._tp_interval_num, self._tp_interval_str))
        self.print_text(
            "【FP】未标记但超过阈值的点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._fp_num, self._fp_interval_num, self._fp_interval_str))
        self.print_text(
            "【FN】漏报异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}"
                .format(self._fn_num, self._fn_interval_num, self._fn_interval_str))
        self.print_info("7.存储重构概率等数据【共用时{}】".format(self._save_all_time))
        self.time_sort()
