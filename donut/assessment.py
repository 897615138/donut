import numpy as np

from donut.util.out.out import print_warn, print_info
from donut.util.utils import is_in

"""
评估结果
"""

class Assessment(object):
    def __init__(self, threshold_value, test_scores, real_test_labels, real_test_missing, a=1, use_plt=True):
        self.__use_plt = use_plt
        self.__a = a
        self.__test_scores = test_scores
        self.__threshold_value = threshold_value
        self.__real_test_labels = real_test_labels
        self.__real_test_missing = real_test_missing
        self.__f_score, self.__catch_num, self.__catch_index, self.__fp_index, self.__fp_num, self.__tp_index, self.__tp_num, self.__fn_index, self.__fn_num, self.__precision, self.__recall = None, None, None, None, None, None, None, None, None, None, None
        self.__min_test_score = np.min(self.__test_scores)
        self.__max_test_score = np.max(self.__test_scores)
        self.__sorted_test_scores = self.__test_scores[np.argsort(self.__test_scores)]
        self.__test_score_intervals = np.unique(np.diff(self.__sorted_test_scores))
        self.__test_interval = self.__test_score_intervals[1]
        if self.__test_interval < 1:
            self.__test_interval = 1
        self.__real_test_labels_index = list(np.where(self.__real_test_labels == 1)[0])
        self.__real_test_missing_index = np.asarray(self.__real_test_missing[0]).tolist()
        # 有阈值进行评估
        if self.__threshold_value is not None:
            self.assessment()
            if self.__f_score is None:
                print_info(use_plt, "当前阈值无F-score，请确认")
            else:
                if self.__f_score < 0.7:
                    print_warn(use_plt, "建议调整阈值分数或使用【默认阈值】以获得更好的效果（F—score）")

        # 计算默认阈值进行评估
        else:
            self.default_assessment()

    def assessment(self):
        self.__catch_index = np.where(self.__test_scores >= float(self.__threshold_value))[0].tolist()
        self.__catch_num = np.size(self.__catch_index)
        # FP 未标记但超过阈值 实际为正常点单倍误判为异常点
        self.fp()
        # TP 成功检测出的异常
        self.tp()
        # FN  漏报
        self.fn()
        # Precision精度
        self.precision()
        # 召回率
        self.recall()
        # f-score
        self.f_score()

    def get_assessment(self):
        return self.__threshold_value, self.__f_score, self.__catch_num, self.__catch_index, self.__fp_index, self.__fp_num, self.__tp_index, self.__tp_num, self.__fn_index, self.__fn_num, self.__precision, self.__recall

    def default_assessment(self):
        print_info(self.__use_plt, "开始计算默认阈值")
        # 降序训练数据中的异常标签对应分值
        self.__threshold_value = self.__max_test_score
        lis = []
        has_big = False
        while True and self.__threshold_value >= self.__min_test_score:
            self.assessment()
            self.__threshold_value = round((self.__threshold_value - self.__test_interval), 7)
            # print(score)
            if self.__f_score is not None:
                # print(score, f_score)
                if has_big and self.__f_score < 0.7:
                    break
                if self.__f_score >= 0.7:
                    self.__test_interval = 1e-2
                    has_big = True
                    catch = {"threshold": self.__threshold_value, "num": self.__catch_num, "index": self.__catch_index,
                             "f": self.__f_score, "fpi": self.__fp_index, "fpn": self.__fp_num, "tpi": self.__tp_index,
                             "tpn": self.__tp_num, "fni": self.__fn_index, "fnn": self.__fn_num, "p": self.__precision,
                             "r": self.__recall}
                    lis.append(catch)
        # 字典按照生序排序 取最大的准确度
        if len(lis) > 0:
            lis = sorted(lis, key=lambda dict_catch: (dict_catch['f'], dict_catch['threshold']))
            catch = lis[- 1]
            # 最优F-score
            self.__threshold_value = catch.get("threshold")
            self.__catch_num = catch.get("num")
            self.__catch_index = catch.get("index")
            self.__f_score = catch.get("f")
            self.__fp_index = catch.get("fpi")
            self.__fp_num = catch.get("fpn")
            self.__tp_index = catch.get("tpi")
            self.__tp_num = catch.get("tpn")
            self.__fn_index = catch.get("fni")
            self.__fn_num = catch.get("fnn")
            self.__precision = catch.get("p")
            self.__recall = catch.get("r")

    def fn(self):
        """
        FN 漏报的异常
        """
        self.__fn_index = list(set(self.__real_test_labels_index) - set(self.__catch_index))
        for fn in self.__fn_index:
            if is_in(fn, self.__catch_index, -10, 10) or is_in(fn, self.__real_test_missing_index, -10, 10):
                self.__fn_index.remove(fn)
        self.__fn_num = np.size(self.__fn_index)

    def tp(self):
        """
        TP 成功检测出的异常
        """
        self.__tp_index = list(set(self.__catch_index).intersection(set(self.__real_test_labels_index)))
        append_list = []
        for tp in self.__tp_index:
            if is_in(tp, self.__real_test_labels_index, -10, 10) or is_in(tp, self.__real_test_missing_index, -10, 10):
                append_list.append(tp)
        for i in append_list:
            self.__tp_index.append(i)
        self.__tp_index = set(self.__tp_index)
        self.__tp_index = list(self.__tp_index)
        self.__tp_num = np.size(self.__tp_index)

    def fp(self):
        """
        FP 未标记但超过阈值 实际为正常点单倍误判为异常点 去除延迟点
        """
        self.__fp_index = list(set(self.__catch_index) - set(self.__real_test_labels_index))
        for fp in self.__fp_index:
            if is_in(fp, self.__real_test_labels_index, -10, 10) or is_in(fp, self.__real_test_missing_index, -10, 10):
                self.__fp_index.remove(fp)
        self.__fp_num = np.size(self.__fp_index)

    def precision(self):
        """
        精度
        """
        if self.__tp_num + self.__fp_num == 0:
            return None
        self.__precision = self.__tp_num / (self.__tp_num + self.__fp_num)

    def recall(self):
        """
        召回率
        """
        if self.__tp_num + self.__fn_num == 0:
            return None
        self.__recall = self.__tp_num / (self.__tp_num + self.__fn_num)

    def f_score(self):
        """
        f-score
        """
        if self.__precision + self.__recall == 0:
            self.__f_score = None
        else:
            self.__f_score = round((self.__a * self.__a + 1) * self.__precision * self.__recall / (
                    self.__a * self.__a * (self.__precision + self.__recall)), 7)