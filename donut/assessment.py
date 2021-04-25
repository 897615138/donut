import numpy as np

from donut.util.out.out import print_warn, print_info
from donut.util.utils import is_in

"""
评估结果
"""


class Assessment(object):
    def __init__(self, threshold_value, test_scores, real_test_labels, real_test_missing, a=1, use_plt=True):
        self._use_plt = use_plt
        self._a = a
        self._test_scores = test_scores
        self._threshold_value = threshold_value
        self._real_test_labels = real_test_labels
        self._real_test_missing = real_test_missing
        self._f_score, self._catch_num, self._catch_index, self._fp_index, self._fp_num, self._tp_index, self._tp_num, self._fn_index, self._fn_num, self._precision, self._recall = None, None, None, None, None, None, None, None, None, None, None
        self._min_test_score = np.min(self._test_scores)
        self._max_test_score = np.max(self._test_scores)
        self._sorted_test_scores = self._test_scores[np.argsort(self._test_scores)]
        self._test_score_intervals = np.unique(np.diff(self._sorted_test_scores))
        self._test_interval = self._test_score_intervals[1]
        if self._test_interval < 1:
            self._test_interval = 1
        self._real_test_labels_index = list(np.where(self._real_test_labels == 1)[0])
        self._real_test_missing_index = list(np.where(self._real_test_missing == 1)[0])
        # 有阈值进行评估
        if self._threshold_value is not None:
            self.assessment()
            if self._f_score is None:
                print_info(use_plt, "当前阈值无F-score，请确认")
            else:
                if self._f_score < 0.7:
                    print_warn(use_plt, "建议调整阈值分数或使用【默认阈值】以获得更好的效果（F—score）")
        # 计算默认阈值进行评估
        else:
            self.default_assessment()

    def assessment(self):
        self._catch_index = np.where(self._test_scores >= float(self._threshold_value))[0].tolist()
        self._catch_num = np.size(self._catch_index)
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
        return self._threshold_value, self._f_score, self._catch_num, self._catch_index, self._fp_index, self._fp_num, self._tp_index, self._tp_num, self._fn_index, self._fn_num, self._precision, self._recall, self._lis

    def default_assessment(self):
        print_info(self._use_plt, "开始计算默认阈值")
        # 降序训练数据中的异常标签对应分值
        self._threshold_value = self._max_test_score
        self._lis = []
        has_big = False
        while self._threshold_value >= self._min_test_score:
            self.assessment()
            self._threshold_value = round((self._threshold_value - self._test_interval), 7)
            # print(score)
            if self._f_score is not None:
                # print(score, f_score)
                if has_big and (self._f_score < 0.7 or len(self._lis) > 20):
                    break
                if self._f_score >= 0.7:
                    self._test_interval = 1e-1
                    has_big = True
                    catch = {"threshold": self._threshold_value, "num": self._catch_num, "index": self._catch_index,
                             "f": self._f_score, "fpi": self._fp_index, "fpn": self._fp_num, "tpi": self._tp_index,
                             "tpn": self._tp_num, "fni": self._fn_index, "fnn": self._fn_num, "p": self._precision,
                             "r": self._recall}
                    self._lis.append(catch)
        # 字典按照生序排序 取最大的准确度
        if len(self._lis) > 0:
            self._lis = sorted(self._lis, key=lambda dict_catch: (dict_catch['f'], dict_catch['threshold']))
            catch = self._lis[- 1]
            # 最优F-score
            self._threshold_value = catch.get("threshold")
            self._catch_num = catch.get("num")
            self._catch_index = catch.get("index")
            self._f_score = catch.get("f")
            self._fp_index = catch.get("fpi")
            self._fp_num = catch.get("fpn")
            self._tp_index = catch.get("tpi")
            self._tp_num = catch.get("tpn")
            self._fn_index = catch.get("fni")
            self._fn_num = catch.get("fnn")
            self._precision = catch.get("p")
            self._recall = catch.get("r")

    def fn(self):
        """
        FN 漏报的异常
        """
        self._fn_index = list(set(self._real_test_labels_index) - set(self._catch_index))
        for fn in self._fn_index:
            if is_in(fn, self._catch_index, -10, 10) or is_in(fn, self._real_test_missing_index, -10, 10):
                self._fn_index.remove(fn)
        self._fn_num = np.size(self._fn_index)

    def tp(self):
        """
        TP 成功检测出的异常
        """
        self._tp_index = list(set(self._catch_index).intersection(set(self._real_test_labels_index)))
        append_list = []
        for tp in self._tp_index:
            if is_in(tp, self._real_test_labels_index, -10, 10) or is_in(tp, self._real_test_missing_index, -10, 10):
                append_list.append(tp)
        for i in append_list:
            self._tp_index.append(i)
        self._tp_index = set(self._tp_index)
        self._tp_index = list(self._tp_index)
        self._tp_num = np.size(self._tp_index)

    def fp(self):
        """
        FP 未标记但超过阈值 实际为正常点单倍误判为异常点 去除延迟点
        """
        self._fp_index = list(set(self._catch_index) - set(self._real_test_labels_index))
        for fp in self._fp_index:
            if is_in(fp, self._real_test_labels_index, -10, 10) or is_in(fp, self._real_test_missing_index, -10, 10):
                self._fp_index.remove(fp)
        self._fp_num = np.size(self._fp_index)

    def precision(self):
        """
        精度
        """
        if self._tp_num + self._fp_num == 0:
            return None
        self._precision = self._tp_num / (self._tp_num + self._fp_num)

    def recall(self):
        """
        召回率
        """
        if self._tp_num + self._fn_num == 0:
            return None
        self._recall = self._tp_num / (self._tp_num + self._fn_num)

    def f_score(self):
        """
        f-score
        """
        self._precision = float(str(self._precision))
        self._recall = float(self._precision)
        if round(float(self._precision) + float(self._recall), 7) == 0:
            self._f_score = None
        else:
            self._f_score = round((self._a * self._a + 1) * self._precision * self._recall / (
                    self._a * self._a * (self._precision + self._recall)), 7)
