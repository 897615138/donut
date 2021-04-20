from donut.assessment import Assessment
from donut.cache import get_test_data
from donut.util.out.out import print_text
from donut.util.time_util import get_constant_timestamp

fill_step = 60
use_plt = True
a = 1
src_threshold_value, test_scores, real_test_labels, real_test_missing = get_test_data()
assessment = Assessment(src_threshold_value, test_scores, real_test_labels, real_test_missing, a, use_plt)
threshold_value, f_score, catch_num, catch_index, fp_index, fp_num, tp_index, tp_num, fn_index, fn_num, precision, recall \
    = assessment.get_assessment()
print_text(use_plt, "捕捉到的异常数：{}".format(catch_num))
print_text(use_plt, "默认阈值：{}，最佳F分值：{}，精度:{}，召回率：{}".format(round(threshold_value, 7), f_score, precision, recall))
tp_interval_num, tp_interval_str = get_constant_timestamp(tp_index, fill_step)
print_text(use_plt, "【TP】成功监测出的异常点（数量：{}）：\n 共有{}段连续  具体为{}".format(tp_num, tp_interval_num, tp_interval_str))
fp_interval_num, fp_interval_str = get_constant_timestamp(fp_index, fill_step)
print_text(use_plt, "【FP】未标记但超过阈值的点（数量：{}）：\n 共有{}段连续  具体为{}".format(fp_num, fp_interval_num, fp_interval_str))
fn_interval_num, fn_interval_str = get_constant_timestamp(fn_index, fill_step)
print_text(use_plt, "【FN】漏报异常点（数量：{}）：\n 共有{}段连续 \n 具体为{}".format(fn_num, fn_interval_num, fn_interval_str))
