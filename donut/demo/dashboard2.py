import show_plt as sp
import data
from donut.demo.train_prediction import train_prediction
from donut.demo.data import show_cache_data, show_new_data, handle_threshold_value, is_has_cache


def dashboard_plt(file_name, test_portion, src_threshold_value, use_cache):
    """

    Args:
        use_cache: 是否使用缓存数据
        file_name: 文件名（路径）
        test_portion: 测试数据比例
        src_threshold_value: 阈值

    Returns:图片信息

    """
    has_cache = is_has_cache(file_name, test_portion, src_threshold_value)
    if use_cache and has_cache:
        show_cache_data("sl", file_name, test_portion, src_threshold_value)
    else:
        show_new_data("sl", file_name, test_portion, src_threshold_value)

    base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
        data.prepare_data("../../sample_data/1.csv")
    sp.prepare_data_one(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                        train_missing, test_missing)
    test_score = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
    sp.show_test_score(test_timestamp, test_values, test_score)
