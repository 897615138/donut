from donut.demo.cache import is_has_cache
from donut.demo.data import show_cache_data, show_new_data


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
        show_cache_data(True, file_name, test_portion, src_threshold_value)
    else:
        show_new_data(True, file_name, test_portion, src_threshold_value)
