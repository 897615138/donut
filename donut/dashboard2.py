from donut.cache import is_has_cache
from donut.dashboard_support import Dashboard
from donut.data import show_cache_data, show_new_data


def dashboard_plt(file_name, test_portion, src_threshold_value, use_cache):
    """

    Args:
        use_cache: 是否使用缓存数据
        file_name: 文件名（路径）
        test_portion: 测试数据比例
        src_threshold_value: 阈值

    Returns:图片信息

    """
    a = file_name.split("/")
    real_name = a[len(a) - 1]
    has_cache = is_has_cache(real_name, test_portion, src_threshold_value, True)
    if use_cache and has_cache:
        try:
            show_cache_data(True, real_name, test_portion, src_threshold_value, True)
        except Exception:
            show_new_data(True, file_name, test_portion, src_threshold_value, False, True)
    else:
        show_new_data(True, file_name, test_portion, src_threshold_value, False, True)


# dashboard_plt("../sample_data/8192_7.24.csv", 0.3, None, False)
use_cache_result = False
use_cache_probability = True
dashboard = Dashboard(use_plt=True,
                      train_file="4096_14.21.csv",
                      test_file="4096_1.88.csv",
                      is_local=True,
                      is_upload=False,
                      src_threshold_value=None,
                      a=1,
                      use_cache_result=use_cache_result,
                      use_cache_probability=use_cache_probability)
