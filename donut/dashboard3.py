from donut.cache import is_has_cache
from donut.data import show_cache_data, self_structure


def dashboard_plt(use_plt=True,
                  train_file="4096_14.21.csv",
                  test_file="4096_1.88.csv",
                  is_local=True,
                  is_upload=False,
                  src_threshold_value=None, use_cache=True):
    """

    Args:
        use_cache: 是否使用缓存数据
        file_name: 文件名（路径）
        test_portion: 测试数据比例
        src_threshold_value: 阈值

    Returns:图片信息

    """
    a = train_file.split("/")
    real_name = a[len(a) - 1]
    has_cache = is_has_cache(real_name, src_threshold_value, True)
    if use_cache and has_cache:
        try:
            show_cache_data(True, real_name, src_threshold_value, True)
        except Exception:
            self_structure(use_plt=True,
                           train_file="4096_14.21.csv",
                           test_file="4096_1.88.csv",
                           is_local=True,
                           is_upload=False,
                           src_threshold_value=None)
    else:
        self_structure(use_plt=True,
                       train_file="4096_14.21.csv",
                       test_file="4096_1.88.csv",
                       is_local=True,
                       is_upload=False,
                       src_threshold_value=None)

