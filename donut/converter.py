def file_name_converter(file_name, test_portion, threshold_value, is_local):
    """
    获得缓存路径
    Args:
        is_local: 本地照片展示
        file_name: 文件名
        test_portion: 测试数据比例
        threshold_value: 阈值
    Returns:
        缓存文件路径
    """
    if is_local:
        return "../cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)
    else:
        return "cache/" + file_name + "_" + str(test_portion) + "_" + str(threshold_value)
