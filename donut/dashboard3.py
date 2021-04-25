import os

import streamlit as st

from donut.dashboard_support import Dashboard
from donut.threshold import handle_src_threshold_value
from donut.util.time_util import format_time

st.title('Donut')
file_option = st.selectbox('数据来源', ('选择储存至项目中的文件', '上传文件'))


def file_name_converter(suffix, train_file, test_file):
    """
    获得缓存路径
    """
    return "cache/{}/{}_{}.db".format(suffix, train_file, test_file)


def has_cache(result_file_path):
    exist = os.path.exists(result_file_path)
    if exist:
        file = os.stat(result_file_path)
        return exist, "该配置有缓存,建议使用缓存数据\n" \
                      "文件名：{},文件大小：{} 字节\n" \
                      "最后一次访问时间:{},最后一次修改时间：{}" \
            .format(result_file_path, file.st_size, format_time(file.st_atime), format_time(file.st_mtime))
    else:
        return exist, "该配置无缓存,默认缓存数据"


if file_option == '选择储存至项目中的文件':
    is_local = False
    train_file_name = str(st.text_input('训练文件名【sample_data目录下】', "4096_1.88.csv"))
    test_file_name = str(st.text_input('测试文件名【sample_data目录下】', "4096_14.21.csv"))
    src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
    src_threshold_value = handle_src_threshold_value(src_threshold_value)
    result_file_path = file_name_converter("result", train_file_name, test_file_name)
    pro_file_path = file_name_converter("probability", train_file_name, test_file_name)
    has_result, result_cache_text = has_cache(result_file_path)
    has_probability, probability_cache_text = has_cache(pro_file_path)
    st.text("结果缓存文件：")
    # st.text(result_file_path)
    st.text(result_cache_text)
    st.text("训练测试文件：")
    # st.text(pro_file_path)
    st.text(probability_cache_text)
    use_cache_result = True
    use_cache_probability = True
    if has_result and has_probability:
        remark = st.selectbox('全部数据（缓存）选择', ('使用缓存数据', '更新缓存数据', '更新评估数据'))
        if remark == '使用缓存数据':
            a = 1
            use_cache_result = True
            use_cache_probability = True
        elif remark == '使用缓存数据':
            a = 1
            use_cache_result = False
            use_cache_probability = True
        else:
            a = st.text_input('a【F-score评估系数】', 1)
            use_cache_result = False
            use_cache_probability = True
    elif has_result and not has_probability:
        remark = st.selectbox('数据（缓存）选择', ('使用缓存数据', '更新缓存数据'))
        if remark == '使用缓存数据':
            a = 1
            use_cache_result: True
            use_cache_probability = False
        else:
            a = st.text_input('a【F-score评估系数】', 1)
            use_cache_result = False
            use_cache_probability = False
    elif not has_result and has_probability:
        remark = st.selectbox('数据（缓存）选择', ('使用已有训练测试结果', '更新训练测试结果'))
        a = st.text_input('a【F-score评估系数】', 1)
        if remark == '使用已有训练测试结果':
            use_cache_result = False
            use_cache_probability = True
        else:
            use_cache_result = False
            use_cache_probability = False
    else:
        st.text("当前无缓存，默认进行缓存")
        a = st.text_input('a【F-score评估系数】', 1)
    button_pd = st.button("分析数据")
    if button_pd:
        dashboard = Dashboard(use_plt=False,
                              train_file=train_file_name,
                              test_file=test_file_name,
                              is_local=False,
                              is_upload=False,
                              src_threshold_value=src_threshold_value,
                              a=a,
                              use_cache_result=use_cache_result,
                              use_cache_probability=use_cache_probability)
else:
    st.write('上传csv文件，进行数据转换 :wave:')
    train_file_name = st.file_uploader('上传文件', type=['csv'], key=None)
    test_file_name = st.file_uploader('上传文件', type=['csv'], key=None)
    if train_file_name is None or test_file_name is None:
        st.warning("请上传文件")
    else:
        src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
        src_threshold_value = handle_src_threshold_value(src_threshold_value)
        result_file_path = file_name_converter("result", train_file_name, test_file_name)
        pro_file_path = file_name_converter("probability", train_file_name, test_file_name)
        has_result, result_cache_text = has_cache('upload_' + result_file_path)
        has_probability, probability_cache_text = has_cache('upload_' + pro_file_path)
        st.text("结果缓存文件：")
        st.text(result_cache_text)
        st.text("训练测试文件：")
        st.text(probability_cache_text)
        use_cache_result = True
        use_cache_probability = True
        if has_result and has_probability:
            remark = st.selectbox('全部数据（缓存）选择', ('使用缓存数据', '更新缓存数据', '更新评估数据'))
            if remark == '使用缓存数据':
                a = 1
                use_cache_result = True
                use_cache_probability = True
            elif remark == '使用缓存数据':
                a = 1
                use_cache_result = False
                use_cache_probability = True
            else:
                a = st.text_input('a【F-score评估系数】', 1)
                use_cache_result = False
                use_cache_probability = False
        elif has_result and not has_probability:
            remark = st.selectbox('数据（缓存）选择', ('使用缓存数据', '更新缓存数据'))
            if remark == '使用缓存数据':
                a = 1
                use_cache_result: True
                use_cache_probability = False
            else:
                a = st.text_input('a【F-score评估系数】', 1)
                use_cache_result = False
                use_cache_probability = False
        elif not has_result and has_probability:
            remark = st.selectbox('数据（缓存）选择', ('使用已有训练测试结果', '更新评估结果'))
            if remark == '使用已有训练测试结果':
                a = 1
                use_cache_result = False
                use_cache_probability = True
            else:
                a = st.text_input('a【F-score评估系数】', 1)
                use_cache_result = False
                use_cache_probability = False
        else:
            st.text("当前无缓存，默认进行缓存")
            a = st.text_input('a【F-score评估系数】', 1)
        button_pd = st.button("分析数据")
        if button_pd:
            dashboard = Dashboard(use_plt=False,
                                  train_file=train_file_name,
                                  test_file=test_file_name,
                                  is_local=False,
                                  is_upload=True,
                                  src_threshold_value=src_threshold_value,
                                  a=a,
                                  use_cache_result=use_cache_result,
                                  use_cache_probability=use_cache_probability)
