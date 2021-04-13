import streamlit as st

from donut.cache import is_has_cache
from donut.data import show_cache_data, show_new_data
from donut.utils import handle_src_threshold_value

st.title('Donut')
file_option = st.selectbox('数据来源', ('选择储存至项目中的文件', '上传文件'))
if file_option == '选择储存至项目中的文件':
    file_name = str(st.text_input('文件名【sample_data目录下】', "test.csv"))
    test_portion = float(st.text_input('test portion', 0.3))
    src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
    src_threshold_value = handle_src_threshold_value(src_threshold_value)
    has_cache, cache_text = is_has_cache(file_name, test_portion, src_threshold_value, False)
    st.text(cache_text)
    if has_cache:
        remark = st.selectbox('数据更新（缓存）', ('使用缓存数据', '新建(更新)缓存数据（文件、比例或阈值变更）'))
    else:
        remark = st.selectbox('数据更新（缓存）', ('新建(更新)缓存数据（文件、比例或阈值变更）', '使用缓存数据'))
    button_pd = st.button("分析数据")
    if button_pd:
        # 读取缓存数据
        if remark == "使用缓存数据" and has_cache:
            # 防止文件缺失等异常
            try:
                show_cache_data(False, file_name, test_portion, src_threshold_value, False)
            except Exception:
                st.text("缓存数据损坏，请重新计算")
        else:
            show_new_data(False, file_name, test_portion, src_threshold_value, False, False)
else:
    st.write('上传csv文件，进行数据转换 :wave:')
    file = st.file_uploader('上传文件', type=['csv'], key=None)
    if file is None:
        st.warning("请上传文件")
    else:
        test_portion = float(st.text_input('test portion', 0.3))
        src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
        src_threshold_value = handle_src_threshold_value(src_threshold_value)
        file_name = file.name
        has_cache, cache_text = is_has_cache('upload_' + file_name, test_portion, src_threshold_value, False)
        st.text(cache_text)
        if has_cache:
            remark = st.selectbox('数据更新（缓存）', ('使用缓存数据', '新建(更新)缓存数据（文件、比例或阈值变更）'))
        else:
            remark = st.selectbox('数据更新（缓存）', ('新建(更新)缓存数据（文件、比例或阈值变更）', '使用缓存数据'))
        button_pd = st.button("分析数据")
        if button_pd:
            # 读取缓存数据
            if remark == "使用缓存数据" and has_cache:
                # 防止文件缺失等异常
                try:
                    show_cache_data(False, file_name, test_portion, src_threshold_value, False)
                except Exception:
                    st.text("缓存数据损坏，请重新计算")
            else:
                show_new_data(False, file, test_portion, src_threshold_value, True, False)
