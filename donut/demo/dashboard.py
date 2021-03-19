import streamlit as st

from donut.demo.data import show_cache_data, show_new_data, handle_threshold_value, is_has_cache

st.title('Donut')
file_name = str(st.text_input('文件名【sample_data目录下】', "test.csv"))
test_portion = float(st.text_input('test portion', 0.3))
src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
st.text("该配置有无缓存{}".format(is_has_cache(file_name, test_portion, src_threshold_value)))
remark = st.selectbox('数据更新（缓存）【测试分数】', ('使用缓存数据', '新建(更新)缓存数据（文件、比例或阈值变更）'))
src_threshold_value = handle_threshold_value(src_threshold_value)
button_pd = st.button("分析数据")

if button_pd:
    # 读取缓存数据
    if remark == "使用缓存数据":
        show_cache_data(file_name, test_portion, src_threshold_value)
    else:
        show_new_data(file_name, test_portion, src_threshold_value)
