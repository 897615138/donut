import streamlit as st
import donut.demo.show_sl as sl
from donut.demo.data import show_new_data, handle_threshold_value, is_has_cache, gain_data_cache

st.title('Donut')
file_name = str(st.text_input('文件名【sample_data目录下】', "test.csv"))
test_portion = float(st.text_input('test portion', 0.3))
src_threshold_value = st.text_input('阈值（不设置则使用默认值）', "默认阈值")
src_threshold_value = handle_threshold_value(src_threshold_value)
has_cache = is_has_cache(file_name, test_portion, src_threshold_value)
if has_cache:
    st.text("该配置有缓存,建议使用缓存数据")
    remark = st.selectbox('数据更新（缓存）', ('使用缓存数据', '新建(更新)缓存数据（文件、比例或阈值变更）'))
else:
    st.text("该配置无缓存,默认缓存数据")
    remark = st.selectbox('数据更新（缓存）', ('新建(更新)缓存数据（文件、比例或阈值变更）', '使用缓存数据'))
button_pd = st.button("分析数据")

if button_pd:
    # 读取缓存数据
    if remark == "使用缓存数据" and has_cache:
        src_timestamps, src_labels, src_values, src_data_num, src_label_num, src_label_proportion, first_time, \
        fill_timestamps, fill_values, fill_data_num, fill_step, fill_num, second_time, third_time, \
        train_data_num, train_label_num, train_label_proportion, test_data_num, test_label_num, test_label_proportion, \
        mean, std, forth_time, epoch_list, lr_list, epoch_time, fifth_time, src_threshold_value, catch_num, labels_num, \
        accuracy, special_anomaly_num, interval_num, interval_str, special_anomaly_t, special_anomaly_v, special_anomaly_s \
            = gain_data_cache(file_name, test_portion, src_threshold_value)
        sl.line_chart(src_timestamps, src_values, 'original csv_data')
        sl.text("共{}条数据,有{}个标注，标签比例约为{:.2%} \n【分析csv数据,共用时{}】"
                .format(src_data_num, src_label_num, src_label_proportion, first_time))
        sl.line_chart(fill_timestamps, fill_values, 'fill_data')
        sl.text("填充至{}条数据，时间戳步长:{},补充{}个时间戳数据 \n【填充数据，共用时{}】"
                .format(fill_data_num, fill_step, fill_num, second_time))
        sl.text("训练数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                "测试数据量：{}，有{}个标注,标签比例约为{:.2%}\n"
                "【填充缺失数据,共用时{}】"
                .format(train_data_num, train_label_num, train_label_proportion,
                        test_data_num, test_label_num, test_label_proportion,
                        third_time))
        sl.text("平均值：{}，标准差：{}\n【标准化训练和测试数据,共用时{}】".format(mean, std, forth_time))
        sl.line_chart(epoch_list, lr_list, 'annealing_learning_rate')
        sl.text("退火学习率随epoch变化")
        sl.text("【所有epoch共用时：{}】".format(epoch_time))
        sl.text("【训练模型与预测获得测试分数,共用时{}】".format(fifth_time))
        sl.text("默认阈值：{},根据默认阈值获得的异常点数量：{},实际异常标注数量:{}".format(src_threshold_value, catch_num, labels_num))
        if accuracy is not None:
            sl.text("标签准确度:{:.2%}".format(accuracy))
        sl.text("未标记但超过阈值的点（数量：{}）：".format(special_anomaly_num))
        sl.text("共有{}段(处)异常".format(interval_num))
        sl.text(interval_str)
        for i, fill_timestamps in enumerate(special_anomaly_t):
            sl.text("时间戳:{},值:{},分数：{}".format(fill_timestamps, special_anomaly_v[i], special_anomaly_s[i]))

else:
    show_new_data(file_name, test_portion, src_threshold_value)
