import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from donut.demo.show_photo_plt import plot_missing


def prepare_data_one(train_timestamp, train_values, test_timestamp, test_values):
    """
      原始数据与测试训练数据多图显示
    """
    chart_data = pd.DataFrame(train_values, index=train_timestamp, columns=['train_data'])
    st.line_chart(chart_data)
    chart_data = pd.DataFrame(test_values, index=test_timestamp, columns=['test_data'])
    st.line_chart(chart_data)

def source_data(base_timestamp, base_values):
    """
      原始数据
    """
    chart_data = pd.DataFrame(base_values, index=base_timestamp, columns=['original csv_data'])
    st.line_chart(chart_data)
    # plt.figure(figsize=(50, 10), dpi=1024)
    # plt.plot(base_timestamp, base_values, label='original csv_data')
    # plt.title("prepare the csv_data")
    # plt.xlabel('timestamp')
    # plt.ylabel('value')
    # plt.legend()
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot()


# def prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
#                      train_missing,
#                      test_missing):
#     """
#       原始数据与测试训练数据单图显示
#     """
#     plt.figure(figsize=(50, 10), dpi=1024)
#     plt.plot(base_timestamp, base_values, label='original csv_data')
#     plt.plot(train_timestamp, train_values, label='train_data')
#     plt.plot(test_timestamp, test_values, label='test_data')
#     plot_missing(plt, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
#     plt.title("prepare the csv_data")
#     plt.xlabel('timestamp')
#     plt.ylabel('value')
#     plt.legend()
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     st.pyplot()


def show_test_score(test_timestamp, test_values, test_scores):
    """
      测试数据与分数单图显示
    """
    plt.figure(figsize=(50, 10), dpi=1024)
    plt.plot(test_timestamp, test_values, label='test csv_data')
    plt.plot(test_timestamp, test_scores, label='test score')
    plt.title("test csv_data and score")
    plt.xlabel('timestamp')
    # plt.ylabel('value')
    plt.legend()
    # plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
