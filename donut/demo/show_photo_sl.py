
import matplotlib.pyplot as plt
import streamlit as st
from donut.demo.show_photo_plt import plot_missing

def prepare_data_one(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing,
                     test_missing):
    """
      原始数据与测试训练数据多图显示
    """
    plt.figure(figsize=(50, 10), dpi=1024)
    plt.figure(1)
    plt.title("prepare the csv_data")
    ax1 = plt.subplot(211)
    ax1.plot(base_timestamp, base_values, color='r', label='original csv_data')
    ax1.set_title("original csv_data")

    ax2 = plt.subplot(212)
    ax2.plot(train_timestamp, train_values, label='train_data', color="y")
    ax2.plot(test_timestamp, test_values, label='test_data', color='b')
    plot_missing(ax2, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
    ax2.set_title("training and testing csv_data")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing,
                     test_missing):
    """
      原始数据与测试训练数据单图显示
    """
    plt.figure(figsize=(50, 10), dpi=1024)
    plt.plot(base_timestamp, base_values, label='original csv_data')
    plt.plot(train_timestamp, train_values, label='train_data')
    plt.plot(test_timestamp, test_values, label='test_data')
    plot_missing(plt, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
    plt.title("prepare the csv_data")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


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