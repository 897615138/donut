import matplotlib

matplotlib.use('TkAgg')
import streamlit as st
import pandas as pd


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


def show_test_score(test_timestamp, test_values, test_scores):
    """
      测试数据与分数单图显示
    """
    chart_data = pd.DataFrame(test_values, index=test_timestamp, columns=['test_data'])
    st.line_chart(chart_data)
    chart_data = pd.DataFrame(test_scores, index=test_timestamp, columns=['test_scores'])
    st.line_chart(chart_data)


def fill_data(timestamp, values):
    chart_data = pd.DataFrame(values.tolist(), index=timestamp.tolist(), columns=['fill data'])
    st.line_chart(chart_data)
    return None
