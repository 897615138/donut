import pandas as pd
import streamlit as st


def text(content):
    """
    打印文字
    Args:
        content: 内容
    """
    st.text(content)


def prepare_data_one(train_timestamp, train_values, test_timestamp, test_values):
    """
    原始数据与测试训练数据多图显示
    Args:
        train_timestamp: 训练数据时间轴
        train_values:  训练数据值
        test_timestamp: 测试数据时间轴
        test_values:  测试数据值
    """
    line_chart(train_timestamp, train_values, 'train_data')
    line_chart(test_timestamp, test_values, 'test_data')


def show_test_score(test_timestamp, test_values, test_scores):
    """
    测试数据与分数单图显示
    Args:
        test_timestamp: 测试数据时间戳
        test_values: 测试数据值
        test_scores: 测试分数
    """
    line_chart(test_timestamp, test_values, 'test_data')
    line_chart(test_timestamp, test_scores, 'test_scores')


def line_chart(x, y, name):
    """
    折线图
    Args:
        x: x轴数据
        y: y轴数据
        name: 显示名称
    """
    df = pd.DataFrame(y, index=x, columns=[name])
    st.line_chart(df)


def special_anomaly(special_anomaly_t, special_anomaly_v, special_anomaly_s):
    """
    特殊点展示
    Args:
        special_anomaly_t: 特殊异常点时间戳
        special_anomaly_v: 特殊异常点值
        special_anomaly_s: 特殊异常点分数
    """

    line_chart(special_anomaly_t, special_anomaly_v, 'special_anomaly_value')
    line_chart(special_anomaly_t, special_anomaly_s, 'special_anomaly_score')


def dot_chart(x, y, name):
    """
    绘制折线图
    Args:
        x: x轴数据
        y: y轴数据
        name: 名称
    """
    df = pd.DataFrame(y, index=x, columns=[name])
    st.table(df)


def bar_chart(chart_data):
    """
    柱状图
    Args:
        chart_data: 数据
    """
    st.bar_chart(chart_data)
