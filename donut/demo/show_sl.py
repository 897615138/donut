import matplotlib


# 显示plot结果图
from tfsnippet.utils import humanize_duration

matplotlib.use('TkAgg')
import streamlit as st
import pandas as pd


def text(content):
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


# def dot_chart(x, y, name):
#     """
#     点图
#     Args:
#         x: x轴数据
#         y: y轴数据
#         name: 显示名称
#     """
#     df = pd.DataFrame(y, index=x, columns=[name])
#     c = alt.Chart(df).mark_circle().encode(
#         x='x', y='y', size='size', color='c')
#     st.altair_chart(c, width=-1)


def dot_chart(x, y, name):
    df = pd.DataFrame(y, index=x, columns=[name])
    st.table(df)


def print_log(loop):
    loop._require_entered()
    metrics = None
    if loop._within_step:
        loop._commit_step_stop_time()
        metrics = loop._step_metrics
    elif loop._within_epoch:
        loop._commit_epoch_stop_time()
        metrics = loop._epoch_metrics
    else:
        loop._require_context()
    best_mark = ' (*)' if loop._is_best_valid_metric else ''
    st.text(metrics.format_logs() + best_mark)
    println(loop, metrics.format_logs() + best_mark, with_tag=True)
    loop._is_best_valid_metric = False
    metrics.clear()


def println(loop, message, with_tag=False):
    """
    Print `message` via `print_function`.

    Args:
        loop: loop
        message (str): Message to be printed.
        with_tag (bool): Whether or not to add the epoch & step tag?
            (default :obj:`False`)
    """
    loop._require_entered()
    if with_tag:
        def format_tag(v, max_v, name):
            if max_v is not None:
                return '{} {}/{}'.format(name, v, max_v)
            else:
                return '{} {}'.format(name, v)

        if not loop._within_step and not loop._within_epoch:
            loop._require_context()
        tags = []
        if loop._max_epoch != 1:
            tags.append(format_tag(loop._epoch, loop._max_epoch, 'Epoch'))
        tags.append(format_tag(loop._step, loop._max_step, 'Step'))
        if loop._show_eta:
            progress = loop.get_progress()
            if progress is not None:
                eta = loop._eta.get_eta(progress)
                if eta is not None:
                    tags.append('ETA {}'.format(humanize_duration(eta)))
        message = '[{}] {}'.format(', '.join(tags), message)
    loop._print_func(message)
