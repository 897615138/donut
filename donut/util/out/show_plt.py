import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def prepare_data_one(name, src_name, train_name, test_name, src_timestamps, src_values, train_timestamps, train_values,
                     test_timestamps, test_values):
    """
    原始数据与测试训练数据多图显示
    Args:
        train_name: 训练图例
        test_name: 测试图例
        src_name: 原数据图例
        name: 图名
        src_timestamps: 原始时间戳
        src_values: 原始值
        train_timestamps: 训练数据时间戳
        train_values: 训练数据值
        test_timestamps: 测试数据时间戳
        test_values: 测试数据值
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.figure(1)
    plt.title(name)
    plt.plot(src_timestamps, src_values, label=src_name, color='r')
    plt.plot(train_timestamps, train_values, label=train_name, color="y")
    plt.plot(test_timestamps, test_values, label=test_name, color='b')


def show_test_score(test_timestamp, test_values, test_scores):
    """
    测试数据与分数单图显示
    Args:
        test_timestamp: 测试时间戳
        test_values: 测试值
        test_scores: 测试分数
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.plot(test_timestamp, test_values, label="test_data", color='r')
    plt.plot(test_timestamp, test_scores, label="test_score", color="y")
    plt.title("test data and score")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()


def line_chart(x, y, name):
    """
    折线图
    Args:
        x: x轴
        y: y轴
        name: 名称
    """
    return
    plt.figure(figsize=(40, 10), dpi=128)
    plt.plot(x, y, label=name)
    plt.title(name)
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()
