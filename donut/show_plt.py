import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def prepare_data_one(src_timestamps, src_values, train_timestamps, train_values, test_timestamps,
                     test_values):
    """
    原始数据与测试训练数据多图显示
    Args:
        src_timestamps: 原始时间戳
        src_values: 原始值
        train_timestamps: 训练数据时间戳
        train_values: 训练数据值
        test_timestamps: 测试数据时间戳
        test_values: 测试数据值
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.figure(1)
    plt.title("prepare the csv_data")
    plt.plot(src_timestamps, src_values, color='r', label='original csv_data')
    plt.plot(train_timestamps, train_values, label='train_data', color="y")
    plt.plot(test_timestamps, test_values, label='test_data', color='b')
    plt.title("original csv_data & training and testing data")


def show_test_score(test_timestamp, test_values, test_scores):
    """
    测试数据与分数单图显示
    Args:
        test_timestamp: 测试时间戳
        test_values: 测试值
        test_scores: 测试分数
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.plot(test_timestamp, test_values, label='test data')
    plt.plot(test_timestamp, test_scores, label='test score')
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
    plt.figure(figsize=(40, 10), dpi=128)
    # plt.figure(1)
    plt.plot(x, y, label=name)
    plt.title("name")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()
