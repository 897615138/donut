import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def prepare_data_one(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                     train_missing,
                     test_missing):
    """
      原始数据与测试训练数据单图显示
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.figure(1)
    plt.plot(base_timestamp, base_values, label='original csv_data')
    plt.plot(train_timestamp, train_values, label='train_data')
    plt.plot(test_timestamp, test_values, label='test_data')
    # plot_missing(matplotlib.pyplot, train_missing, train_timestamp, train_values, test_missing, test_timestamp,test_values)
    plt.title("prepare the csv_data")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()


# def plot_missing(p, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values):
#     """
#       缺失点在图上特别标注
#     """
#     for i, is_missing in enumerate(train_missing):
#         if 1 == is_missing:
#             p.plot(train_timestamp[i], train_values[i], 'r')
#     for i, is_missing in enumerate(test_missing):
#         if 1 == is_missing:
#             p.plot(test_timestamp[i], test_values[i], 'r')


def prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                     train_missing,
                     test_missing):
    """
      原始数据与测试训练数据多图显示
    """
    plt.figure(figsize=(40, 10), dpi=128)
    plt.figure(1)
    plt.title("prepare the csv_data")
    ax1 = plt.subplot(211)
    ax1.plot(base_timestamp, base_values, color='r', label='original csv_data')
    ax1.set_title("original csv_data")

    ax2 = plt.subplot(212)
    ax2.plot(train_timestamp, train_values, label='train_data', color="y")
    ax2.plot(test_timestamp, test_values, label='test_data', color='b')
    # plot_missing(ax2, train_missing, train_timestamp, train_values, test_missing, test_timestamp, test_values)
    ax2.set_title("training and testing data")
    plt.show()


def show_test_score(test_timestamp, test_values, test_scores):
    """
      测试数据与分数单图显示
    """
    # plt.figure(figsize=(40, 10), dpi=128)
    plt.figure(2)
    plt.plot(test_timestamp, test_values, label='test data')
    plt.plot(test_timestamp, test_scores, label='test score')
    plt.title("test data and score")
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()


# show_test_score(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
