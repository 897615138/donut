import donut.show_plt as plt
import donut.show_sl as sl


def show_test_score(use_plt, test_timestamps, test_values, test_scores):
    """
    展示测试数据
    Args:
        use_plt: 展示方式
        test_timestamps: 测试数据时间戳
        test_values: 测试数据值
        test_scores: 测试数据分数
    """
    if use_plt:
        plt.show_test_score(test_timestamps, test_values, test_scores)
    else:
        sl.show_test_score(test_timestamps, test_values, test_scores)


def show_line_chart(use_plt, x, y, name):
    """
    展示折线图
    Args:
        use_plt:  展示方式
        x: x轴
        y: y轴
        name: 名称
    """
    if use_plt:
        plt.line_chart(x, y, name)
    else:
        sl.line_chart(x, y, name)



def print_info(use_plt, content):
    """
       展示文字
       Args:
           use_plt: 展示方式
           content: 文字内容
       """
    if use_plt:
        print(content)
    else:
        sl.info(content)

def print_text(use_plt, content):
    """
    展示文字
    Args:
        use_plt: 展示方式
        content: 文字内容
    """
    if use_plt:
        print(content)
    else:
        sl.text(content)


def print_warn(use_plt, content):
    if not use_plt:
        sl.warning(content)
    else:
        print(content)


def show_prepare_data_one(use_plt, src_timestamps, src_values, train_timestamps, train_values, test_timestamps,
                          test_values):
    """
    展示准备数据过程
    Args:
        use_plt: 展示方式
        src_timestamps: 原始时间戳
        src_values: 原始值
        train_timestamps:训练数据时间戳
        train_values: 训练数据值
        test_timestamps: 测试时间戳
        test_values: 测试值
    """
    if use_plt:
        plt.prepare_data_one(src_timestamps, src_values, train_timestamps, train_values, test_timestamps,
                             test_values)
    else:
        sl.prepare_data_one(train_timestamps, train_values, test_timestamps, test_values)
