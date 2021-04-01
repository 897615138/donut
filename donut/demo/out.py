import show_plt as plt
import show_sl as sl


def show_test_score(use_plt, test_timestamps, test_values, test_scores):
    if use_plt:
        plt.show_test_score(test_timestamps, test_values, test_scores)
    else:
        sl.show_test_score(test_timestamps, test_values, test_scores)


def show_line_chart(use_plt, x, y, name):
    if use_plt:
        plt.line_chart(x, y, name)
    else:
        sl.line_chart(x, y, name)


def print_text(use_plt, content):
    if use_plt:
        print(content)
    else:
        sl.text(content)
def bar_chart(use_plt,chart_data):
    if use_plt:
        pass
    else:
        sl.bar_chart(chart_data)

def show_prepare_data_one(use_plt, src_timestamps, src_values, train_timestamps, train_values, test_timestamps,
                          test_values):
    if use_plt:
        plt.prepare_data_one(src_timestamps, src_values, train_timestamps, train_values, test_timestamps,
                             test_values)
    else:
        sl.prepare_data_one(train_timestamps, train_values, test_timestamps, test_values)
