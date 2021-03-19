import show_photo_plt as sp
import data

from donut.demo.train_prediction import score

base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
    data.prepare_data("../../sample_data/1.csv")
sp.prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                    train_missing, test_missing)
test_score = score(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
sp.show_test_score(test_timestamp, test_values, test_score)
