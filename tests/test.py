import numpy as np
# 机器学习框架搭建
import tensorflow as tf


def test_percent():
    test_template = np.array([1, 2, 3, 4, 5])
    test_labels = np.array([0, 1, 0, 1, 0])
    test_score = np.array([1, 2, 3, 4, 5])
    labels_index = np.where(test_labels == 1)
    print(labels_index)
    labels_score = test_score[labels_index]
    labels_score_max = np.max(labels_score)
    labels_score_min = np.min(labels_score)
    catch_index = np.where(test_score > labels_score_min)
    catch_num = np.size(catch_index)
    labels_num = np.size(labels_index)
    # 准确度
    accuracy = labels_num / catch_num
    print("{:.2%}".format(accuracy))
    if accuracy < 1:
        a = set(catch_index[0].tolist())
        b = set(labels_index[0].tolist())
        special_anomaly_index = list(a.difference(b))
        special_anomaly_t = test_template[special_anomaly_index]
        special_anomaly_s = test_score[special_anomaly_index]
        print(special_anomaly_t, special_anomaly_t)

def test_reduce_mean():
    x = [[1., 2.], [3., 4.]]
    mean1 = tf.reduce_mean(x,-1)
    mean2= tf.reduce_mean(x,1)
    print(mean1,mean2)