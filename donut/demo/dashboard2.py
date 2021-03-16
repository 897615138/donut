import data
import numpy as np

import tensorflow as tf
from donut import DonutTrainer, DonutPredictor
from donut.demo.donut_model import get_model
from donut.demo.train_prediction import train_prediction

# st.title('Donut')
# st.markdown("- 准备数据")

# if st.button('多图'):
#     base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
#         data.prepare_data("test.csv", 1)
#     print(mean)
# if st.button('单图'):
#     base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
#         data.prepare_data("test.csv", 0)
# st.text('mean:' + str(mean) + '     std:' + str(std))
base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
    data.prepare_data("1.csv", 1)

model, model_vs = get_model()
test_score =train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)

# model, model_vs = get_model()
# trainer = DonutTrainer(model=model, model_vs=model_vs)
# predictor = DonutPredictor(model)
# with tf.Session().as_default():
#     trainer.fit(train_values, train_labels, train_missing, mean, std)
#     test_score = predictor.get_score(test_values, test_missing)
# test_score = get_score(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
# test_score = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
print(1)