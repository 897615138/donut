import streamlit as st
import data
import numpy as np

from donut.demo.donut_model import get_model
from donut.demo.train_prediction import train_prediction

st.title('Donut')
st.markdown("- 准备数据")
base_timestamp = []
base_values = []
train_timestamp = np.array([])
train_values = np.array([])
test_timestamp = np.array([])
test_values = np.array([])
train_missing = np.array([])
test_missing = np.array([])
train_labels = np.array([])
test_labels = np.array([])
mean = 0.0
std = 0.0

if st.button('多图'):
    base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
        data.prepare_data("test.csv", 1)
    print(mean)
if st.button('单图'):
    base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
        data.prepare_data("test.csv", 0)
st.text('mean:' + str(mean) + '     std:' + str(std))
model, model_vs = get_model()
print(model,model_vs)
test_score = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std, model,model_vs)
