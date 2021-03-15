import streamlit as st
import data
import numpy as np

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

if st.button('多图'):
    base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing = \
        data.prepare_data("test.csv", 1)
if st.button('单图'):
    base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing = \
        data.prepare_data("test.csv", 0)

