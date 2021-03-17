import streamlit as st
import data

from donut.demo.show_photo_sl import prepare_data_one, prepare_data_two
from donut.demo.donut_model import get_model
from donut.demo.train_prediction import train_prediction

st.title('Donut')
st.markdown("- 准备数据")

base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values, train_missing, test_missing, train_labels, test_labels, mean, std = \
    data.prepare_data("donut/1.csv")
prepare_data_one(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                 train_missing, test_missing)
prepare_data_two(base_timestamp, base_values, train_timestamp, train_values, test_timestamp, test_values,
                 train_missing, test_missing)

st.text('mean:' + str(mean) + '     std:' + str(std))

model, model_vs = get_model()
test_score = train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std)
