import streamlit as st
import pandas as pd
import csv

# 调取operating system模块
with open("../../sample_data/1.csv", 'r') as f:
    base_timestamp = []
    base_values = []
    # 默认无标签
    base_labels = []
    reader = csv.reader(f)
    next(reader)
    for i in reader:
        base_timestamp.append(int(i[0]))
        base_values.append(float(i[1]))
        base_labels.append(int(i[2]))
chart_data = pd.DataFrame(base_values, index=base_timestamp, columns=['original csv_data'])
st.line_chart(chart_data)
