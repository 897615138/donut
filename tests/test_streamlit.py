import pandas as pd
import streamlit as st

# 调取operating system模块

chart_data = pd.DataFrame(
    [[20, 30, 50]],
    columns=['a', 'b', 'c']
    )
st.bar_chart(chart_data)