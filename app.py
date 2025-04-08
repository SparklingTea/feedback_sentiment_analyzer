#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# In[4]:


# pwd

# In[5]:


# pip install streamlit pandas 

# In[6]:


import streamlit as st
import pandas as pd
from analyzer import get_sentiment
from visualizer import sentiment_pie_chart,sentiment_trend_chart

st.title("Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Sentiment'] = df['reviews.text'].astype(str).apply(get_sentiment)
    st.dataframe(df.head())
    st.subheader("📈 Sentiment Analysis Results")

    # Pie chart
    st.subheader("🥧 Pie Chart of Sentiment")
    st.pyplot(sentiment_pie_chart(df))

    # Time-based trend chart
    if date_col != "None":
        st.subheader("📅 Trend of Sentiment Over Time")
        fig = sentiment_trend_chart(df, date_col)
        st.plotly_chart(fig)

# In[ ]:



