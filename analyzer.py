#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# In[ ]:


# pip install pandas transformers torch

# Import a sample of large dataset of consumer reviews for Amazon products like the Kindle, Fire TV Stick sourced from https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

# In[ ]:

# call the API

import requests
import streamlit as st

# ✅ Load Hugging Face Token securely from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ✅ Decode raw Hugging Face labels to readable ones
def decode_label(label):
    return {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }.get(label, "Unknown")

# ✅ Single sentence sentiment analysis
def get_sentiment(text):
    try:
        payload = {"inputs": text[:512]}
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            result = response.json()[0]
            top = max(result, key=lambda x: x['score'])
            return decode_label(top['label'])
        else:
            return "Request Failed"
    except Exception as e:
        return "Error"

# ✅ Batch DataFrame analyzer: takes df and text column name
def analyze_dataframe(df, text_column):
    df = df.copy()
    df['Sentiment'] = df[text_column].astype(str).apply(get_sentiment)
    return df




