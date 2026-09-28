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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import streamlit as st

# ✅ Load Hugging Face Token securely from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment"
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

# ✅ Single text with retries for rate limits / model loading
def get_sentiment_with_retry(text, retries=4):
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": text[:512]}, timeout=60)
            if response.status_code == 200:
                scores = response.json()
                # The API returns either [{...}, ...] or [[{...}, ...]] depending on the backend
                if scores and isinstance(scores[0], list):
                    scores = scores[0]
                return decode_label(max(scores, key=lambda x: x['score'])['label'])
            if response.status_code not in (429, 503):
                return "Request Failed"
        except (requests.RequestException, ValueError, KeyError, IndexError):
            pass
        time.sleep(2 ** attempt)
    return "Request Failed"

# ✅ DataFrame analyzer: the API takes one text per request, so send several in parallel
def analyze_dataframe(df, text_column, workers=8, progress=None):
    df = df.copy()
    texts = df[text_column].astype(str).tolist()
    labels = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get_sentiment_with_retry, t): i for i, t in enumerate(texts)}
        for done, future in enumerate(as_completed(futures), 1):
            labels[futures[future]] = future.result()
            if progress and (done % 10 == 0 or done == len(texts)):
                progress(done / len(texts))
    df['Sentiment'] = labels
    return df




