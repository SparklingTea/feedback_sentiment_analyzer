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

# In[2]:


pip install matplotlib seaborn plotly

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# In[4]:


# Sentiment bar chart
def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel', ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    return fig

# In[5]:


# # bar chart - lack of perc of total x

# sentiment_counts = df['Sentiment'].value_counts()

# plt.figure(figsize=(6, 4))
# sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel')
# plt.title("Sentiment Distribution")
# plt.ylabel("Number of Reviews")
# plt.xlabel("Sentiment")
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# In[6]:


# # pie chart - good

# plt.figure(figsize=(6, 6))
# plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title("Sentiment Breakdown (Pie Chart)")
# plt.axis('equal')
# plt.show()

# In[7]:


def sentiment_pie_chart(df, sentiment_col='Sentiment'):
    sentiment_counts = df[sentiment_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sentiment_counts.values,
           labels=sentiment_counts.index,
           autopct='%1.1f%%',
           startangle=140,
           colors=sns.color_palette('pastel')[:len(sentiment_counts)])
    ax.set_title("Sentiment Breakdown (Pie Chart)")
    ax.axis('equal')
    return fig

# In[8]:


# from wordcloud import WordCloud

# for sentiment in sentiment_counts.index:
#     text = " ".join(df[df['Sentiment'] == sentiment]['reviews.text'])
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(f"Most Common Words in {sentiment} Reviews")
#     plt.axis('off')
#     plt.show()

# In[9]:


def generate_wordcloud(df, sentiment_col, text_col, sentiment):
    text = " ".join(df[df[sentiment_col] == sentiment][text_col].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Word Cloud - {sentiment} Reviews")
    return fig

# In[10]:


# fig = px.pie(
#     df,
#     names='Sentiment',
#     title='Interactive Sentiment Breakdown',
#     hole=0.3  # for donut-style
# )
# fig.show()

# In[11]:


# # Time-based trends

# df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
# df['month'] = df['reviews.date'].dt.to_period('M')

# monthly_sentiment = df.groupby(['month', 'Sentiment']).size().unstack().fillna(0)

# monthly_sentiment.plot(figsize=(10, 6))
# plt.title("Sentiment Trends Over Time")
# plt.xlabel("Month")
# plt.ylabel("Review Count")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# In[12]:


def sentiment_trend_chart(df, date_col, sentiment_col='Sentiment'):
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    
    # Convert to datetime and round to month (or day for fine grain)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['TimeGroup'] = df[date_col].dt.to_period('M').astype(str)

    # Group by time and sentiment
    trend_df = df.groupby(['TimeGroup', sentiment_col]).size().reset_index(name='Count')

    fig = px.line(trend_df, x='TimeGroup', y='Count', color=sentiment_col,
                  title='Sentiment Trends Over Time',
                  labels={'TimeGroup': 'Month', 'Count': 'Review Count'})
    return fig
