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


# pip install matplotlib seaborn plotly

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


def sentiment_trend_chart(df, date_col='reviews.date', sentiment_col='Sentiment'):
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

# In[13]:


import re
from collections import Counter
from math import log

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves
im ive id ill it's i'm i've i'd i'll you're you've that's there's he's she's we're they're let's
also get got getting go going one two thing things much many really even still well now since though yet etc lot lots
make made may might must need want every ever us bit way anything something everything someone
product products item items amazon bought buy buying purchase purchased order ordered review reviews
""".split())
NEGATIONS = {"no", "not", "never", "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't",
             "isn't", "wasn't", "aren't", "weren't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't"}


def _review_terms(text):
    # Stopwords are dropped before pairing, so "easy to use" becomes "easy use";
    # negations are kept inside phrases so "doesn't work" survives.
    terms = set()
    for sentence in re.split(r"[.!?;\n]+", str(text).lower().replace("\u2019", "'")):
        words = [w.strip("'") for w in re.findall(r"[a-z][a-z']*", sentence)]
        words = [w for w in words if len(w) > 1 and w not in STOPWORDS]
        terms.update(w for w in words if w not in NEGATIONS)
        terms.update(f"{a} {b}" for a, b in zip(words, words[1:]) if b not in NEGATIONS)
    return terms


def sentiment_keywords(df, text_col, sentiment_col='Sentiment', top_n=10):
    """Rank terms by how many reviews of a sentiment mention them, weighted by how much more
    often they appear there than in the other sentiments."""
    # Scraped review datasets often repeat the same review, which would flood the counts with its phrases
    unique = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    doc_terms = unique[text_col].apply(_review_terms)
    labels = unique[sentiment_col]
    results = {}
    for sentiment in labels.unique():
        mask = labels == sentiment
        n_in, n_out = mask.sum(), (~mask).sum()
        c_in = Counter(t for terms in doc_terms[mask] for t in terms)
        c_out = Counter(t for terms in doc_terms[~mask] for t in terms)
        min_count = 2 if n_in >= 5 else 1

        scored = []
        for term, count in c_in.items():
            if count < min_count:
                continue
            if n_out == 0:
                score = count
            else:
                ratio = ((count + 0.5) / (n_in + 1)) / ((c_out[term] + 0.5) / (n_out + 1))
                score = count * log(ratio)
            if score > 0:
                # Boost phrases so "battery life" outranks "battery"
                scored.append((score * (1.5 if " " in term else 1), term, count))
        scored.sort(reverse=True)

        picked = []
        for _, term, count in scored:
            words = set(term.split())
            if any(words <= set(p.split()) or set(p.split()) <= words for p, _ in picked):
                continue
            picked.append((term, count))
            if len(picked) == top_n:
                break
        results[sentiment] = picked
    return results
