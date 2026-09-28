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
from analyzer import analyze_dataframe
from visualizer import sentiment_pie_chart,sentiment_trend_chart,sentiment_keywords
from scraper_agent import run_scraper_agent
from cleaner_agent import clean_reviews

st.title("📊 Feedback Sentiment Analyser")


@st.cache_data(show_spinner=False)
def run_sentiment(df, text_col):
    bar = st.progress(0.0, text="Analysing sentiment...")
    result = analyze_dataframe(df, text_col, progress=lambda f: bar.progress(f, text="Analysing sentiment..."))
    bar.empty()
    return result


source = st.radio("Feedback source", ["Upload a CSV", "Scrape online reviews"], horizontal=True)
df = None

if source == "Upload a CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select the text column:", df.columns)
        date_col = st.selectbox("Select the date column (optional):", ["None"] + list(df.columns))
else:
    with st.form("scrape"):
        request = st.text_input("Product name and/or product links",
                                placeholder="e.g. Sony WH-1000XM5 headphones")
        platforms = st.multiselect("Platforms", ["Amazon", "eBay", "TikTok"], default=["Amazon", "eBay", "TikTok"])
        max_reviews = st.slider("Max reviews per platform", 50, 2000, 300, step=50)
        submitted = st.form_submit_button("Collect feedback")
    st.caption("Scraping runs through Apify and is billed to your Apify account. "
               "Check that your use complies with each platform's terms.")

    if submitted and request and platforms:
        with st.status("Collecting feedback...", expanded=True) as status:
            try:
                raw, summary = run_scraper_agent(request, platforms, max_reviews, log=st.write)
                st.write(summary)
                status.update(label="Cleaning data...")
                st.session_state["scraped"] = clean_reviews(raw, log=st.write)
                status.update(label="Feedback collected", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Collection failed", state="error")
                st.error(str(e))

    if st.session_state.get("scraped") is not None:
        df = st.session_state["scraped"]
        if df.empty:
            st.warning("No feedback was collected. Try adding direct product links.")
            df = None
        else:
            st.write(df["platform"].value_counts().rename("rows"))
            st.download_button("Download cleaned data", df.to_csv(index=False).encode("utf-8-sig"), "cleaned_feedback.csv")
            text_col, date_col = "text", ("date" if df["date"].notna().any() else "None")

if df is not None:
    df = run_sentiment(df, text_col)

    st.subheader("Sentiment Results")
    extra = [c for c in ("platform", "rating") if c in df.columns and c != text_col]
    st.dataframe(df[['Sentiment', text_col] + extra])

    st.download_button("Download Results", df.to_csv(index=False).encode("utf-8-sig"), "sentiment_results.csv")

    # Pie chart
    st.subheader("🥧 Pie Chart of Sentiment")
    st.pyplot(sentiment_pie_chart(df))

    # Keyword panels
    st.subheader("🔑 What Drives Each Sentiment")
    st.caption("Words and phrases most typical of each sentiment. The number is how many reviews mention it.")
    keywords = sentiment_keywords(df, text_col)
    panels = [("Positive", ":green[**😊 Positive**]"),
              ("Neutral", ":gray[**😐 Neutral**]"),
              ("Negative", ":red[**😞 Negative**]")]
    for col, (sentiment, heading) in zip(st.columns(3), panels):
        with col:
            st.markdown(heading)
            terms = keywords.get(sentiment, [])
            if terms:
                st.markdown("\n".join(f"- {term} ({count})" for term, count in terms))
            else:
                st.caption("Not enough reviews")

    # Time-based trend chart
    if date_col != "None":
        st.subheader("📅 Trend of Sentiment Over Time")
        fig = sentiment_trend_chart(df, date_col)
        st.plotly_chart(fig)

# In[ ]:



