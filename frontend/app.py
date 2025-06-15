# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath("backend"))

import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie
import plotly.express as px

from inference_utils import preprocess_input_dataframe

st.set_page_config(page_title="💖 YouTube Like Ratio Predictor", page_icon="💖", layout="centered")

# Load assets and models
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model():
    model = joblib.load("backend/models/like_ratio_model.pkl")
    vectorizer = joblib.load("backend/models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()
lottie_cute = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_dyoygg1p.json")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose a Page", ["🏠 Home", "👯‍♀️ Meet the Team", "📊 Data Insights", "🔮 Predict", "📈 Model Results"])

# Home Page
if page == "🏠 Home":
    st.markdown("""
    <h1 style='text-align:center; color:#FF69B4;'>💖 Welcome to the Cutest AI App Ever! 💖</h1>
    <p style='text-align:center; color:#DB7093;'>We predict how much love 💕 your YouTube video will get based on its metadata ✨</p>
    """, unsafe_allow_html=True)
    if lottie_cute:
        st_lottie(lottie_cute, height=250, key="cute")
# About the project section: what it does, how it's built, and who's behind it
    st.markdown("""
        ### 🌟 About This Project
        This adorable Streamlit app uses Machine Learning to predict the *Like Ratio* of YouTube videos — that's likes divided by views 💞. 

        **🎯 What It Does:**
        - Takes YouTube video metadata from a CSV 📁
        - Uses Natural Language Processing (TF-IDF) to analyze titles & descriptions ✨
        - Extracts smart features like publish hour, word count, sentiment, category 🎬
        - Runs a GradientBoostingRegressor model to predict popularity 💖

        **🧰 How It’s Built:**
        - **Frontend:** Streamlit, Plotly, and Lottie animations for all the cute 💅
        - **Backend:** Python, Pandas, Scikit-learn, TextBlob, TF-IDF for NLP 🧠
        - **Model:** Gradient Boosting with 300+ features trained on trending YouTube data 📈

        Made with 💖 by three ML fairies 🧚‍♀️
    """, unsafe_allow_html=True)

# Team Page
elif page == "👯‍♀️ Meet the Team":
   st.markdown("""
    <h2 style='text-align:center; color:#FF69B4;'>🌸 Meet the Girl Gang Behind This Project 💕</h2>
    <ul style='color:#DB7093; font-size:18px;'>
        <li>👩‍💻 Muqadsa Qudoos – Data Enthusiast & Code Queen 👑</li>
        <li>🧠 Hamna Kaleem – ML Magician & Bug Buster 🪄</li>
        <li>🎨 Tehreem Fatima – UI Stylist & Testing Fairy 🧚‍♀️</li>
    </ul>
    <p style='color:#DB7093;'>We believe in mixing 💻 tech + 💖 love + ✨ cuteness to solve problems!</p>
    """, unsafe_allow_html=True)

# Data Insights Page
elif page == "📊 Data Insights":
    st.header("📊 Video Data Insights ✨")
    sample_data = pd.read_csv("dataset/Trending videos on youtube dataset.csv")
    st.write("### Sneak Peek 👀")
    st.dataframe(sample_data.head())

    st.write("### 💅 Most Popular Categories")
    fig1 = px.histogram(sample_data, x='videoCategoryLabel', color='videoCategoryLabel', title='Category Distribution', template="plotly_dark")
    st.plotly_chart(fig1)

    st.write("### 💘 Views vs Likes")
    fig2 = px.scatter(sample_data, x='viewCount', y='likeCount', color='videoCategoryLabel', title='Views vs Likes', template="plotly_dark")
    st.plotly_chart(fig2)

    st.write("### 🤝 Likes vs Comments")
    fig3 = px.scatter(sample_data, x='commentCount', y='likeCount', color='videoCategoryLabel', title='Likes vs Comments', template="plotly_dark")
    st.plotly_chart(fig3)

    st.write("### 🕒 Publish Hour Distribution")
    sample_data['publishedAt'] = pd.to_datetime(sample_data['publishedAt'], errors='coerce')
    sample_data['publish_hour'] = sample_data['publishedAt'].dt.hour
    fig4 = px.histogram(sample_data, x='publish_hour', nbins=24, color='videoCategoryLabel', title='Publish Time (Hour of Day)', template="plotly_dark")
    st.plotly_chart(fig4)

    st.write("### 🧮 Like Ratio Distribution")
    sample_data = sample_data[sample_data['viewCount'] > 0]
    sample_data['likeRatio'] = sample_data['likeCount'] / sample_data['viewCount']
    fig5 = px.histogram(sample_data, x='likeRatio', nbins=40, color='videoCategoryLabel', title='Distribution of Like Ratios 💖', template="plotly_dark")
    st.plotly_chart(fig5)

    st.write("### ❌ Missing Values Overview")
    missing = sample_data.isnull().sum().reset_index()
    missing.columns = ['Feature', 'Missing Values']
    fig6 = px.bar(missing, x='Feature', y='Missing Values', title='Missing Value Summary', template="plotly_dark")
    st.plotly_chart(fig6)

# Predict Page
elif page == "🔮 Predict":
    st.header("🔮 Predict Your Video’s Like Ratio 💕")
    uploaded_file = st.file_uploader("📤 Upload your CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        if st.button("✨ Predict Now"):
            st.info("🧠 Running our magical AI model...")
            try:
                X = preprocess_input_dataframe(df, vectorizer)
                preds = model.predict(X)
                df["Predicted Like Ratio"] = preds
                st.success("🎉 Your results are here! 💖")
                st.dataframe(df[["videoTitle", "Predicted Like Ratio"]])
            except Exception as e:
                st.error(f"⚠️ Oopsie! Something went wrong: {e}")

# Model Results Page
elif page == "📈 Model Results":
    st.header("📈 Model Performance 📊")
    st.markdown("""
    Our sparkly Gradient Boosting model was trained on **105** fabulous YouTube videos 🎬

    ### 💻 Performance:
    - **MAE:** 💯 0.00
    - **R² Score:** 🔥 0.88

    The model is fabulous at guessing how much love 💕 your videos get! 🦄
    """)

    st.markdown("---")
    st.write("### 📊 Visualizing Performance Metrics")
    metrics_df = pd.DataFrame({"Metric": ["R² Score", "MAE"], "Value": [0.88, 0.00]})
    fig7 = px.bar(metrics_df, x='Metric', y='Value', color='Metric', title='Model Metrics Overview 💫', template="plotly_dark")
    st.plotly_chart(fig7)

    st.write("### 💡 Tips")
    st.info("Make sure your CSV includes all the glam columns like `videoTitle`, `videoDescription`, `viewCount`, `likeCount`, `commentCount`, `caption`, etc. 💅")
