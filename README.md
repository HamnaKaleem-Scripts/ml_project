# 💖 YouTube Like Ratio Predictor

A cute, ML-powered Streamlit app that predicts how much *love* (like ratio) your YouTube videos will receive, based on metadata, natural language analysis, and regression modeling.

---

## 🎯 What It Does

- 📤 Upload a CSV with YouTube video metadata
- 💬 Analyze titles & descriptions using TF-IDF NLP
- 🧠 Extract smart features (publish hour, word count, sentiment, etc.)
- 🤖 Predict the **like ratio** using a trained Gradient Boosting model
- 📈 Visualize insights: category, views, likes, publish time, and more

---

## 🧰 Tech Stack

| Layer       | Tools Used |
|-------------|------------|
| Frontend    | Streamlit + Plotly + Lottie |
| Backend     | Python + Pandas + Scikit-learn |
| NLP         | TextBlob + TF-IDF Vectorizer |
| Model       | GradientBoostingRegressor |
| Dataset     | Kaggle's Trending YouTube Videos (modified) |

---

## 🏗️ Project Structure

```bash
ml_project/
│
├── backend/
│   ├── inference_utils.py       # Preprocessing logic
│   ├── models/
│   │   ├── like_ratio_model.pkl
│   │   └── tfidf_vectorizer.pkl
│
├── dataset/
│   ├── Trending videos on youtube dataset.csv
│   └── test_input_sample.csv
│
├── frontend/
│   └── app.py                   # Main Streamlit app
│
├── .gitignore
├── requirements.txt
└── README.md
