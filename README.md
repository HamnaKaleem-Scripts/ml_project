# ml_project
# 🎥 YouTube Video Popularity Predictor 📈

**Predict the popularity of YouTube videos before they go viral — or flop.**  
Using **Machine Learning** and **Natural Language Processing**, this project predicts the **like/view ratio** of YouTube videos based solely on metadata (title, tags, duration, etc.).

---

## 🔧 Tech Stack

- **Language**: Python 3.11+
- **Machine Learning**: `scikit-learn`, `GradientBoostingRegressor`
- **NLP**: `TextBlob`, `TF-IDF` via `scikit-learn`
- **API Backend**: FastAPI + Uvicorn
- **Web UI**: 🖥️ Streamlit
- **Data Source**: [YouTube Trending Videos Dataset (Kaggle)](https://www.kaggle.com/datasets/datasnaek/youtube-new)

---

## 📁 Project Structure

```
ml_project/
├── backend/                     ← All ML, preprocessing, and API code
│   ├── main.py                  ← FastAPI server with /predict, /top-features, /stats
│   ├── train_model.py           ← Trains ML model & saves .pkl files
│   ├── data_preprocessing.py    ← Cleans dataset, encodes text, builds features
│   └── models/                  
│       ├── like_ratio_model.pkl       ← Trained GradientBoosting model
│       └── tfidf_vectorizer.pkl       ← Saved TF-IDF vectorizer
├── dataset/
│   └── Trending videos on youtube dataset.csv
├── frontend/
│   └── streamlit_app.py         ← 🎯 Streamlit UI for predictions
├── requirements.txt             ← Python dependencies
└── README.md                    ← You're reading it!
```

---

## 📊 Dataset Overview

- **Source**: Kaggle — YouTube Trending Videos  
- **Target Variable**: `like/view ratio`  
- **Features Used**:
  - `title` – Video title  
  - `tags` – Comma-separated tags  
  - `category` – Content type  
  - `duration` – Video length in seconds  
  - `publish_time` – Datetime  
  - `likes`, `views` – Used to calculate ratio

---

## 🧠 ML/NLP Pipeline

### Text Preprocessing:
- Combine `title`, `tags`, and `category`
- Lowercase, tokenize, remove punctuation
- Convert to TF-IDF vectors

### Feature Engineering:
- Merge TF-IDF with numerical features (e.g., duration)

### Model Training:
- Model: `GradientBoostingRegressor`
- Output: like/view ratio
- Evaluation: `MAE`, `R² Score`

### Model Saving:
- `like_ratio_model.pkl` — trained regressor
- `tfidf_vectorizer.pkl` — saved vectorizer for future input

---

## 🌐 FastAPI Endpoints

| Method | Endpoint         | Description                                 |
|--------|------------------|---------------------------------------------|
| GET    | `/`              | Health check or frontend render             |
| POST   | `/predict`       | Predict like/view ratio from metadata       |
| GET    | `/top-features`  | Return top weighted TF-IDF features         |
| GET    | `/stats`         | Return model details and performance scores |

---

## 🖥️ Streamlit Frontend

You can also use our simple **Streamlit-based interface** instead of Swagger:

```bash
streamlit run frontend/streamlit_app.py
```

This opens a form where you enter title, tags, category, duration, etc., and see predictions instantly with a progress bar ✅

---

## 📂 Example Input to `/predict`

```json
{
  "title": "Learn Python in 10 Minutes",
  "tags": "python, tutorial, beginner",
  "category": "Education",
  "duration": 300,
  "publish_time": "2024-06-01T14:00:00"
}
```

---

## 🧪 Sample API Response

```json
{
  "predicted_like_ratio": 0.8347,
  "message": "✅ Prediction successful"
}
```

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repo

```bash
git clone https://github.com/HamnaKaleem-Scripts/ml_project
cd ml_project
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (if not already trained)

```bash
python backend/train_model.py
```

### 5. Run FastAPI Server

```bash
uvicorn backend.main:app --reload
```

Open API Docs:
```
http://127.0.0.1:8000/docs
```

### 6. Run Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```



> ✨ Built with love, Python, and a shared passion for tech-powered insights.



## 📦 Future Improvements

- 🖼️ Frontend dashboard (form + charts)
- 🌍 Deploy API using Render or Heroku
- 🔐 API key authentication
- 📈 Real-time trending data scraper
- 🎯 Convert from regression to classification (Low/Medium/High popularity)

---

## 📜 License

MIT License © 2025  
Team BSDSF22A — Data Science for the Win 🧠
