# ml_project
# 🎥 YouTube Video Popularity Predictor 📈


**Predict whether a YouTube video will go viral — before it's uploaded.**  
This project uses **Machine Learning** and **Natural Language Processing (NLP)** to estimate a video's **like/view ratio** based on its metadata — including title, tags, category, and duration.

---

## 🔧 Tech Stack

| Layer         | Tools & Libraries                                                                 |
|---------------|------------------------------------------------------------------------------------|
| Language      | Python 3.11+                                                                       |
| ML Algorithm  | `GradientBoostingRegressor` via `scikit-learn`                                     |
| NLP Pipeline  | `TextBlob`, `TfidfVectorizer`                                                      |
| Backend API   | `FastAPI` with `Uvicorn`                                                           |
| Frontend UI   | `Streamlit`                                                                        |
| Dataset       | [YouTube Trending Videos Dataset (Kaggle)](https://www.kaggle.com/datasets/datasnaek/youtube-new) |

---

## 📁 Project Structure

```
ml_project/
├── backend/                     ← All ML, preprocessing, and API code
│   ├── main.py                  ← FastAPI server with /predict, /top-features, /stats
│   ├── train_model.py           ← Trains ML model & saves .pkl files
│   ├── data_preprocessing.py    ← Cleans dataset, encodes text, builds features
│   ├── inference_utils.py       ← Utility used to vectorize and align inputs for predictions
│   └── models/
│       ├── like_ratio_model.pkl       ← Trained Gradient Boosting model
│       └── tfidf_vectorizer.pkl       ← TF-IDF vectorizer used in training
├── dataset/
│   ├── Trending videos on youtube dataset.csv   ← Main training dataset
│   ├── sample_youtube_test_1.csv                ← Test sample 1
│   ├── sample_youtube_test_2.csv                ← Test sample 2
│   └── sample_youtube_test_3.csv                ← Test sample 3
├── frontend/
│   ├── app.py                  ← 🎯 Streamlit UI for interactive predictions
│   └── requirements.txt        ← Streamlit + frontend dependencies
└── README.md                   ← You're reading it!
```

---

## 📊 Dataset Details

- **Source**: Kaggle – YouTube Trending Videos  
- **Target Variable**: like/view ratio  
- **Features Used**:
  - `title`: Video title
  - `tags`: Comma-separated tags
  - `category`: Video category
  - `duration`: Length in seconds
  - `publish_time`: Timestamp (used to extract publish time context)
  - `likes` & `views`: Used to calculate target variable

---

## 🧠 ML Pipeline

### 🔹 Text + Metadata Preprocessing
- Merge `title`, `tags`, and `category`
- Clean using lowercase, punctuation removal, tokenization
- Vectorized using **TF-IDF**

### 🔹 Feature Engineering
- Combine vectorized text with numerical features like `duration`

### 🔹 Model Training
- Model: `GradientBoostingRegressor`
- Evaluation Metrics: `MAE`, `R²`
- Outputs:
  - `like_ratio_model.pkl` — trained model
  - `tfidf_vectorizer.pkl` — text vectorizer

---

## 🌐 Backend API (FastAPI)

| Method | Endpoint         | Description                                  |
|--------|------------------|----------------------------------------------|
| GET    | `/`              | Serves HTML (or welcome response)            |
| POST   | `/predict`       | Predict like/view ratio from metadata input  |
| GET    | `/top-features`  | Return top-weighted TF-IDF features          |
| GET    | `/stats`         | Return model info and evaluation metrics     |

---

## 🖥️ Frontend UI (Streamlit)

Launch an interactive prediction UI using Streamlit:

```bash
cd frontend
streamlit run app.py
```

Users can:
- Enter title, tags, category, etc.
- Get real-time predicted like/view ratio
- View results in a clean dashboard layout

---

## 📂 Sample Input (POST `/predict`)

```json
{
  "title": "Master Python in 10 Minutes",
  "tags": "python, tutorial, beginner",
  "category": "Education",
  "duration": 420,
  "publish_time": "2024-06-10T14:00:00"
}
```

---

## 🧪 Sample Output

```json
{
  "predicted_like_ratio": 0.8347,
  "message": "✅ Prediction successful"
}
```

---

## 🚀 Running the Project Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/HamnaKaleem-Scripts/ml_project
cd ml_project
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
```

### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (if not already trained)

```bash
python backend/train_model.py
```

### 5️⃣ Start the FastAPI Server

```bash
uvicorn backend.main:app --reload
```

Open API documentation:
```
http://127.0.0.1:8000/docs
```

### 6️⃣ Run the Streamlit App

```bash
cd frontend
streamlit run app.py
```

> Built with 💙 and Python by 3 developers passionate about making data-driven predictions that matter.



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
