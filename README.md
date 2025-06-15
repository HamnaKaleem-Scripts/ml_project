# ml_project
# 🎥 YouTube Video Popularity Predictor 📈

Predict the popularity of YouTube videos before they go viral — or flop.  
Using **Machine Learning** and **Natural Language Processing**, we predict the **like/view ratio** of videos based solely on metadata (title, tags, duration, etc.).

---

## 🔧 Tech Stack

- **Language**: Python 3.11+
- **Machine Learning**: scikit-learn, GradientBoostingRegressor
- **NLP**: TextBlob, TF-IDF (via scikit-learn)
- **API Backend**: FastAPI + Uvicorn
- **Data**: YouTube Trending Videos Dataset (Kaggle)

---

## 📁 Project Structure & File Descriptions

```
ml_project/
├── backend/                     ← All ML and API code lives here
│   ├── main.py                  ← FastAPI server with /predict, /top-features, /stats routes
│   ├── train_model.py           ← Trains ML model & saves it as .pkl files
│   ├── data_preprocessing.py    ← Cleans dataset: removes noise, encodes text, extracts features
│   └── models/                  ← Serialized trained models
│       ├── like_ratio_model.pkl       ← Trained ML model (Gradient Boosting)
│       └── tfidf_vectorizer.pkl       ← TF-IDF vectorizer used in training
├── dataset/
│   └── Trending videos on youtube dataset.csv ← Raw dataset used for training
├── requirements.txt             ← Python package dependencies
└── README.md                    ← You're reading it!
```

---

## 📊 Dataset Details

- **Source**: [Kaggle – YouTube Trending Videos](https://www.kaggle.com/datasets)
- **Columns Used**:
  - `title`: Video title
  - `tags`: Comma-separated list of tags
  - `category`: Content category
  - `duration`: Video length in seconds
  - `publish_time`: Timestamp
  - `likes`, `views`: Used to calculate like/view ratio

---

## 🧠 ML/NLP Pipeline

1. **Text Preprocessing**:
   - Combine `title`, `tags`, and `category`
   - Lowercase, tokenize, remove punctuation
   - Convert to TF-IDF vector

2. **Feature Engineering**:
   - Merge TF-IDF vector with numerical features (`duration`, etc.)

3. **Model Training**:
   - Model: `GradientBoostingRegressor`
   - Output: **like/view ratio**
   - Evaluation: MAE, R² Score

4. **Model Saving**:
   - Trained model saved as `like_ratio_model.pkl`
   - TF-IDF vectorizer saved as `tfidf_vectorizer.pkl`

---

## 🌐 FastAPI Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/`              | Welcome message (test if server is live) |
| POST   | `/predict`       | Predict like/view ratio from metadata     |
| GET    | `/top-features`  | Return top weighted features (TF-IDF)     |
| GET    | `/stats`         | Return model details & metrics            |

---

## 📂 Example Input for `/predict`

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

## 🚀 How to Run the Project Locally

### 1. Clone the Repo

```bash
git clone https://github.com/HamnaKaleem-Scripts/ml_project
cd ml_project
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Train the Model (if `.pkl` not included)

```bash
python backend/train_model.py
```

### 5. Run the API Server

```bash
uvicorn backend.main:app --reload
```

Go to:  
📍 http://127.0.0.1:8000/docs

---

## 🧪 Example API Response

```json
{
  "predicted_like_ratio": 0.8347,
  "message": "✅ Prediction successful"
}
```

---

## 👥 Team Members & Contributions

| Name             | Role                                 |
|------------------|--------------------------------------|
| **Hamna**        | Backend API Developer (FastAPI), Integration, GitHub management |
| **Muqadsa**      | Data Cleaning, Preprocessing, Model Training |
| **Tehreem**      | Frontend Development (React/HTML/CSS), UI Integration with API |

---

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
