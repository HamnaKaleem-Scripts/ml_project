# backend/train_model.py

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_preprocessing import preprocess_data

# Load preprocessed data
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data("dataset/Trending videos on youtube dataset.csv")

# Train model
# ✅ Model training
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# ✅ Evaluation
y_pred = model.predict(X_test)
print("✅ Model trained with GradientBoostingRegressor")
print(f"📉 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")

# ✅ Save model and vectorizer
import os
import joblib

os.makedirs("backend/models", exist_ok=True)
joblib.dump(model, "backend/models/like_ratio_model.pkl")
joblib.dump(vectorizer, "backend/models/tfidf_vectorizer.pkl")
