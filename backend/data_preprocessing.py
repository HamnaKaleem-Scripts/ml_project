import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

nltk.download("stopwords")
stopwords_set = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords_set])
    return text

def preprocess_data(path):
    df = pd.read_csv(path)
    print("✅ Raw data loaded.")
    print("📊 Columns:", df.columns.tolist())

    df = df.dropna(subset=['videoTitle', 'videoDescription', 'viewCount', 'likeCount'])
    df = df[df['viewCount'] > 0]
    df['likeRatio'] = df['likeCount'] / df['viewCount']

    df['clean_title'] = df['videoTitle'].apply(clean_text)
    df['clean_desc'] = df['videoDescription'].apply(clean_text)
    df['combined_text'] = df['clean_title'] + " " + df['clean_desc']

    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['publish_day'] = df['publishedAt'].dt.dayofweek.fillna(0).astype(int)
    df['publish_hour'] = df['publishedAt'].dt.hour.fillna(0).astype(int)

    df['title_word_count'] = df['videoTitle'].apply(lambda x: len(str(x).split()))
    df['desc_word_count'] = df['videoDescription'].apply(lambda x: len(str(x).split()))
    df['title_sentiment'] = df['videoTitle'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['desc_sentiment'] = df['videoDescription'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['like_per_comment'] = df['likeCount'] / (df['commentCount'] + 1)
    df['engagement_ratio'] = (df['likeCount'] + df['dislikeCount'] + df['commentCount']) / df['viewCount']

    df['category_encoded'] = LabelEncoder().fit_transform(df['videoCategoryLabel'].astype(str))
    df['caption_encoded'] = LabelEncoder().fit_transform(df['caption'].astype(str))
    df['definition_encoded'] = LabelEncoder().fit_transform(df['definition'].astype(str))

    tfidf = TfidfVectorizer(max_features=300)
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    final_df = pd.concat([
        df[['durationSec', 'commentCount', 'dislikeCount', 'publish_day', 'publish_hour',
            'title_word_count', 'desc_word_count', 'title_sentiment', 'desc_sentiment',
            'like_per_comment', 'engagement_ratio',
            'category_encoded', 'definition_encoded', 'caption_encoded']],
        tfidf_df
    ], axis=1)

    X = final_df
    y = df['likeRatio']
    X = X.iloc[:len(y)].reset_index(drop=True)
    y = y.iloc[:len(X)].reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Preprocessing complete. Final shape: {X.shape}")
    return (X_train, X_test, y_train, y_test), tfidf

if __name__ == "__main__":
    preprocess_data("dataset/Trending videos on youtube dataset.csv")
