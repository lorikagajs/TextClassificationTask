import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


def load_data(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    # Clean the text column
    df['clean_text'] = df['text'].apply(clean_text)

    # Multi-label target
    y = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]
    X = df['clean_text']

    return X, y


def vectorize_text(X_train, X_test, max_features=5000):
 
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def train_test_split_data(X, y, test_size=0.2, random_state=42):
   
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Quick test
    X, y = load_data(r"C:\Users\User\Documents\GitHub\TextClassificationTask\data\track-a.csv")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    print("Preprocessing complete!")
    print("Sample vector shape:", X_train_vec.shape)
