import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Clean text: lowercase, remove non-letters, stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)


def load_data(csv_path="C:/Users/User/Documents/GitHub/TextClassificationTask/data/track-a.csv"):
    """Load dataset and return dataframe."""
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        raise FileNotFoundError(f"{csv_path} not found or empty.")
    return df


def prepare_data(df, test_size=0.2):
    """Clean text, split into train/test, and return everything needed."""
    # Labels (multi-label target)
    y = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, max_features=5000):
    """Turn text into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def preprocess_text(text):
 
    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    print("Preprocessing complete!")
    print("Sample vector shape:", X_train_vec.shape)
