import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import load_data, prepare_data, vectorize_text, clean_text


def exploratory_analysis(df):
    print("===== Exploratory Data Analysis =====")

    # Basic stats 
    print(f"\nNumber of samples: {len(df)}")

    # Count how many samples per label
    label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    label_counts = df[label_cols].sum().sort_values(ascending=False)
    print("\nSamples per category:")
    print(label_counts)

    # Plot label distribution
    plt.figure(figsize=(6,4))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, dodge=False, palette="viridis", legend=False)
    plt.title("Number of samples per category")
    plt.ylabel("Count")
    plt.xlabel("Emotion")
    plt.show()

    # --- Word frequency ---
    print("\nMost frequent words (in cleaned text):")
    all_words = " ".join(df['clean_text']).split()
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(15)
    for word, freq in most_common:
        print(f"{word}: {freq}")

    # Wordcloud
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Most Frequent Words")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    # Make sure to clean text before analysis
    df["clean_text"] = df["text"].apply(clean_text)

    exploratory_analysis(df)
