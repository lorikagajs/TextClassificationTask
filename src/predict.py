import os
import joblib
import pandas as pd
from preprocess import preprocess_text  # assuming you have a function for cleaning text

def load_models():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    log_reg = joblib.load(os.path.join(base_dir, "log_reg_model.pkl"))
    nb = joblib.load(os.path.join(base_dir, "naive_bayes_model.pkl"))
    vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.pkl"))
    return log_reg, nb, vectorizer

def predict_emotions(model, vectorizer, text):
  
    # Preprocess the input text (same as training)
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_vec = vectorizer.transform([cleaned_text])
    
    # Predict
    y_pred = model.predict(text_vec)
    
    # Convert prediction to dictionary
    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    return dict(zip(emotions, y_pred[0]))

if __name__ == "__main__":
    # Load models and vectorizer
    log_reg, nb, vectorizer = load_models()

    print("Text Classification Prediction")
    print("Enter a sentence to predict its emotions (type 'exit' to quit).")

    while True:
        text = input("\nYour sentence: ")
        if text.lower() == "exit":
            break

        # Predict using Logistic Regression
        log_pred = predict_emotions(log_reg, vectorizer, text)
        # Predict using Naive Bayes
        nb_pred = predict_emotions(nb, vectorizer, text)

        print("\n--- Logistic Regression Prediction ---")
        for emotion, val in log_pred.items():
            print(f"{emotion}: {val}")

        print("\n--- Naive Bayes Prediction ---")
        for emotion, val in nb_pred.items():
            print(f"{emotion}: {val}")
