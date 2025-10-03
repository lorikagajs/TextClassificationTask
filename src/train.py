import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

from preprocess import load_data, prepare_data, vectorize_text


def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, vectorizer):


    save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    print("Saving models to:", os.path.abspath(save_dir))

    # Logistic Regression wrapped in One-vs-Rest
    log_reg = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    log_reg.fit(X_train_vec, y_train)
    y_pred_log = log_reg.predict(X_test_vec)
    print("\n--- Logistic Regression (OvR) ---")
    print(classification_report(y_test, y_pred_log, target_names=y_test.columns))

    # Save Logistic Regression + vectorizer
    joblib.dump(log_reg, os.path.join(save_dir, "log_reg_model.pkl"))
    joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))

    # Naive Bayes wrapped in One-vs-Rest 
    nb = OneVsRestClassifier(MultinomialNB())
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    print("\n--- Naive Bayes (OvR) ---")
    print(classification_report(y_test, y_pred_nb, target_names=y_test.columns))

    # Save Naive Bayes
    joblib.dump(nb, os.path.join(save_dir, "naive_bayes_model.pkl"))


if __name__ == "__main__":
    # Load & preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # Train & evaluate models
    train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, vectorizer)
