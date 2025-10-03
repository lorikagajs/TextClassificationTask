import joblib
from preprocess import load_data, prepare_data, vectorize_text
from sklearn.metrics import classification_report

# Load test data
df = load_data()
X_train, X_test, y_train, y_test = prepare_data(df)

# Load saved models
log_reg = joblib.load("models/log_reg_model.pkl")
nb = joblib.load("models/naive_bayes_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Vectorize test data
X_test_vec = vectorizer.transform(X_test)

# Evaluate models
y_pred_log = log_reg.predict(X_test_vec)
y_pred_nb = nb.predict(X_test_vec)

print("--- Logistic Regression ---")
print(classification_report(y_test, y_pred_log, target_names=y_test.columns))

print("--- Naive Bayes ---")
print(classification_report(y_test, y_pred_nb, target_names=y_test.columns))
