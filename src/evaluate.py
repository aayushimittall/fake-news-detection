import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocess import preprocess_dataframe

test_df = pd.read_csv("data/test.csv")
test_df = preprocess_dataframe(test_df)

X_test = test_df["content"]
y_test = test_df["label"]

tfidf = joblib.load("models/tfidf.pkl")
model = joblib.load("models/svm_model.pkl")

X_test_tfidf = tfidf.transform(X_test)

y_pred = model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))