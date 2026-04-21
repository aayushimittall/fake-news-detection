import joblib
from preprocess import clean_text

tfidf = joblib.load("models/tfidf.pkl")
model = joblib.load("models/svm_model.pkl")

def predict_news(title, text):
    content = clean_text(title + " " + text)
    vector = tfidf.transform([content])
    prediction = model.predict(vector)[0]
    return "Real" if prediction == 1 else "Fake"

# Example
print(
    predict_news(
        "Government announces new policy",
        "The ministry stated that the policy will improve economy."
    )
)