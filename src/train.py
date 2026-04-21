import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import preprocess_dataframe
import os

print("✅ Starting training script...")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load datasets
print("✅ Loading CSV files...")
train_df = pd.read_csv("data/train.csv")

# Preprocess text
print("✅ Preprocessing text...")
train_df = preprocess_dataframe(train_df)

X_train = train_df["content"]
y_train = train_df["label"]

# TF-IDF
print("✅ Creating TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Train SVM
print("✅ Training SVM...")
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Save models
print("✅ Saving model files...")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(model, "models/svm_model.pkl")

print("🎉 SUCCESS: .pkl files created in /models")