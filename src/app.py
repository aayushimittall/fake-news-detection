import streamlit as st
import joblib
from preprocess import clean_text
from pathlib import Path

st.title("Fake News Detection System")

# Get absolute path to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

try:
    tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
    model = joblib.load(MODEL_DIR / "svm_model.pkl")
except Exception as e:
    st.error("❌ Model files not found. Please train the model first.")
    st.exception(e)
    st.stop()

title = st.text_input("News Title")
body = st.text_area("News Content")

if st.button("Check"):
    if title.strip() == "" or body.strip() == "":
        st.warning("Please enter both title and content.")
    else:
        content = clean_text(title + " " + body)
        vector = tfidf.transform([content])
        result = model.predict(vector)[0]
        st.success("✅ Real News" if result == 1 else "❌ Fake News")
