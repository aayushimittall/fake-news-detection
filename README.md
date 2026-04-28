
# Fake News Detection System Research Paper

Submitted by:
2210991139(Aayushi Mittal)
2210991172(Abhishek Rana)

Current status:Working Project

A machine learning–based fake news detection system that classifies news articles as **Fake** or **Real** using **TF‑IDF features** and a **Linear Support Vector Machine (SVM)**.  
The system is deployed as a simple and interactive **Streamlit web application**.

---

## 🚀 Features
- Text preprocessing using **NLTK**
- Feature extraction with **TF‑IDF (unigrams & bigrams)**
- Fake vs Real news classification using **Linear SVM**
- Interactive **Streamlit** web interface
- Clean, reproducible project structure

---

## 🛠️ Tech Stack
- **Python**
- **Scikit‑learn**
- **NLTK**
- **Pandas & NumPy**
- **Streamlit**
- **Joblib**

---

## 📁 Project Structure

``
Fake_news_detection/
│
├── src/
│   ├── app.py              # Streamlit application
│   ├── train.py            # Model training script
│   ├── preprocess.py       # Text preprocessing utilities
│
├── data/                   # Dataset directory (CSV files not tracked)
│
├── models/                 # Saved model files (ignored by Git)
│
├── make_full_data.py       # Create full dataset from Fake & True news
├── split_data.py           # Train / Validation / Test split
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files and folders
---

## 📊 Dataset

This project uses the **Fake and Real News Dataset** from **Kaggle**.

⚠️ **Dataset files are not included in this repository** due to GitHub file size limitations.

### To prepare the dataset:

1. Download the dataset from Kaggle  
   *(Fake.csv and True.csv)*

2. Place the files inside the `data/` folder:data/
├── Fake.csv
└── True.csv
3. Generate the combined dataset:
```bash
python make_full_data.py
1.Split into train, validation, and test sets:python split_data.py


🧠 Model Training
Train the TF‑IDF + SVM model using:python src/train.py
This will:

Train the TF‑IDF vectorizer
Train the Linear SVM classifier
Save model files (tfidf.pkl, svm_model.pkl) in the models/ directory

🌐 Run the Web Application
Start the Streamlit app using:streamlit run src/app.py
The app allows you to:

Enter a News Title
Enter News Content
Get a prediction: Fake News ❌ or Real News ✅

⚠️ Limitations

The system performs style‑based classification, not factual verification
Legitimate news articles may sometimes be misclassified
Dataset bias may affect predictions for unseen domains or regions
Intended as a decision‑support tool, not a fact‑checking authority

🔮 Future Enhancements

Add confidence/probability scores
Use transformer models (BERT, RoBERTa)
Incorporate source credibility signals
Evidence‑aware claim verification

✅ Conclusion
This project demonstrates that classical machine learning techniques combined with TF‑IDF features can effectively identify fake news content.
The approach balances interpretability, efficiency, and practicality, making it suitable for academic and real‑world prototyping scenarios.

👤 Author
Aayushi Mittal
Abhishek Rana

📌 Acknowledgment
This project was developed as part of an academic research and implementation on fake news detection using machine learning.

