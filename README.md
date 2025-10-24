# 📰 News Authenticity Detection using Machine Learning

A machine learning project to classify news articles as **Real** or **Fake** using NLP and text classification techniques.  
Deployed as an interactive **Streamlit web app** for live predictions.

---

## 📘 Overview
The model analyzes the textual content of news articles to detect misinformation.  
TF-IDF vectorization was applied for feature extraction, followed by multiple machine learning models for classification.

---

## 🧠 Model Performance

| Model                              | Accuracy (%) |
|------------------------------------|---------------|
| RandomForestClassifier             | 89.32         |
| SVC (class_weight='balanced')      | 91.74         |
| LogisticRegression                 | 91.48         |
| MultinomialNB                      | 85.06         |

After **GridSearchCV** hyperparameter tuning, the optimized **SVC model** achieved an accuracy of **91.97%**.

---

## ⚙️ Tools & Libraries
- Python  
- Scikit-learn  
- Pandas, NumPy  
- NLTK  
- Streamlit  

---

## 🚀 Run the App
```bash
streamlit run app.py
