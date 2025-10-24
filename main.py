import streamlit as st
import pickle
from scipy.sparse import hstack
from preprocess import preprocess
with open(r"C:\Users\User\Desktop\PY DS\Dixson\ML\Natural Language Processing\news.csv\model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\User\Desktop\PY DS\Dixson\ML\Natural Language Processing\news.csv\title.pkl","rb") as f:
    title_vec=pickle.load(f)
with open(r"C:\Users\User\Desktop\PY DS\Dixson\ML\Natural Language Processing\news.csv\text.pkl","rb") as f:
    text_vec=pickle.load(f)

st.title("Title Text Classifier")

title_input=preprocess(st.text_input("Enter title"))
text_input=preprocess(st.text_area("Enter text"))

if st.button("Predict"):
    if title_input.strip() and text_input.strip():
        X_title=title_vec.transform([title_input])
        X_text=text_vec.transform([text_input])
        X=hstack([X_title,X_text])
        pred=model.predict(X)[0]
        st.success(f"Prediction:{(lambda x: "Real" if x==1 else "Fake")(pred)}")
    else:
        st.warning("Please enter both title and text.")
