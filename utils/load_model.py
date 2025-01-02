import pickle 
import streamlit as st

# Load from pickle Model file
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("Model file not found")
    

# Load from pickle Victorizer file
def load_tfidf():
    try:
        with open("Tfidf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("Tfidf file not found")