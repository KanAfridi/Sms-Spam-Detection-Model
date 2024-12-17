import streamlit as st
import sklearn
import pickle

# Load the model
try:
    with open("Model.pkl", "rb") as f:
        model_data = pickle.load(f)
        print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please make sure to train the model first.")
    exit()  
    
model = model_data["model"]
threshold = model_data["threshold"]


st.title("Message Spam Detector")

message = st.text_area("Message Spam Detector")


# 2. Make predictions with the saved threshold
new_probs = model.predict_proba(message)[:, 1]
new_pred = (new_probs >= threshold).astype(int)
