# Import required libraries
import streamlit as st
import os, sys

# import preprocessing and wordcount function from preprocessing.py file
from preprocessing import word_count, preprocess_text 

# import model and tfidf from app file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_model import load_model, load_tfidf
# ----------------------------------------------------------------


# Load the model and tfidf
model, tfidf = load_model(), load_tfidf()

# Title for the webapp
st.title("SMS Spam Detection Model")

# 3. The box of input text 
text_input = st.text_area(
    "Enter your message:",
    placeholder="Type your SMS message here...",
    help="Ensure the message has at least 3 words to be processed."
)


# 4. The predictions
if st.button('Predict'):
    if not text_input.strip():
        st.error("Please enter a message.")
    else:
        num_words = word_count(text_input)

        # 1. Preprocess the input text
        if num_words:
            transformed_text = preprocess_text(num_words)

            # 2. Text into vector
            vector = tfidf.transform([transformed_text])

            # 3. Make predictions with the saved threshold
            proba = model.predict_proba(vector)[:, 1]
            prediction = (proba >= 75).astype(int) # i'm gonna use probability more than 75 for a good precision 
                
            # 4. Make predictions
            if prediction[0] == 0:
                st.success("This message is not spam.")
            elif prediction[0] == 1:
                st.warning("This message is spam.")
                st.write("Probability of being spam:", proba[0])
            else:
                st.error("An error occurred while processing the message.")   

