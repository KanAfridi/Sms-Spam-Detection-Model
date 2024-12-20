# Import required libraries
import streamlit as st
import pickle
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Load from pickle Model file
with open("Model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Load from pickle Victorizer file
with open("Tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load the model data
model = model_data["model"]
threshold = model_data["threshold"]

# Download nltk words and punkt_tab
nltk.download('stopwords')
nltk.download('punkt_tab')

# Title for the webapp
st.title("Message Spam Detector")

# Input text preprocessing function
def preprocess_text(text):
    swords = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    data = text.lower()
    
    tokenzie = nltk.word_tokenize(data)
    
    filter_tokenzie = []
    for word in tokenzie:
        if word.isalnum() and len(word) > 1:
            filter_tokenzie.append(word)
    
    stopwords_filter = []
    for word in filter_tokenzie:
        if word not in swords:
            stopwords_filter.append(word)
    
    stemming_filter = []
    for word in stopwords_filter:
        stemming_filter.append(stemmer.stem(word))
        
    return " ".join(stemming_filter)


# 1. Input text
text_input = st.text_area("Message Spam Detector")

# 2. Preprocess the input text
transformed_text = preprocess_text(text_input)

# 3. Text into vector
vector = tfidf.transform([transformed_text])

# 3. Make predictions with the saved threshold
proba = model.predict_proba(vector)[:, 1]
prediction = (proba >= threshold).astype(int)

# 4. Display the predictions
if st.button('Predict'):
    if prediction[0] == 0:
        st.success("This message is not spam.")
    elif prediction[0] == 1:
        st.warning("This message is spam.")
        st.write("Probability of being spam:", proba[0])
    else:
        st.error("An error occurred while processing the message.")   

