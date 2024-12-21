# Import required libraries
import streamlit as st
import pickle
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Download nltk words and punkt_tab
#nltk.download('stopwords')
#nltk.download('punkt_tab')

# Load from pickle Model file
with open("Model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Load from pickle Victorizer file
with open("Tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load the model data
model = model_data["model"]
threshold = model_data["threshold"]


# Title for the webapp
st.title("Message Spam Detector")

# 1. Validate word count function
def word_count(text):
    if not text or len(text.strip()) == 0:
        st.error("Please enter a valid message.")
        return None
    words = text.split()
    if len(words) >= 3:
        return text
    else:
        st.error("Message should contain at least 3 words.")
        return None


# 2. Input text preprocessing function
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


# 3. Input text
text_input = st.text_area("Message Spam Detector", help = "Message should contain at least 3 words")


# 4. The predictions
if st.button('Predict'):
    num_words = word_count(text_input)

    # 1. Preprocess the input text
    if num_words:
        transformed_text = preprocess_text(num_words)

        # 2. Text into vector
        vector = tfidf.transform([transformed_text])

        # 3. Make predictions with the saved threshold
        proba = model.predict_proba(vector)[:, 1]
        prediction = (proba >= threshold).astype(int)
            
        # 4. Make predictions
        if prediction[0] == 0:
            st.success("This message is not spam.")
        elif prediction[0] == 1:
            st.warning("This message is spam.")
            st.write("Probability of being spam:", proba[0])
        else:
            st.error("An error occurred while processing the message.")   

