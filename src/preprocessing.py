import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import streamlit as st


# 1. Validate word count function
def word_count(text):
    if text is None or len(text.strip()) == 0:
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