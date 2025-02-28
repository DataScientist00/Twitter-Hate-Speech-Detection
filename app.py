import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
stacking_model = joblib.load("stacking_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit App
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to classify it as Hate Speech or Not Hate Speech.")

user_input = st.text_area("Enter Tweet:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_text = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = stacking_model.predict(transformed_text)[0]

        result = "Hate Speech 😡" if prediction == 1 else "Not Hate Speech 😊"
        st.title(f"Prediction: **{result}**")
    else:
        st.title("Please enter a tweet.")

