import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only needed once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define your preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the trained model
model = joblib.load("spam_classifier.pkl")

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message to check whether it's spam or not.")

user_input = st.text_area("Message:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned_text = preprocess(user_input)
        prediction = model.predict([cleaned_text])[0]

        if prediction == 0:
            st.error("🚫 Spam Detected")
        else:
            st.success("✅ Not Spam")

# Optional: Info section
with st.expander("ℹ️ About"):
    st.write("""
        - This app uses a Naive Bayes model trained on a spam dataset.
        - Messages are cleaned using tokenization, stopword removal, punctuation removal, and lemmatization.
        - Developed using Streamlit and scikit-learn.
    """)
