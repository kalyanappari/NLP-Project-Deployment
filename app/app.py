import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import string
import os

# Load dataset

# Get the current directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the dataset using a relative path
data_path = os.path.join(BASE_DIR, r"/app/languages_dataset.csv")
data = pd.read_csv(data_path)

# Preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

# Apply preprocessing
texts = data['Texts'].apply(preprocess_text).tolist()
labels = data['Languages'].tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('classifier', MultinomialNB())
])
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌍 Indian Language Classification")
st.write("Enter a text snippet, and the model will predict the language!")

# User input
txt_input = st.text_area("Enter Text:")

if st.button("Classify Language"):
    if txt_input:
        prediction = model.predict([preprocess_text(txt_input)])[0]
        st.success(f"Predicted Language: {prediction}")
    else:
        st.warning("Please enter text to classify.")
