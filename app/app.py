import streamlit as st
import pandas as pd
import string
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Define paths for the dataset and model
dataset_path = '/app/languages_dataset.csv'
model_dir = '/app/model'
model_path = os.path.join(model_dir, 'language_classifier.pkl')

# Create the model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Get the modification times of the dataset and model
dataset_mtime = os.path.getmtime(dataset_path)
model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0

# Preprocess text
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Retrain if the dataset is newer than the model or if model doesn't exist
if dataset_mtime > model_mtime:
    print("Dataset has changed. Retraining the model...")

    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Preprocess text and labels
    texts = data['Texts'].apply(preprocess_text).tolist()
    labels = data['Languages'].tolist()

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Train the model
    model = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)

else:
    print("Model is up to date. Using the existing model...")

# Load the model
model = joblib.load(model_path)

# Streamlit UI
st.title("🌍 Indian Language Classification")
st.write("Enter a text snippet/Sentence/word, and the model will predict the language!")

# User input
txt_input = st.text_area("Enter Text:")

if st.button("Classify Language"):
    if txt_input:
        prediction = model.predict([preprocess_text(txt_input)])[0]
        st.success(f"Predicted Language: {prediction}")
    else:
        st.warning("Please enter text to classify.")
