import pytest
import os
import pandas as pd
from app.app import preprocess_text, model  # Import functions from app.py

# Test dataset loading
def test_dataset_loading():
    """Test if the dataset file exists and contains valid data."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../app/languages_dataset.csv")
    assert os.path.exists(data_path), "Dataset file not found!"
    data = pd.read_csv(data_path)
    assert not data.empty, "Dataset is empty!"
    assert 'Texts' in data.columns and 'Languages' in data.columns, "Missing required columns in dataset!"

# Test text preprocessing function
def test_preprocess_text():
    """Test text preprocessing function to ensure proper text cleaning."""
    assert preprocess_text("Hello, World!") == "hello world"
    assert preprocess_text("Python is Great!!") == "python is great"
    assert preprocess_text("123@#$%^&") == "123"
    assert preprocess_text("नमस्ते! कैसे हैं?") == "नमस्ते कैसे हैं"

# Test model predictions
def test_model_predictions():
    """Test if the model predicts the correct language labels."""
    assert model.predict(["नमस्ते"])[0] == "Hindi"  # Adjust based on your dataset
    assert model.predict(["Bonjour"])[0] == "French"
    assert model.predict(["Hello"])[0] == "English"
