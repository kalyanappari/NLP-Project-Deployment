import pytest
import sys
import os
import numpy as np

# Ensure the app directory is correctly set in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

# Import functions from app.py
from app import preprocess_text, model  

# Sample input for testing
test_input_text = "Hello, how are you?"

def test_model_prediction():
    """Test if the model runs without error and returns a valid prediction."""
    processed_input = preprocess_text(test_input_text)
    result = model.predict([processed_input])

    # Ensure result is not None
    assert result is not None, "Model returned None"

    # Convert NumPy array to list if needed
    if isinstance(result, np.ndarray):
        result = result.tolist()

    # Ensure result is a list
    assert isinstance(result, list), f"Expected list, got {type(result)}"

    # Ensure list is not empty
    assert len(result) > 0, "Model returned an empty list"

    # Ensure each element in result is a string (assuming classification task)
    assert all(isinstance(label, str) for label in result), "Expected list of strings"

