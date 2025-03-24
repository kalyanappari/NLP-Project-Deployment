import sys
import os
import pytest

# Ensure /app is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the functions from app.py
from app.app import preprocess_text, model  # Update based on your actual functions

# Sample test data
test_input_text = "Hello, how are you?"
expected_output_text = ["hello", "how", "are", "you"]  # Adjust based on your function's behavior

def test_preprocess_text():
    """Test the text preprocessing function."""
    processed_text = preprocess_text(test_input_text)
    assert processed_text == expected_output_text, f"Expected {expected_output_text}, but got {processed_text}"

def test_model_prediction():
    """Test if the model function runs without error and returns a result."""
    result = model.predict([test_input_text])  # Update this based on your actual model function
    assert result is not None, "Model returned None"
    assert isinstance(result, list), "Model output should be a list"

if __name__ == "__main__":
    pytest.main()
