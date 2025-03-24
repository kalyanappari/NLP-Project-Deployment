import pytest
import sys
import os

# Ensure the correct module path is added
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from app import preprocess_text, model

# Sample test inputs
test_input_text = "Hello, how are you?"
expected_output_text = ["hello", "how", "are", "you"]

def test_preprocess_text():
    """Test the text preprocessing function."""
    processed_text = preprocess_text(test_input_text)
    assert isinstance(processed_text, str), f"Expected string, got {type(processed_text)}"
    assert processed_text == "hello how are you", f"Expected 'hello how are you', but got '{processed_text}'"

def test_model_prediction():
    """Test if the model runs without error and returns a result."""
    processed_input = preprocess_text(test_input_text)
    result = model.predict([processed_input])

    assert result is not None, "Model returned None"
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert isinstance(result[0], str), f"Expected string in list, got {type(result[0])}"
