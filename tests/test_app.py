import pytest
from app.app import preprocess_text, model  # Import functions from app.py

test_input_text = "Hello, how are you?"
expected_output_text = ["hello", "how", "are", "you"]  # Ensure this matches actual preprocessing output

def test_preprocess_text():
    """Test the text preprocessing function."""
    processed_text = preprocess_text(test_input_text)
    
    # Ensure output is a list
    assert isinstance(processed_text, list), f"Expected a list, but got {type(processed_text)}"
    
    # Check if output matches expected tokens
    assert processed_text == expected_output_text, f"Expected {expected_output_text}, but got {processed_text}"

def test_model_prediction():
    """Test if the model function runs without error and returns a valid result."""
    result = model.predict([test_input_text])  # Ensure model accepts a list
    
    # Ensure output is not None
    assert result is not None, "Model returned None"
    
    # Convert NumPy array to list before assertion
    result = result.tolist() if not isinstance(result, list) else result
    
    # Ensure output is a list
    assert isinstance(result, list), f"Model output should be a list, but got {type(result)}"
