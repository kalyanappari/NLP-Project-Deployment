import pytest
from app import preprocess_text, model

# Sample test inputs
test_input_text = "Hello, how are you?"
expected_output_text = ["hello", "how", "are", "you"]  # Expected output as a list

def test_preprocess_text():
    """Test the text preprocessing function."""
    processed_text = preprocess_text(test_input_text)

    # Ensure output is a list
    assert isinstance(processed_text, list), f"Expected a list, but got {type(processed_text)}"

    # Ensure the output matches expected tokenized format
    assert processed_text == expected_output_text, f"Expected {expected_output_text}, but got {processed_text}"

def test_model_prediction():
    """Test if the model function runs without error and returns a result."""
    
    processed_input = " ".join(preprocess_text(test_input_text))  # Convert list back to string
    result = model.predict([processed_input])  # Predict using trained model

    # Ensure output is not None
    assert result is not None, "Model returned None"

    # Ensure the model output is a list (as expected from sklearn pipeline)
    assert isinstance(result, list), f"Model output should be a list, but got {type(result)}"

    # Ensure model returns a string prediction inside the list
    assert isinstance(result[0], str), f"Model prediction should be a string, but got {type(result[0])}"
