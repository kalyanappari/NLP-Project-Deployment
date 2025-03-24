import pytest
from app import preprocess_text, model
import numpy as np

# Sample input for testing
test_input_text = "Hello, how are you?"

def test_model_prediction():
    """Test if the model runs without error and returns a result."""
    processed_input = preprocess_text(test_input_text)
    result = model.predict([processed_input])

    assert result is not None, "Model returned None"

    # Convert NumPy array to a list before assertion
    result_list = result.tolist() if isinstance(result, np.ndarray) else result

    assert isinstance(result_list, list), f"Expected list, got {type(result_list)}"
