import pytest
import string
from app.app import preprocess_text, model

# Test text preprocessing function
def test_preprocess_text():
    text = "Hello, World!"
    processed_text = preprocess_text(text)
    assert processed_text == "hello world", "Preprocessing failed to remove punctuation and lowercase text."

# Test model prediction
def test_model_prediction():
    sample_text = "नमस्ते, यह एक परीक्षण है"  # Example Hindi text
    processed_text = preprocess_text(sample_text)
    
    prediction = model.predict([processed_text])
    assert prediction is not None, "Model prediction failed."
    assert isinstance(prediction[0], str), "Prediction output should be a string."

# Run all tests when executed
if __name__ == "__main__":
    pytest.main()
