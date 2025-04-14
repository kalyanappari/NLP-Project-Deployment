# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code (including app directory and tests)
COPY app /app/app/
COPY tests /app/tests/

# Ensure the dataset is included
COPY languages_dataset.csv /app/

# Add empty __init__.py files to mark directories as Python packages
RUN touch /app/app/__init__.py
RUN touch /app/tests/__init__.py

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
