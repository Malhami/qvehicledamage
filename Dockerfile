# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (this helps with caching layer for dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set entry point for the container
CMD ["streamlit", "run", "vehicledamage2.py", "--server.port=8501", "--server.address=0.0.0.0"]
