 
# Base image with Ubuntu + Python
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit-specific environment settings
ENV PORT=8501
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "vehicledamage2.py", "--server.port=$PORT", "--server.address=0.0.0.0"]

