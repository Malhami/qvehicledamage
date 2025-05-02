FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Make the startup script executable
RUN chmod +x startup.sh

# Expose the port Streamlit will run on
EXPOSE 8501

# Start the application
CMD ["./startup.sh"]