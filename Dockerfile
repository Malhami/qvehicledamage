FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "vehicledamage2.py", "--server.port=8501", "--server.address=0.0.0.0"]