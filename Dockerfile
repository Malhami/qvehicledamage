FROM streamlit/streamlit:latest  # Prebuilt image with Streamlit
WORKDIR /app
COPY . .
CMD ["streamlit", "run", "vehicledamage2.py"]
