FROM python:3.11.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "vehicledamage2.py", "--server.port=8000", "--server.address=0.0.0.0"]
