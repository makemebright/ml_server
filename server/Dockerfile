FROM python:3.11-slim
WORKDIR /ml_server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
COPY classifier.py .
COPY .env .
EXPOSE 8991
ENV PORT=8991
CMD ["python", "server.py"]