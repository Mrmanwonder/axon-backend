FROM python:3.11-slim

WORKDIR /app

# Install system deps for potential native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Google Cloud Run sets PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 300 app:app
