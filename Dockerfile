# Dockerfile
FROM python:3.9-slim

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

CMD ["python", "main.py"]
