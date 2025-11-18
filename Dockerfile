# syntax=docker/dockerfile:1

FROM python:3.11-slim

RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code + Makefile
COPY main.ipynb train.py predict.py Makefile ./

# Copy data directory (including zone_city_zip.csv if it's under data/)
COPY data ./data

ENV METERED_DIR=/app/data \
    WEATHER_DIR=/app/data/weather_cities \
    ARTIFACT_DIR=/app/data/artifacts

CMD ["/bin/bash"]
