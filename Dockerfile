FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends clang && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY keen/ keen/
COPY models/keen-1/ models/keen-1/
RUN useradd -m keen && mkdir /app/data && chown keen:keen /app/data
USER keen
ENV KEEN_DB_PATH=/app/data/keen.sqlite3
EXPOSE 8080
CMD ["sh", "-c", "uvicorn keen.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
