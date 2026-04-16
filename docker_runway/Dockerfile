FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .

# Configurable env vars
ENV UPSTREAM_TYPE=aws
ENV LISTEN_PORT=8080
ENV SAVE_DIR=/data/logs
ENV SAVE_ALL_REQUESTS=false

# Create default save directory
RUN mkdir -p /data/logs

EXPOSE 8080

CMD uvicorn proxy:app --host 0.0.0.0 --port ${LISTEN_PORT}
