FROM python:3.11-slim

WORKDIR /app

COPY start.sh /start.sh
RUN chmod 555 /start.sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .

# Configurable env vars
ENV UPSTREAM_TYPE=google
ENV LISTEN_PORT=18080
ENV SAVE_DIR=/claude-proxy/data
ENV SAVE_ALL_REQUESTS=1

# Create default save directory
RUN mkdir -p /claude-proxy/data

CMD uvicorn proxy:app --host 0.0.0.0 --port ${LISTEN_PORT}
