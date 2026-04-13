FROM python:3.12-slim

COPY start.sh /start.sh
RUN chmod 555 /start.sh

# install git
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ChenShawn/Claude-Code-Token-in-Token-out-Proxy.git /claude-proxy

RUN pip install -r /claude-proxy/requirements.txt
RUN pip install litellm

WORKDIR /claude-proxy

CMD ["/start.sh"]
