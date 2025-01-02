FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG REQ_TXT
COPY ${REQ_TXT} ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r ${REQ_TXT}

COPY . .
