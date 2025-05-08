FROM python:3.11

WORKDIR /usr/src

ARG REQ_TXT
COPY ${REQ_TXT} ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r ${REQ_TXT}
