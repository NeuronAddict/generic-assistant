FROM docker.io/library/python:3.12 AS builder

USER 0

RUN mkdir /app
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .

# Install build dependencies
#RUN apt update \
#    && apt install -y build-essential --no-install-recommends \
#    && rm -fr /var/lib/apt/lists/* \
#    && apt-get clean

RUN python -m venv $VIRTUAL_ENV

RUN pip install --upgrade pip \
    && pip install -r requirements.txt


FROM docker.io/library/python:3.12-slim as base

RUN mkdir /app
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

COPY src .

RUN chown -R root:root /app \
    && find /app -type d -exec chmod 0755 '{}' \; \
    && find /app -type f -exec chmod 0644 '{}' \;

# Install missing packages. Example for postgres
# RUN apt update \
#    && apt install -y libpq5 --no-install-recommends \
#    && rm -fr /var/lib/apt/lists/* \
#    && apt-get clean

USER 1001
