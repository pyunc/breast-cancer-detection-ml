FROM python:3.11-slim

RUN mkdir -p /opt/ml

WORKDIR /opt/ml/


RUN apt-get -y update
RUN apt-get install -y --no-install-recommends build-essential libgomp1 nginx


# Install UV - faster Python package installer
RUN pip install --no-cache-dir uv

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN uv init --bare

RUN uv venv

RUN . .venv/bin/activate

RUN uv add -r requirements.txt

COPY ./src/ /opt/ml/src
COPY ./tools/ /opt/ml/tools
COPY ./pipelines/ /opt/ml/pipelines
COPY ./models/ /opt/ml/models

# ENTRYPOINT [ "uv", "run", "python", "tools/run.py", "--run-all" ]

RUN chmod +x /opt/ml/tools/*

# Make port 8000 available (if your app exposes a web interface)
EXPOSE 8000
