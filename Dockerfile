FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-10000}"]