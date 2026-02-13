FROM python:3.11-slim

# OpenCV sistem bagimliliklari
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proyekt fayllarini kopyala
COPY . .

# Port 3001 ac
EXPOSE 3001

# Uvicorn ile FastAPI baslat
CMD ["uvicorn", "map_api:app", "--host", "0.0.0.0", "--port", "3001"]
