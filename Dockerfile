FROM python:3.11-slim

# OpenCV üçün sistem asılılıqları
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements kopyala və qur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proyekt fayllarını kopyala
COPY . .

# Port 3001 aç
EXPOSE 3001

# Tətbiqi işlət
CMD ["python", "park.py"]
