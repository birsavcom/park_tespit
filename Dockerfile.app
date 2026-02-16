# Hizli uygulama build'i - Base image uzerinden (~1-2 dakika)
# Oncelik: Dockerfile.base ile base image olusturulmus olmali
#
# Build: docker build -f Dockerfile.app -t park-tespit:latest .
# Run:   docker run -p 3001:3001 park-tespit:latest

ARG BASE_IMAGE=park-tespit-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Kalan bagimliliklari kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalari
COPY map_api.py .
COPY park.py .
COPY park_detector.py .
COPY park_master.py .
COPY kml_to_json.py .
COPY kalibrasyon.py .
COPY canli_kalibrasyon.py .
COPY park_secici.py .

# Veri dosyalari
COPY *.kml .
COPY *.jpg .

# Web arayuzu
COPY *.html .
COPY *.json .

# Wrapper modulu (varsa)
COPY park_tespit/ ./park_tespit/

EXPOSE 3001

# Uvicorn ile FastAPI baslat
CMD ["uvicorn", "map_api:app", "--host", "0.0.0.0", "--port", "3001"]
