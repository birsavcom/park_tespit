# Hizli uygulama build'i - Base image uzerinden (~1-2 dakika)
# Oncelik: Dockerfile.base ile base image olusturulmus olmali
#
# Build: docker build -f Dockerfile.app -t park-tespit:latest .
# Run:   docker run -p 3001:3001 park-tespit:latest

# Base image'i kullan (yerel veya registry'den)
# Yerel: park-tespit-base:latest
# Registry: YOUR_REGISTRY/park-tespit-base:latest
ARG BASE_IMAGE=park-tespit-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Sadece uygulama dosyalarini kopyala (bagimliliklar base'de zaten var)
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
COPY *.pkl .

# Web arayuzu
COPY *.html .
COPY *.json .

# Video dosyasini kopyalama (buyuk, volume ile baglanmali)
# COPY gorukle.mp4 .

# Wrapper modulu (varsa)
COPY park_tespit/ ./park_tespit/ 2>/dev/null || true

EXPOSE 3001

CMD ["python", "park.py"]
