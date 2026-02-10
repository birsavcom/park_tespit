# Park Tespit - Kullanim Kilavuzu

Bu proje baska projelerde 3 farkli sekilde kullanilabilir.

---

## Yontem 1: Docker Base Image (Onerilen - Build suresi 20dk -> 2dk)

### Adim 1: Base image'i bir kez olusturun

```bash
cd park_tespit/
docker build -f Dockerfile.base -t park-tespit-base:latest .
```

Bu islem ~15-20 dakika surecek ama **sadece bir kez** yapilacak.

### Adim 2: Base image'i registry'ye pushlayın (opsiyonel)

```bash
docker tag park-tespit-base:latest YOUR_REGISTRY/park-tespit-base:latest
docker push YOUR_REGISTRY/park-tespit-base:latest
```

### Adim 3: Uygulama build'i (artik 1-2 dakika)

```bash
docker build -f Dockerfile.app -t park-tespit:latest .
```

### Adim 4: Calistirin

```bash
docker run -p 3001:3001 -v $(pwd)/otoparklar.json:/app/otoparklar.json park-tespit:latest
```

---

## Yontem 2: Python Modulu Olarak Import

### Kurulum

```bash
# Dogrudan klasorden
pip install -e /path/to/park_tespit/

# veya Git'ten
pip install git+https://github.com/birsavcom/park_tespit.git
```

### Kullanim

```python
from park_tespit import ParkDetector, ParkConfig

# Varsayilan ayarlarla
detector = ParkDetector()
detector.start()

# Ozel ayarlarla
config = ParkConfig(
    video_source="rtsp://192.168.1.100/stream",  # IP kamera
    kml_file="benim_otoparkım.kml",
    on_cephe_referans="on_kamera.jpg",
    arka_cephe_referans="arka_kamera.jpg",
    show_window=False,  # Sunucuda False
    json_output_file="durum.json"
)

detector = ParkDetector(config)

# Callback ile (durum degisince bildirim)
def durum_degisti(park_id, yeni_durum, tum_parklar):
    print(f"Park {park_id}: {yeni_durum}")
    # Burada kendi API'nize bildirim gonderebilirsiniz

detector.start(on_status_change=durum_degisti)
```

### Ozet Bilgi Alma

```python
detector = ParkDetector()
detector._initialize()

# Ozet
print(detector.get_summary())
# {'bos': 45, 'dolu': 12, 'toplam': 57}

# Detayli durum
print(detector.get_status())
# [{'id': 0, 'status': 'BOS', 'coords': [[40.22, 28.84], ...]}, ...]
```

---

## Yontem 3: Git Submodule

```bash
cd baska_proje/
git submodule add https://github.com/birsavcom/park_tespit libs/park_tespit

# Import
import sys
sys.path.insert(0, 'libs/park_tespit')
from park_tespit import ParkDetector
```

---

## Kendi Otoparkınız Icin Kalibrasyon

### 1. Referans resimler olusturun

Her kamera acisi icin bir referans frame kaydedin:
- `on_cephe_referans.jpg`
- `arka_cephe_referans.jpg`

### 2. Homografi noktalarini belirleyin

```bash
python kalibrasyon.py  # Dosya icin
# veya
python canli_kalibrasyon.py  # Canli yayin icin
```

Haritadan 5-6 nokta secin, ayni noktalari videoda tiklayin.

### 3. Config'i guncelleyin

```python
config = ParkConfig(
    src_pts_on=np.array([
        [40.123, 28.456],  # GPS koordinatlari
        [40.124, 28.457],
        # ...
    ]),
    dst_pts_on=np.array([
        [312, 680],  # Video piksel koordinatlari
        [661, 469],
        # ...
    ]),
    # ayni sekilde arka cephe icin...
)
```

---

## Dosya Yapisi

```
park_tespit/
├── park.py              # Orijinal uygulama (degistirilmedi)
├── Dockerfile           # Orijinal Dockerfile (degistirilmedi)
├── Dockerfile.base      # YENİ: Base image (agir bagimliliklar)
├── Dockerfile.app       # YENİ: Hizli uygulama build'i
├── pyproject.toml       # YENİ: Python paket tanimı
├── KULLANIM.md          # YENİ: Bu dosya
└── park_tespit/         # YENİ: Import edilebilir modul
    ├── __init__.py
    └── detector.py      # Wrapper sinif
```

**Not:** Orijinal dosyalara (park.py, Dockerfile, vs.) hic dokunulmadi.
