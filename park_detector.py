import cv2
import pickle
import numpy as np
import json
import random
from datetime import datetime, timedelta

# --- AYARLAR ---
on_cephe_sayisi = 5 
pkl_path = 'park_koordinatlari.pkl'
video_path = 'gorukle.mp4'

# --- RASTGELE VERİ ÜRETİCİLERİ ---
def plaka_uret():
    sehirler = ['16'] * 8 + ['34', '35', '10', '06'] # %80 Bursa ağırlıklı
    sehir = random.choice(sehirler)
    harfler = "".join(random.choices("ABCDEFGHJKLMNPRSTUVYZ", k=random.randint(1, 3)))
    sayilar = random.randint(100, 999)
    return f"{sehir} {harfler} {sayilar}"

def saat_uret():
    simdi = datetime.now()
    rastgele_dakika = random.randint(10, 120)
    giris_vakti = simdi - timedelta(minutes=rastgele_dakika)
    return giris_vakti.strftime("%H:%M")

# --- HAFIZA SİSTEMİ ---
# Her park yeri için araç bilgilerini burada tutacağız
park_hafizasi = {} # {park_id: {"plaka": "...", "saat": "...", "no": "..."}}

with open(pkl_path, 'rb') as f:
    video_poligonlari = pickle.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

def check_spaces_pro(imgPro, imgCanvas, active_indices):
    display_data = []
    
    for idx in active_indices:
        if idx >= len(video_poligonlari): continue
        park = video_poligonlari[idx]
        
        mask = np.zeros(imgPro.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(park)], 255)
        res = cv2.bitwise_and(imgPro, imgPro, mask=mask)
        count = cv2.countNonZero(res)

        # DOLULUK TESPİTİ (Eşik 1100)
        is_full = count > 1100 

        if is_full:
            status, color, fill = "DOLU", (0, 0, 255), "#E74C3C"
            # Eğer bu park yeri hafızada yoksa yeni araç ata
            if idx not in park_hafizasi:
                park_hafizasi[idx] = {
                    "no": f"{idx+1:02d}",
                    "plaka": plaka_uret(),
                    "saat": saat_uret()
                }
            
            info = park_hafizasi[idx]
            # Görsel Etiket Çizimi
            # Kutunun üstüne siyah bir şerit çekelim yazı okunsun diye
            cv2.rectangle(imgCanvas, (park[0][0]-5, park[0][1]-45), (park[0][0]+160, park[0][1]-5), (0,0,0), -1)
            text = f"No:{info['no']} | {info['plaka']}"
            time_text = f"Giris: {info['saat']}"
            cv2.putText(imgCanvas, text, (park[0][0], park[0][1]-28), 1, 0.8, (255, 255, 255), 1)
            cv2.putText(imgCanvas, time_text, (park[0][0], park[0][1]-10), 1, 0.8, (0, 255, 255), 1)
        else:
            status, color, fill = "BOS", (0, 255, 0), "#27AE60"
            # Araç çıktıysa hafızayı temizle
            if idx in park_hafizasi:
                del park_hafizasi[idx]

        cv2.polylines(imgCanvas, [np.array(park)], True, color, 2)
            
    return display_data

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_sec = (cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) % 64
    img = cv2.resize(img, (1280, 720))
    
    # Görüntü İşleme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 1), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # --- ZAMANLAMA (26-27 SN DÜZENLEMESİ) ---
    active_range = None
    if 0 <= current_sec < 26:
        active_range = range(0, on_cephe_sayisi)
        label = "ON CEPHE"
    elif 27 <= current_sec < 58:
        active_range = range(on_cephe_sayisi, len(video_poligonlari))
        label = "ARKA CEPHE"
    else:
        label = "GEÇİŞ SÜRECİ..."

    if active_range is not None:
        check_spaces_pro(thresh, img, active_range)

    # Arayüz Bilgileri
    cv2.rectangle(img, (0,0), (1280, 50), (40, 40, 40), -1)
    cv2.putText(img, f"BIRSAV AI OTOPARK TAKIP | {label} | Zaman: {current_sec:.1f}s", (250, 35), 1, 1.5, (255, 255, 255), 2)
    
    cv2.imshow("Birsav AI Pro - Gorukle", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()