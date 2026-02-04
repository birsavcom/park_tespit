import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import time
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

# =========================================================================
# --- 0. OTOMATİK BAŞLATMA MODÜLÜ (SİSTEMİ TEK TIKLA AÇAN KISIM) ---
# =========================================================================
def sistemi_otomatik_baslat():
    print("\n>>> [1/3] Sistem ve Dosyalar Kontrol Ediliyor...")

    # 1. KML -> JSON Dönüştürme (Harici dosya varsa çalıştırır)
    # Eğer kml_to_json.py dosyası aynı klasördeyse çalıştırır.
    if os.path.exists("kml_to_json.py"):
        print(">>> [2/3] Harita Verisi Dönüştürülüyor (KML -> JSON)...")
        os.system("python kml_to_json.py")
    else:
        # Dosya yoksa bile sistemin çökmemesi için uyarı verir ama devam eder
        print(">>> [UYARI] 'kml_to_json.py' bulunamadi. Mevcut JSON kullanilacak.")

    # 2. Yerel Sunucuyu (Localhost) Arka Planda Başlatma
    def sunucuyu_baslat():
        port = 3001
        try:
            # Sunucu ayarları
            server_address = ('', port)
            # Log kirliliğini önlemek için handler
            class SessizHandler(SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass # Konsola log basmayı engeller (Video akışını bozmasın diye)

            httpd = HTTPServer(server_address, SessizHandler)
            print(f">>> [3/3] Yerel Sunucu Aktif: http://localhost:{port}")
            httpd.serve_forever()
        except OSError:
            print(f">>> [BİLGİ] Port {port} zaten dolu veya sunucu zaten açık.")

    # Sunucuyu ana programı dondurmaması için ayrı bir "Thread" (İş Parçacığı) olarak başlatıyoruz
    t = threading.Thread(target=sunucuyu_baslat)
    t.daemon = True # Ana program (pencere) kapanınca sunucu da kapansın
    t.start()

    # 3. Tarayıcıyı Otomatik Açma
    time.sleep(1.5) # Sunucunun tam oturması için kısa bir bekleme
    url = "http://localhost:3001/otopark_demo.html"
    print(f">>> Tarayici Aciliyor: {url}\n")
    webbrowser.open(url)

# --- SİSTEMİ BAŞLAT ---
sistemi_otomatik_baslat()

# =========================================================================
# --- GÖRÜNTÜ İŞLEME VE PARK ANALİZ KODLARI (DEĞİŞTİRİLMEDEN EKLENDİ) ---
# =========================================================================

# --- AYARLAR ---
KARAR_SURESI_SANIYE = 20  # Analiz süresi (Arka planda çalışır)
MIN_MATCH_THRESH = 140    # Cephe tanıma hassasiyeti
FRAME_SKIP = 4
ONAY_SAYISI = 25

# 1. HARİTA (KML) OKUYUCU
def get_kml_coords(kml_file):
    coords_list = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        for pm in root.findall('.//kml:Placemark', ns):
            coord_text = pm.find('.//kml:coordinates', ns)
            if coord_text is not None:
                raw = coord_text.text.strip().split()
                polygon = []
                for c in raw:
                    lon, lat, _ = c.split(',')
                    polygon.append([float(lat), float(lon)])
                coords_list.append(np.array(polygon))
    except Exception as e:
        print(f"KML Hatasi: {e}")
    return coords_list

# 2. REFERANS NOKTALARI
src_pts_on = np.array([
    [40.226769273110655, 28.845551470761293], [40.22686092974981, 28.845459604561885],
    [40.226980216833795, 28.845554152430562], [40.22730684734409, 28.845179313718837],
    [40.22691719208239, 28.8450261768546], [40.227106617666976, 28.84507780937765]
])
dst_pts_on = np.array([
    [312, 680], [661, 469], [1193, 496], [1243, 301], [528, 284], [876, 283]
])
H_ON, _ = cv2.findHomography(src_pts_on[:, [1,0]], dst_pts_on, 0)

src_pts_arka = np.array([
    [40.22666193600637, 28.84610909492169], [40.22647097328205, 28.846383350795396],
    [40.22641977675054, 28.84629752010563], [40.22638189129232, 28.846253934208484],
    [40.22656394246121, 28.845982327216127]
])
dst_pts_arka = np.array([
    [445, 505], [679, 268], [869, 272], [993, 278], [1153, 505]
])
H_ARKA, _ = cv2.findHomography(src_pts_arka[:, [1,0]], dst_pts_arka, 0)

# 3. CEPHE TANIMA
orb = cv2.ORB_create(nfeatures=2500)
try:
    ref_on_img = cv2.imread('on_cephe_referans.jpg', 0)
    ref_arka_img = cv2.imread('arka_cephe_referans.jpg', 0)
    if ref_on_img is None or ref_arka_img is None: raise FileNotFoundError
    kp1, des1 = orb.detectAndCompute(ref_on_img, None)
    kp2, des2 = orb.detectAndCompute(ref_arka_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
except Exception as e:
    print(f"HATA: Referans resimler eksik! {e}")
    exit()

def get_active_homography(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = orb.detectAndCompute(gray, None)
    if des_f is None: return None, "BELIRSIZ"
    
    matches_on = len(bf.match(des1, des_f))
    matches_arka = len(bf.match(des2, des_f))
    max_match = max(matches_on, matches_arka)

    if max_match < MIN_MATCH_THRESH: return None, "GECIS"

    if matches_on > matches_arka + 30: return H_ON, "ON_CEPHE"
    if matches_arka > matches_on + 30: return H_ARKA, "ARKA_CEPHE"
    return None, "GECIS"

# =========================================================================
# --- 4. ANA DÖNGÜ (YAYA FİLTRELİ VERSİYON) ---
# =========================================================================
video_source = "gorukle.mp4"
cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 25
wait_time = int(1000 / fps)

FRAME_LIMIT = int(fps * KARAR_SURESI_SANIYE)
kml_parks = get_kml_coords('Birsav_akilli_park.kml') 

# --- AYARLAR ---
FRAME_SKIP = 5       # Analiz hızı
ONAY_SAYISI = 15     # Yaya filtresi (Yaklaşık 3-4 saniye bekleme süresi)

# --- SAYAÇLAR ---
counter_on = 0
counter_arka = 0

# Veritabanını Hazırla (Sayaçlı Versiyon)
global_park_db = []
for i, kml_poly in enumerate(kml_parks):
    global_park_db.append({
        "id": i,
        "status": "BOS", 
        "color": (0, 255, 0),
        "fill_hex": "#27AE60",
        "kml_coords": kml_poly,
        "poly_video": None,    
        "assigned_view": None,
        "dolu_sayaci": 0  # <--- ÖNEMLİ: Her parkın kendi sayacı var
    })

frame_sayac = 0
analiz_sikligi = 3
son_label = "BELIRSIZ"

print(f"Sistem Baslatildi. Yaya filtresi aktif. Onay Limiti: {ONAY_SAYISI}")

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- HIZLANDIRMA (FRAME SKIP) ---
    frame_sayac += 1
    if frame_sayac % FRAME_SKIP != 0:
        # Analiz yapma, sadece çizim yap
        img_display = cv2.resize(img, (1280, 720))
        for p in global_park_db:
            if p["poly_video"] is not None:
                cv2.polylines(img_display, [p["poly_video"]], True, p["color"], 2)
        
        # Bilgi Paneli
        cv2.rectangle(img_display, (0,0), (350, 60), (0,0,0), -1)
        text_color = (0, 255, 0) if "ON" in son_label else ((0, 255, 255) if "ARKA" in son_label else (0, 0, 255))
        cv2.putText(img_display, f"MOD: {son_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        cv2.imshow("Birsav AI - Master Otonom", img_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # --- ANALİZ KARESİ ---
    img = cv2.resize(img, (1280, 720))
    h, w = img.shape[:2]
    
    active_H, son_label = get_active_homography(img)
    
    # Zamanlayıcılar
    analiz_yapilsin_mi = False
    if son_label == "ON_CEPHE":
        if counter_on < FRAME_LIMIT:
            counter_on += FRAME_SKIP
            analiz_yapilsin_mi = True
        else: analiz_yapilsin_mi = False 
    elif son_label == "ARKA_CEPHE":
        if counter_arka < FRAME_LIMIT:
            counter_arka += FRAME_SKIP
            analiz_yapilsin_mi = True
        else: analiz_yapilsin_mi = False
    else: analiz_yapilsin_mi = False

    if active_H is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 1), 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        
        for i, park_data in enumerate(global_park_db):
            
            # Cephe Kontrolü
            if park_data["assigned_view"] is not None and park_data["assigned_view"] != son_label:
                park_data["poly_video"] = None; continue 

            # Perspektif Dönüşüm
            points = np.array([park_data["kml_coords"][:, [1,0]]], dtype='float32')
            try:
                video_poly = cv2.perspectiveTransform(points, active_H)[0].astype(int)
                
                # Ekran Dışı Kontrolü
                if not np.all((video_poly[:,0] > -200) & (video_poly[:,0] < 1500) & (video_poly[:,1] > -200) & (video_poly[:,1] < 1000)): 
                    park_data["poly_video"] = None; continue
                if not np.any((video_poly[:,0] > 0) & (video_poly[:,0] < w) & (video_poly[:,1] > 0) & (video_poly[:,1] < h)): 
                    park_data["poly_video"] = None; continue

                if park_data["assigned_view"] is None: park_data["assigned_view"] = son_label
                park_data["poly_video"] = video_poly

                # --- KARAR MEKANİZMASI (BURASI KRİTİK) ---
                if analiz_yapilsin_mi:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [video_poly], 255)
                    count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                    rect_area = cv2.contourArea(video_poly)
                    
                    if rect_area > 0:
                        doluluk_orani = count / rect_area
                        anlik_tespit = doluluk_orani > 0.11
                        
                        # DEBUG: Terminale yazdır ki ne olduğunu görelim
                        # print(f"Park {park_data['id']} Oran: {doluluk_orani:.2f} Sayac: {park_data['dolu_sayaci']}")

                        # SAYAÇ MANTIĞI:
                        if anlik_tespit:
                            park_data["dolu_sayaci"] += 1
                        else:
                            park_data["dolu_sayaci"] = 0 # Boş görünce hemen sıfırla (veya yavaş yavaş düşür)

                        # NİHAİ KARAR: Sadece sayaç limiti geçerse DOLU de!
                        if park_data["dolu_sayaci"] >= ONAY_SAYISI:
                            # Sayacı tavan yap ki taşmasın
                            if park_data["dolu_sayaci"] > 100: park_data["dolu_sayaci"] = 100
                            yeni_durum = "DOLU"
                        else:
                            # Sayaç 24 bile olsa, henüz 'BOS' demelisin
                            yeni_durum = "BOS"

                        park_data["status"] = yeni_durum
                        park_data["color"] = (0, 0, 255) if yeni_durum == "DOLU" else (0, 255, 0)
                        park_data["fill_hex"] = "#E74C3C" if yeni_durum == "DOLU" else "#27AE60"

            except Exception:
                park_data["poly_video"] = None
    else:
        for p in global_park_db: p["poly_video"] = None

    # JSON KAYIT
    json_output = []
    for p in global_park_db:
        json_output.append({
            "id": p["id"], "status": p["status"], "fill": p["fill_hex"],
            "coords": [{"lat": pt[0], "lng": pt[1]} for pt in p["kml_coords"]]
        })
    with open('otoparklar.json', 'w') as f: json.dump(json_output, f)

    # Görselleştirme
    for p in global_park_db:
        if p["poly_video"] is not None:
            cv2.polylines(img, [p["poly_video"]], True, p["color"], 2)

    # Panel
    cv2.rectangle(img, (0,0), (350, 60), (0,0,0), -1)
    text_color = (0, 255, 0) if "ON" in son_label else ((0, 255, 255) if "ARKA" in son_label else (0, 0, 255))
    cv2.putText(img, f"MOD: {son_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    cv2.imshow("Birsav AI - Master Otonom", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()