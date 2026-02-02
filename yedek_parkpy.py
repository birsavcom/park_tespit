import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET

# --- AYARLAR ---
KARAR_SURESI_SANIYE = 20  # Analiz süresi (Arka planda çalışır)
MIN_MATCH_THRESH = 140    # Cephe tanıma hassasiyeti

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

# 4. ANA DÖNGÜ
video_source = "gorukle.mp4"
cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 25
wait_time = int(1000 / fps)

FRAME_LIMIT = int(fps * KARAR_SURESI_SANIYE)
kml_parks = get_kml_coords('Birsav_akilli_park.kml') 

# --- SAYAÇLAR ---
counter_on = 0
counter_arka = 0

# Veritabanı
global_park_db = []
for i, kml_poly in enumerate(kml_parks):
    global_park_db.append({
        "id": i,
        "status": "BOS", 
        "color": (0, 255, 0),
        "fill_hex": "#27AE60",
        "kml_coords": kml_poly,
        "poly_video": None,    
        "assigned_view": None 
    })

frame_sayac = 0
analiz_sikligi = 3
son_label = "BELIRSIZ"

print(f"Sistem Başlatıldı. Sadece MOD bilgisi ekranda görünecek.")

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img = cv2.resize(img, (1280, 720))
    h, w = img.shape[:2]
    
    if frame_sayac % analiz_sikligi == 0:
        active_H, son_label = get_active_homography(img)
        
        # --- ZAMANLAYICI MANTIĞI (ARKA PLAN) ---
        analiz_yapilsin_mi = False
        
        if son_label == "ON_CEPHE":
            if counter_on < FRAME_LIMIT:
                counter_on += analiz_sikligi
                analiz_yapilsin_mi = True
            else:
                analiz_yapilsin_mi = False 

        elif son_label == "ARKA_CEPHE":
            if counter_arka < FRAME_LIMIT:
                counter_arka += analiz_sikligi
                analiz_yapilsin_mi = True
            else:
                analiz_yapilsin_mi = False
        
        else:
            analiz_yapilsin_mi = False

        # --- PARK ALANLARINI HESAPLA ---
        if active_H is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 1), 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            
            for i, park_data in enumerate(global_park_db):
                
                # Başka cephenin malıysa çizme
                if park_data["assigned_view"] is not None and park_data["assigned_view"] != son_label:
                    park_data["poly_video"] = None 
                    continue 

                # Koordinat hesapla (Her zaman gerekli)
                points = np.array([park_data["kml_coords"][:, [1,0]]], dtype='float32')
                try:
                    video_poly = cv2.perspectiveTransform(points, active_H)[0].astype(int)
                    
                    # Basit geometri kontrolleri
                    if not np.all((video_poly[:,0] > -200) & (video_poly[:,0] < 1500) & 
                                  (video_poly[:,1] > -200) & (video_poly[:,1] < 1000)): 
                        park_data["poly_video"] = None; continue
                    
                    if not np.any((video_poly[:,0] > 0) & (video_poly[:,0] < w) & 
                                  (video_poly[:,1] > 0) & (video_poly[:,1] < h)): 
                        park_data["poly_video"] = None; continue

                    # Zimmetleme
                    if park_data["assigned_view"] is None:
                        park_data["assigned_view"] = son_label

                    # Koordinatı kaydet
                    park_data["poly_video"] = video_poly

                    # --- ANALİZ KISMI (SADECE SÜRE BİTMEDİYSE) ---
                    if analiz_yapilsin_mi:
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [video_poly], 255)
                        count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        rect_area = cv2.contourArea(video_poly)
                        if rect_area > 0:
                            doluluk_orani = count / rect_area
                            yeni_durum = "DOLU" if doluluk_orani > 0.11 else "BOS"
                            
                            park_data["status"] = yeni_durum
                            park_data["color"] = (0, 0, 255) if yeni_durum == "DOLU" else (0, 255, 0)
                            park_data["fill_hex"] = "#E74C3C" if yeni_durum == "DOLU" else "#27AE60"

                except Exception:
                    park_data["poly_video"] = None
        
        else:
            # Geçiş anı: Gizle
            for p in global_park_db:
                p["poly_video"] = None

        # --- JSON KAYIT ---
        json_output = []
        for p in global_park_db:
            json_output.append({
                "id": p["id"],
                "status": p["status"],
                "fill": p["fill_hex"],
                "coords": [{"lat": pt[0], "lng": pt[1]} for pt in p["kml_coords"]]
            })
        with open('otoparklar.json', 'w') as f: json.dump(json_output, f)

    # --- GÖRSELLEŞTİRME ---
    for p in global_park_db:
        if p["poly_video"] is not None:
            cv2.polylines(img, [p["poly_video"]], True, p["color"], 2)

    # --- BİLGİ PANELİ (SADELEŞTİRİLMİŞ) ---
    cv2.rectangle(img, (0,0), (350, 60), (0,0,0), -1)
    
    if "ON" in son_label:
        text_color = (0, 255, 0) # Yeşil
    elif "ARKA" in son_label:
        text_color = (0, 255, 255) # Sarı/Cyan
    else:
        text_color = (0, 0, 255) # Kırmızı (Geçiş)
    
    cv2.putText(img, f"MOD: {son_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    cv2.imshow("Birsav AI - Master Otonom", img)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'): break
    
    frame_sayac += 1

cap.release()
cv2.destroyAllWindows()