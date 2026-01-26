import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET

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

# 2. HASSAS REFERANS NOKTALARI (GÜNCEL: 5 ÖN + 5 ARKA)

# --- ÖN CEPHE VERİLERİ (5 Nokta) ---
src_pts_on = np.array([
    [40.22686015538617, 28.845458302530535],
    [40.22691647120361, 28.845021773017173],
    [40.22730709690207, 28.84517935279603],
    [40.22698149031735, 28.845552850400534],
    [40.22713149480372, 28.84502847853899]
])

dst_pts_on = np.array([
    [445, 471], [316, 287], [1026, 303], [977, 498], [682, 269]
])

H_ON, _ = cv2.findHomography(src_pts_on[:, [1,0]], dst_pts_on, 0)

# --- ARKA CEPHE VERİLERİ (5 Nokta) ---
src_pts_arka = np.array([
    [40.22667730079056, 28.846126288495114],
    [40.22647507487638, 28.8463750633837],
    [40.22638240913122, 28.846254363979543],
    [40.22656620460386, 28.84591372343518],
    [40.22659743442532, 28.84607599708005]
])

dst_pts_arka = np.array([
    [157, 500], [466, 269], [777, 279], [1203, 582], [596, 451]
])

H_ARKA, _ = cv2.findHomography(src_pts_arka[:, [1,0]], dst_pts_arka, 0)

# 3. CEPHE TANIMA
orb = cv2.ORB_create(nfeatures=1000)
try:
    ref_on_img = cv2.imread('on_cephe_referans.jpg', 0)
    ref_arka_img = cv2.imread('arka_cephe_referans.jpg', 0)
    
    if ref_on_img is None or ref_arka_img is None:
        raise FileNotFoundError("Referans resimler eksik!")

    kp1, des1 = orb.detectAndCompute(ref_on_img, None)
    kp2, des2 = orb.detectAndCompute(ref_arka_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

except Exception as e:
    print(f"HATA: {e}")
    exit()

def get_active_homography(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = orb.detectAndCompute(gray, None)
    if des_f is None: return None, "BELIRSIZ"
    
    matches_on = len(bf.match(des1, des_f))
    matches_arka = len(bf.match(des2, des_f))
    
    if matches_on > matches_arka + 20: return H_ON, "ON CEPHE"
    if matches_arka > matches_on + 20: return H_ARKA, "ARKA CEPHE"
    return None, "GECIS / BELIRSIZ"

# 4. ANA DÖNGÜ
canli_yayin_url = "https://canliyayin.bursa.bel.tr/cdnlive/Gorukle_700.stream/chunklist_w265563719.m3u8?t=q_Aluu26_h--TRIhvf6v9w&e=1773011176"
cap = cv2.VideoCapture(canli_yayin_url)

# --- FPS KONTROLÜ ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 25
wait_time = int(1000 / fps)

kml_parks = get_kml_coords('Birsav_akilli_park.kml') 

frame_sayac = 0
analiz_sikligi = 5
son_sonuclar = []   
son_label = "BELIRSIZ"
son_active_H = None

# Park Durum Hafızası
park_durumlari = {} 

while True:
    success, img = cap.read()
    if not success:
        cap = cv2.VideoCapture(canli_yayin_url)
        continue

    img = cv2.resize(img, (1280, 720))
    h, w = img.shape[:2]
    
    if frame_sayac % analiz_sikligi == 0:
        
        son_active_H, son_label = get_active_homography(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 1), 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        
        yeni_sonuclar = []
        
        if son_active_H is not None:
            for i, kml_poly in enumerate(kml_parks):
                points = np.array([kml_poly[:, [1,0]]], dtype='float32')
                try:
                    video_poly = cv2.perspectiveTransform(points, son_active_H)[0].astype(int)
                    
                    # Filtreleme
                    if not np.all((video_poly[:,0] > -1000) & (video_poly[:,0] < 2000) & 
                                  (video_poly[:,1] > -1000) & (video_poly[:,1] < 2000)): continue 
                    rect_area = cv2.contourArea(video_poly)
                    if rect_area < 50 or rect_area > (w * h * 0.4): continue 
                    if not np.any((video_poly[:,0] > 0) & (video_poly[:,0] < w) & 
                                  (video_poly[:,1] > 0) & (video_poly[:,1] < h)): continue 

                    # --- ANALİZ ---
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [video_poly], 255)
                    count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                    
                    doluluk_orani = count / rect_area
                    
                    # --- SCHMITT TRIGGER (GÜNCELLENDİ: %15 KURALI) ---
                    eski_durum = park_durumlari.get(i, "BOS")
                    
                    if eski_durum == "BOS":
                        # Dolu olması için %15'i geçmesi yeterli
                        yeni_durum = "DOLU" if doluluk_orani > 0.15 else "BOS"
                    else:
                        # Doluysa, Boş olması için %8'in altına inmeli (İnatçı Mod)
                        yeni_durum = "BOS" if doluluk_orani < 0.08 else "DOLU"
                    
                    park_durumlari[i] = yeni_durum
                    
                    color = (0, 0, 255) if yeni_durum == "DOLU" else (0, 255, 0)
                    fill_hex = "#E74C3C" if yeni_durum == "DOLU" else "#27AE60"
                    
                    yeni_sonuclar.append({
                        "poly": video_poly,
                        "color": color,
                        "id": i,
                        "fill_hex": fill_hex,
                        "kml_coords": kml_poly,
                        "oran": doluluk_orani
                    })

                except Exception as e:
                    pass
            
            son_sonuclar = yeni_sonuclar
            
            json_output = [{"id": r["id"], "status": "DOLU" if r["color"] == (0,0,255) else "BOS", 
                            "fill": r["fill_hex"], "coords": [{"lat": p[0], "lng": p[1]} for p in r["kml_coords"]]} 
                           for r in son_sonuclar]
            with open('otoparklar.json', 'w') as f: json.dump(json_output, f)

    # --- ÇİZİM ---
    for res in son_sonuclar:
        cv2.polylines(img, [res["poly"]], True, res["color"], 2)
        
        M = cv2.moments(res["poly"])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if 0 < cX < 1280 and 0 < cY < 720:
                cv2.circle(img, (cX, cY), 10, (0,0,0), -1)
                cv2.putText(img, str(res["id"]), (cX-5, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(img, (0,0), (450, 50), (0,0,0), -1)
    if "ON" in son_label: text_color = (0, 255, 0)
    elif "ARKA" in son_label: text_color = (0, 255, 255)
    else: text_color = (0, 0, 255)
    
    cv2.putText(img, f"MOD: {son_label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    cv2.imshow("Birsav AI - Master Otonom", img)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'): break
    
    frame_sayac += 1

cap.release()
cv2.destroyAllWindows()