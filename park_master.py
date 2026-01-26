import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET

# 1. HARİTA (KML) OKUYUCU
def get_kml_coords(kml_file):
    coords_list = []
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
    return coords_list

# 2. REFERANS NOKTALARI (Senin Gönderdiğin Veriler)
# Ön Cephe (Açı 1)
src_pts1 = np.array([[40.226765, 28.845562], [40.226884, 28.845235], [40.226860, 28.845459], [40.226980, 28.845553]])
dst_pts1 = np.array([[292, 705], [556, 353], [661, 468], [1192, 497]])

# Arka Cephe (Açı 2)
src_pts2 = np.array([[40.226676, 28.846126], [40.226431, 28.846460], [40.226419, 28.846298], [40.226287, 28.846283]])
dst_pts2 = np.array([[373, 499], [672, 246], [869, 272], [1044, 257]])

# Homografi Matrislerini Hesapla
H1, _ = cv2.findHomography(src_pts1[:, [1,0]], dst_pts1) # Lng, Lat -> X, Y
H2, _ = cv2.findHomography(src_pts2[:, [1,0]], dst_pts2)

def transform_coords(polygon, H):
    # Harita koordinatlarını videodaki piksellere çevirir
    points = np.array([polygon[:, [1,0]]], dtype='float32')
    transformed = cv2.perspectiveTransform(points, H)
    return transformed[0].astype(int)

# 3. ANA DÖNGÜ
cap = cv2.VideoCapture('gorukle.mp4')
kml_parks = get_kml_coords('Birsav_akilli_park.kml')

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.resize(img, (1280, 720))
    
    # Zaman ve Açı Kontrolü
    fps = cap.get(cv2.CAP_PROP_FPS)
    curr_sec = (cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) % 64
    active_H = H1 if curr_sec < 32 else H2
    
    # Görüntü İşleme (Doluluk Analizi İçin)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3,3), 1), 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    
    json_output = []
    for i, kml_poly in enumerate(kml_parks):
        # Haritadaki poligonu videonun üzerine izdüşür
        video_poly = transform_coords(kml_poly, active_H)
        
        # Sadece ekranda olanları işle (Görünürlük Kontrolü)
        if np.any((video_poly[:,0] > 0) & (video_poly[:,0] < 1280) & (video_poly[:,1] > 0) & (video_poly[:,1] < 720)):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [video_poly], 255)
            count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            
            # Renk ve Durum (Eşik değerini 1500 yaptık)
            status, color, fill = ("BOS", (0,255,0), "#27AE60") if count < 1500 else ("DOLU", (0,0,255), "#E74C3C")
            
            cv2.polylines(img, [video_poly], True, color, 2)
            cv2.putText(img, f"ID:{i}", (video_poly[0][0], video_poly[0][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # HTML için JSON hazırla
            json_output.append({"id": i, "status": status, "fill": fill, 
                               "coords": [{"lat": p[0], "lng": p[1]} for p in kml_poly]})

    # Dosyayı kaydet
    with open('otoparklar.json', 'w') as f:
        json.dump(json_output, f)

    cv2.imshow("Birsav AI - Harita Entegrasyonlu Analiz", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()