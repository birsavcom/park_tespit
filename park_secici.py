import cv2
import pickle
import numpy as np

# Değişkenler
points = [] 
current_screen_parks = [] # Ekrandaki poligonlar
all_saved_parks = [] # Arka plandaki ana liste

# Varsa eski verileri yükle
try:
    with open('park_koordinatlari_yeni.pkl', 'rb') as f:
        all_saved_parks = pickle.load(f)
except:
    all_saved_parks = []

def mouseClick(events, x, y, flags, params):
    global points, current_screen_parks
    
    if events == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            current_screen_parks.append(points)
            points = []
    
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, park in enumerate(current_screen_parks):
            if cv2.pointPolygonTest(np.array(park), (x, y), False) >= 0:
                current_screen_parks.pop(i)
                break

def save_data():
    global all_saved_parks, current_screen_parks
    # Ekrandaki poligonları ana listeye kopyala (önceki kayıtlarla birleştir)
    temp_total = all_saved_parks + current_screen_parks
    # Dosyaya yaz
    with open('park_koordinatlari_yeni.pkl', 'wb') as f:
        pickle.dump(temp_total, f)
    print(f"Kaydedildi. Şu anki toplam: {len(temp_total)}")

cap = cv2.VideoCapture('gorukle.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

window_width, window_height = 1280, 720
cv2.namedWindow("Park Poligon Secici")
cv2.setMouseCallback("Park Poligon Secici", mouseClick)

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        success, img = cap.read()

    img_resized = cv2.resize(img, (window_width, window_height))
    
    while True:
        img_copy = img_resized.copy()
        
        # Saniye ve Kayıt Bilgisi
        current_sec = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)
        cv2.putText(img_copy, f"Sure: {current_sec} sn | Ekrandaki: {len(current_screen_parks)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, "S: Kaydet | F: Ilerle ve Ekrani Temizle | Q: Cikis", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Çizilenleri ekranda tut (Yeşil)
        for park in current_screen_parks:
            cv2.polylines(img_copy, [np.array(park)], True, (0, 255, 0), 2)
        
        # Noktalar (Kırmızı)
        for p in points:
            cv2.circle(img_copy, p, 5, (0, 0, 255), -1)
        if len(points) > 0:
            cv2.polylines(img_copy, [np.array(points)], False, (0, 0, 255), 1)

        cv2.imshow("Park Poligon Secici", img_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # Sadece Kaydet (Ekran temizlenmez)
            save_data()
            
        if key == ord('f'): # İlerle ve Ekranı Temizle
            # Mevcut ekrandakileri ana listeye aktar ve ekranı boşalt
            all_saved_parks.extend(current_screen_parks)
            current_screen_parks = []
            
            # 10 saniye ileri sar
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (fps * 10))
            break # Yeni kareyi oku
            
        if key == ord('q'):
            # Çıkmadan önce ekranda kalanları da kaydet
            all_saved_parks.extend(current_screen_parks)
            with open('park_koordinatlari_yeni.pkl', 'wb') as f:
                pickle.dump(all_saved_parks, f)
            cap.release()
            cv2.destroyAllWindows()
            exit()