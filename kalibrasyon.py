import cv2

# Video ayarları
video_path = 'gorukle.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Kare okuma fonksiyonu (1280x720)
def get_frame():
    success, frame = cap.read()
    if not success:
        return None
    return cv2.resize(frame, (1280, 720))

# İlk kareyi al
base_img = get_frame()
if base_img is None:
    print("Video açılamadı!")
    exit()

current_points = [] # Koordinatları tutacak liste

# Mouse Tıklama Fonksiyonu
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Noktayı listeye ekle
        current_points.append([x, y])
        
        # ANINDA TERMİNALE YAZ (Tek tek görmek istersen)
        print(f"Tıklanan Nokta: {x}, {y}")

# Pencere ayarı
cv2.namedWindow("Basit_Mod")
cv2.setMouseCallback("Basit_Mod", click_event)

print("--- KULLANIM ---")
print("1. Mouse ile noktaları tıkla.")
print("2. ENTER bas: O anki noktaların listesini terminale 'KOPYALANACAK FORMATTA' yazar ve sıfırlar.")
print("3. 'f' bas: 3 Saniye ileri sarar.")
print("4. 'o' bas: 'on_cephe_referans.jpg' kaydeder.")
print("5. 'a' bas: 'arka_cephe_referans.jpg' kaydeder.")
print("6. 'q' bas: Çıkış.")
print("----------------")

while True:
    # Ekrana çizim yapacağımız kopya (Orijinal bozulmasın diye)
    display_img = base_img.copy()

    # Tıkladığın noktaları kırmızı göster
    for p in current_points:
        cv2.circle(display_img, (p[0], p[1]), 5, (0, 0, 255), -1)

    cv2.imshow("Basit_Mod", display_img)
    key = cv2.waitKey(1) & 0xFF

    # --- ENTER: LİSTEYİ YAZDIR ---
    if key == 13: 
        print("\n--- KOPYALA AŞAĞIYI ---")
        print(f"Park_Alani = {current_points}")
        print("-----------------------")
        current_points = [] # Listeyi temizle, sıradaki park alanı için hazırla

    # --- F: 3 SANİYE İLERİ ---
    elif key == ord('f'):
        curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr + (fps * 3)) # 3 saniye atla
        new_frame = get_frame()
        if new_frame is not None:
            base_img = new_frame
            current_points = [] # Yeni sahnede noktaları sıfırla
            print(">>> 3 Saniye ilerletildi.")
        else:
            print("Video bitti.")

    # --- O: ÖN CEPHE KAYDET ---
    elif key == ord('o'):
        cv2.imwrite("on_cephe_referans.jpg", base_img)
        print(">>> on_cephe_referans.jpg kaydedildi (Temiz).")

    # --- A: ARKA CEPHE KAYDET ---
    elif key == ord('a'):
        cv2.imwrite("arka_cephe_referans.jpg", base_img)
        print(">>> arka_cephe_referans.jpg kaydedildi (Temiz).")

    # --- Q: ÇIKIŞ ---
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()