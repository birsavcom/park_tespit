import cv2

# Videoyu aç
cap = cv2.VideoCapture('gorukle.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

def get_frame():
    success, frame = cap.read()
    if not success:
        return None
    return cv2.resize(frame, (1280, 720))

img = get_frame()
ref_points = []

def select_points(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))
        # Tıklanan yeri işaretle
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"Nokta {len(ref_points)}", (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Kalibrasyon", img)
        
        # Terminale saniye bilgisiyle yazdır ki hangisi hangi açıya ait bilelim
        curr_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        print(f"Saniye {curr_sec:.1f} -> Nokta {len(ref_points)}: X={x}, Y={y}")

cv2.namedWindow("Kalibrasyon")
cv2.setMouseCallback("Kalibrasyon", select_points)

print("--- KALİBRASYON MODU ---")
print("1. Ön cephede 4 uzak nokta seç.")
print("2. 'f' tuşuna basıp 30 saniye ilerle (Arka cepheye geç).")
print("3. Arka cephede de 4 uzak nokta seç.")
print("4. İşlem bitince 'q' ile çık.")

while True:
    cv2.imshow("Kalibrasyon", img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('f'): # 30 Saniye İleri Sar
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (fps * 30))
        new_img = get_frame()
        if new_img is not None:
            img = new_img
            print(">>> 30 saniye ilerletildi. Yeni açı için noktaları seçebilirsin.")
        else:
            print(">>> Video sonuna gelindi!")

    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()