import cv2
import numpy as np

# Canlı yayın linkin
url = "https://canliyayin.bursa.bel.tr/cdnlive/Gorukle_700.stream/chunklist_w584479935.m3u8?t=48M9f8Yu-dD94g8B9v62Rw&e=1772578259"
cap = cv2.VideoCapture(url)

# Noktaları tutacak liste
noktalar = []
img_static = None # Dondurulmuş kareyi tutmak için
donduruldu = False # O an donuk mu canlı mı?

def click_event(event, x, y, flags, param):
    global img_static
    if event == cv2.EVENT_LBUTTONDOWN and donduruldu:
        noktalar.append([x, y])
        
        # Görsel üzerine işaret koy (Dondurulmuş resim üzerine çiziyoruz)
        cv2.circle(img_static, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_static, f"{len(noktalar)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Canli Kalibrasyon Araci", img_static)
        
        print(f"-> Nokta {len(noktalar)} Eklendi")

cv2.namedWindow("Canli Kalibrasyon Araci")
cv2.setMouseCallback("Canli Kalibrasyon Araci", click_event)

print("--- CANLI KALİBRASYON REHBERİ ---")
print("1. Kamera istediğin açıya (Ön/Arka) gelene kadar izle.")
print("2. 'BOSLUK (SPACE)' tuşuna basıp görüntüyü DONDUR.")
print("3. Referans noktalarını fare ile tıkla.")
print("4. 'c' tuşuna basıp koordinatları KOPYALANACAK SEKILDE al.")
print("5. 'r' tuşuna basıp listeyi SIFIRLA ve yayını DEVAM ETTİR (Diğer cephe için).")
print("6. 'q' ile çıkış yap.")

while True:
    # Eğer dondurulmadıysa yeni kare oku
    if not donduruldu:
        success, frame = cap.read()
        if not success:
            # Yayın koparsa tekrar bağlan
            cap = cv2.VideoCapture(url)
            continue
        
        # EN ÖNEMLİ KISIM BURASI: Senin ana kodunla aynı çözünürlüğe getiriyoruz
        frame = cv2.resize(frame, (1280, 720))
        img_display = frame.copy()
        
        # Bilgilendirme yazısı
        cv2.putText(img_display, "CANLI YAYIN (Dondurmak icin SPACE)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Canli Kalibrasyon Araci", img_display)

    else:
        # Eğer dondurulduysa, statik resmi göster
        cv2.imshow("Canli Kalibrasyon Araci", img_static)

    key = cv2.waitKey(1) & 0xFF

    # --- TUŞ KOMBİNASYONLARI ---
    
    if key == ord(' '): # BOŞLUK: Dondur / Devam Et
        if not donduruldu:
            donduruldu = True
            img_static = frame.copy() # O anki kareyi hafızaya al
            print("\n>>> GÖRÜNTÜ DONDURULDU! Nokta seçmeye başlayabilirsin.")
        else:
            print(">>> Zaten dondurulmuş durumda. Devam etmek için 'r' ile sıfırla.")

    elif key == ord('c'): # C: Koordinatları Yazdır (SENİN İSTEDİĞİN FORMAT)
        if len(noktalar) > 0:
            print("\n" + "="*40)
            print(f"--- {len(noktalar)} ADET NOKTA SECILDI ---")
            for i, p in enumerate(noktalar):
                # İşte burası senin istediğin çıktı formatı:
                print(f"Nokta {i+1}: X={p[0]}, Y={p[1]}")
            print("="*40 + "\n")
        else:
            print(">>> Henüz nokta seçmedin!")

    elif key == ord('r'): # R: Resetle ve Devam Et
        noktalar = []
        donduruldu = False
        print("\n>>> SIFIRLANDI! Canlı yayın devam ediyor. Diğer cepheyi bekle...")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()