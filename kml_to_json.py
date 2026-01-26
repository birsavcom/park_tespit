import xml.etree.ElementTree as ET
import json
import re
import random
from datetime import datetime, timedelta

def plaka_olustur():
    # Çoğunlukla 16 (Bursa) plaka, arada bir 34, 06, 35 serpiştirelim
    sehir = random.choices(["16", "34", "06", "35"], weights=[70, 10, 10, 10])[0]
    harf_grupları = ["A", "AB", "ABC", "D", "DE", "F", "FG", "H", "J", "K", "L"]
    harf = random.choice(harf_grupları)
    sayi = random.randint(100, 9999)
    return f"{sehir} {harf} {sayi}"

def kml_to_json(kml_file):
    try:
        with open(kml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Namespace temizliği
        content = re.sub(r' xmlns="[^"]+"', '', content)
        root = ET.fromstring(content)
        
        final_data = []
        # KML içindeki her bir Placemark'ı tara
        for pm in root.findall('.//Placemark'):
            # 1. İSMİ BUL
            name_node = pm.find('name')
            park_adi = name_node.text.strip().upper() if name_node is not None else ""
            
            # Değişkenleri hazırla
            status = "BOS"
            fill_color = "#00FF00"
            plaka = None
            giris_timestamp = None
            giris_saati = None

            # 2. DOLU/BOŞ KONTROLÜ VE VERİ ATAMA
            if "DOLU" in park_adi:
                status = "DOLU"
                fill_color = "#FF0000"
                plaka = plaka_olustur()
                
                # Rastgele giriş saati (Son 30 dk ile 300 dk arası - Max 5 saat)
                dakika_once = random.randint(30, 300)
                giris_vakti = datetime.now() - timedelta(minutes=dakika_once)
                giris_timestamp = giris_vakti.isoformat() # JavaScript'te hesaplama için
                giris_saati = giris_vakti.strftime("%H:%M") # Panelde göstermek için
            
            # 3. KOORDİNATLARI AL
            coords_node = pm.find('.//coordinates')
            if coords_node is not None:
                pts = []
                for p in coords_node.text.strip().split():
                    parts = p.split(',')
                    if len(parts) >= 2:
                        pts.append({"lat": float(parts[1]), "lng": float(parts[0])})
                
                if pts:
                    final_data.append({
                        "coords": pts,
                        "status": status,
                        "fill": fill_color,
                        "fillOp": 0.5,
                        "stroke": "#0000FF",
                        "plaka": plaka,
                        "giris_timestamp": giris_timestamp,
                        "giris_saati": giris_saati
                    })

        # JSON DOSYASINA YAZ
        with open("otoparklar.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"Bursa: {len(final_data)} park alanı plaka ve zaman atamasıyla JSON yapıldı.")

    except Exception as e:
        print(f"Hata oluştu: {e}")

# KML dosyanın adının doğru olduğundan emin ol
kml_to_json("Birsav_akilli_park.kml")