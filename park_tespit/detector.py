"""
Park Tespit Modulu - Baska projelerde kullanmak icin wrapper sinif

Kullanim:
    from park_tespit import ParkDetector, ParkConfig

    config = ParkConfig(
        video_source="rtsp://kamera_ip/stream",
        kml_file="otopark.kml",
        on_cephe_referans="on_cephe.jpg",
        arka_cephe_referans="arka_cephe.jpg"
    )

    detector = ParkDetector(config)
    detector.start()  # Blocking

    # veya callback ile:
    def durum_degisti(park_id, yeni_durum, tum_parklar):
        print(f"Park {park_id}: {yeni_durum}")

    detector.start(on_status_change=durum_degisti)
"""

import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ParkConfig:
    """Park tespit sistemi konfigurasyonu"""

    # Video kaynagi (dosya yolu, RTSP URL, veya kamera index)
    video_source: str = "gorukle.mp4"

    # KML dosyasi (park alanlari koordinatlari)
    kml_file: str = "Birsav_akilli_park.kml"

    # Referans resimler (cephe tanima icin)
    on_cephe_referans: str = "on_cephe_referans.jpg"
    arka_cephe_referans: str = "arka_cephe_referans.jpg"

    # Homografi noktalari - ON CEPHE
    # GPS koordinatlari (lat, lon)
    src_pts_on: np.ndarray = field(default_factory=lambda: np.array([
        [40.226769273110655, 28.845551470761293],
        [40.22686092974981, 28.845459604561885],
        [40.226980216833795, 28.845554152430562],
        [40.22730684734409, 28.845179313718837],
        [40.22691719208239, 28.8450261768546],
        [40.227106617666976, 28.84507780937765]
    ]))
    # Video piksel koordinatlari
    dst_pts_on: np.ndarray = field(default_factory=lambda: np.array([
        [312, 680], [661, 469], [1193, 496], [1243, 301], [528, 284], [876, 283]
    ]))

    # Homografi noktalari - ARKA CEPHE
    src_pts_arka: np.ndarray = field(default_factory=lambda: np.array([
        [40.22666193600637, 28.84610909492169],
        [40.22647097328205, 28.846383350795396],
        [40.22641977675054, 28.84629752010563],
        [40.22638189129232, 28.846253934208484],
        [40.22656394246121, 28.845982327216127]
    ]))
    dst_pts_arka: np.ndarray = field(default_factory=lambda: np.array([
        [445, 505], [679, 268], [869, 272], [993, 278], [1153, 505]
    ]))

    # Tespit parametreleri
    karar_suresi_saniye: int = 20
    min_match_thresh: int = 140
    frame_skip: int = 5
    onay_sayisi: int = 15
    doluluk_esigi: float = 0.11

    # Cikti
    json_output_file: str = "otoparklar.json"
    show_window: bool = False  # Sunucuda False olmali
    window_size: tuple = (1280, 720)


class ParkDetector:
    """Ana park tespit sinifi"""

    def __init__(self, config: Optional[ParkConfig] = None):
        self.config = config or ParkConfig()
        self._running = False
        self._park_db: List[Dict[str, Any]] = []
        self._cap = None
        self._orb = None
        self._bf = None
        self._H_ON = None
        self._H_ARKA = None
        self._des1 = None
        self._des2 = None

    def _load_kml_coords(self) -> List[np.ndarray]:
        """KML dosyasindan park koordinatlarini yukle"""
        coords_list = []
        try:
            tree = ET.parse(self.config.kml_file)
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            for pm in root.findall('.//kml:Placemark', ns):
                coord_text = pm.find('.//kml:coordinates', ns)
                if coord_text is not None:
                    raw = coord_text.text.strip().split()
                    polygon = []
                    for c in raw:
                        parts = c.split(',')
                        if len(parts) >= 2:
                            lon, lat = float(parts[0]), float(parts[1])
                            polygon.append([lat, lon])
                    if polygon:
                        coords_list.append(np.array(polygon))
        except Exception as e:
            print(f"KML Hatasi: {e}")
        return coords_list

    def _initialize(self):
        """Sistemi baslat"""
        # Homografi matrislerini hesapla
        self._H_ON, _ = cv2.findHomography(
            self.config.src_pts_on[:, [1, 0]],
            self.config.dst_pts_on, 0
        )
        self._H_ARKA, _ = cv2.findHomography(
            self.config.src_pts_arka[:, [1, 0]],
            self.config.dst_pts_arka, 0
        )

        # ORB ve BFMatcher
        self._orb = cv2.ORB_create(nfeatures=2500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Referans resimleri yukle
        ref_on = cv2.imread(self.config.on_cephe_referans, 0)
        ref_arka = cv2.imread(self.config.arka_cephe_referans, 0)

        if ref_on is None or ref_arka is None:
            raise FileNotFoundError(
                f"Referans resimler bulunamadi: "
                f"{self.config.on_cephe_referans}, {self.config.arka_cephe_referans}"
            )

        _, self._des1 = self._orb.detectAndCompute(ref_on, None)
        _, self._des2 = self._orb.detectAndCompute(ref_arka, None)

        # Video kaynagini ac
        self._cap = cv2.VideoCapture(self.config.video_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {self.config.video_source}")

        # Park veritabanini olustur
        kml_parks = self._load_kml_coords()
        self._park_db = []
        for i, kml_poly in enumerate(kml_parks):
            self._park_db.append({
                "id": i,
                "status": "BOS",
                "color": (0, 255, 0),
                "fill_hex": "#27AE60",
                "kml_coords": kml_poly,
                "poly_video": None,
                "assigned_view": None,
                "dolu_sayaci": 0
            })

        print(f"Park Tespit Sistemi Baslatildi: {len(self._park_db)} park alani yuklendi")

    def _get_active_homography(self, frame):
        """Aktif kamera cephesini tespit et"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des_f = self._orb.detectAndCompute(gray, None)

        if des_f is None:
            return None, "BELIRSIZ"

        matches_on = len(self._bf.match(self._des1, des_f))
        matches_arka = len(self._bf.match(self._des2, des_f))
        max_match = max(matches_on, matches_arka)

        if max_match < self.config.min_match_thresh:
            return None, "GECIS"

        if matches_on > matches_arka + 30:
            return self._H_ON, "ON_CEPHE"
        if matches_arka > matches_on + 30:
            return self._H_ARKA, "ARKA_CEPHE"

        return None, "GECIS"

    def _save_json(self):
        """Durumu JSON dosyasina kaydet"""
        json_output = []
        for p in self._park_db:
            json_output.append({
                "id": p["id"],
                "status": p["status"],
                "fill": p["fill_hex"],
                "coords": [{"lat": pt[0], "lng": pt[1]} for pt in p["kml_coords"]]
            })
        with open(self.config.json_output_file, 'w') as f:
            json.dump(json_output, f)

    def get_status(self) -> List[Dict[str, Any]]:
        """Tum parklarin guncel durumunu getir"""
        return [
            {
                "id": p["id"],
                "status": p["status"],
                "coords": p["kml_coords"].tolist()
            }
            for p in self._park_db
        ]

    def get_summary(self) -> Dict[str, int]:
        """Ozet istatistik"""
        bos = sum(1 for p in self._park_db if p["status"] == "BOS")
        dolu = sum(1 for p in self._park_db if p["status"] == "DOLU")
        return {"bos": bos, "dolu": dolu, "toplam": len(self._park_db)}

    def stop(self):
        """Sistemi durdur"""
        self._running = False

    def start(self, on_status_change: Optional[Callable] = None):
        """
        Ana tespit dongusunu baslat

        Args:
            on_status_change: Durum degistiginde cagrilacak callback
                              callback(park_id, yeni_durum, tum_parklar)
        """
        self._initialize()
        self._running = True

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 25

        frame_limit = int(fps * self.config.karar_suresi_saniye)
        counter_on = 0
        counter_arka = 0
        frame_sayac = 0
        w, h = self.config.window_size

        previous_status = {p["id"]: p["status"] for p in self._park_db}

        print("Tespit dongusu baslatildi...")

        while self._running:
            success, img = self._cap.read()
            if not success:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_sayac += 1
            if frame_sayac % self.config.frame_skip != 0:
                continue

            img = cv2.resize(img, (w, h))
            active_H, label = self._get_active_homography(img)

            # Zamanlayicilar
            analiz_yapilsin_mi = False
            if label == "ON_CEPHE":
                if counter_on < frame_limit:
                    counter_on += self.config.frame_skip
                    analiz_yapilsin_mi = True
            elif label == "ARKA_CEPHE":
                if counter_arka < frame_limit:
                    counter_arka += self.config.frame_skip
                    analiz_yapilsin_mi = True

            if active_H is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(
                    cv2.GaussianBlur(gray, (5, 5), 1), 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
                )

                for park_data in self._park_db:
                    if park_data["assigned_view"] is not None and park_data["assigned_view"] != label:
                        park_data["poly_video"] = None
                        continue

                    points = np.array([park_data["kml_coords"][:, [1, 0]]], dtype='float32')
                    try:
                        video_poly = cv2.perspectiveTransform(points, active_H)[0].astype(int)

                        # Ekran disi kontrolu
                        if not np.all((video_poly[:, 0] > -200) & (video_poly[:, 0] < w + 200) &
                                      (video_poly[:, 1] > -200) & (video_poly[:, 1] < h + 200)):
                            park_data["poly_video"] = None
                            continue

                        if park_data["assigned_view"] is None:
                            park_data["assigned_view"] = label
                        park_data["poly_video"] = video_poly

                        if analiz_yapilsin_mi:
                            mask = np.zeros(gray.shape, dtype=np.uint8)
                            cv2.fillPoly(mask, [video_poly], 255)
                            count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                            rect_area = cv2.contourArea(video_poly)

                            if rect_area > 0:
                                doluluk_orani = count / rect_area
                                anlik_tespit = doluluk_orani > self.config.doluluk_esigi

                                if anlik_tespit:
                                    park_data["dolu_sayaci"] += 1
                                else:
                                    park_data["dolu_sayaci"] = 0

                                if park_data["dolu_sayaci"] >= self.config.onay_sayisi:
                                    if park_data["dolu_sayaci"] > 100:
                                        park_data["dolu_sayaci"] = 100
                                    yeni_durum = "DOLU"
                                else:
                                    yeni_durum = "BOS"

                                # Durum degisti mi?
                                if park_data["status"] != yeni_durum:
                                    park_data["status"] = yeni_durum
                                    park_data["color"] = (0, 0, 255) if yeni_durum == "DOLU" else (0, 255, 0)
                                    park_data["fill_hex"] = "#E74C3C" if yeni_durum == "DOLU" else "#27AE60"

                                    if on_status_change:
                                        on_status_change(
                                            park_data["id"],
                                            yeni_durum,
                                            self.get_status()
                                        )
                    except Exception:
                        park_data["poly_video"] = None

            # JSON kaydet
            self._save_json()

            # Pencere goster (opsiyonel)
            if self.config.show_window:
                for p in self._park_db:
                    if p["poly_video"] is not None:
                        cv2.polylines(img, [p["poly_video"]], True, p["color"], 2)
                cv2.imshow("Park Tespit", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._cap.release()
        if self.config.show_window:
            cv2.destroyAllWindows()
        print("Tespit dongusu durduruldu.")


# Basit kullanim icin fonksiyon
def detect_once(config: Optional[ParkConfig] = None) -> List[Dict[str, Any]]:
    """Tek seferlik tespit yap ve sonucu don"""
    detector = ParkDetector(config)
    detector._initialize()

    success, img = detector._cap.read()
    if not success:
        return []

    img = cv2.resize(img, detector.config.window_size)
    # ... tek frame analizi

    detector._cap.release()
    return detector.get_status()
