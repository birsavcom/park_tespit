import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import time
import os
import sys
import threading
import math
import gc
import queue
import atexit
import webbrowser
import torch
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ultralytics import YOLO

# =========================================================
# HEADLESS MOD
# =========================================================
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass

OPENCL_ACTIVE = False
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        OPENCL_ACTIVE = cv2.ocl.useOpenCL()
except Exception:
    OPENCL_ACTIVE = False

try:
    gc.enable()
    gc.set_threshold(1200, 20, 20)
except Exception:
    pass

# =========================================================
# 0) SİSTEM / GPU
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
device_arg = 0 if device == "cuda" else "cpu"
print(f">>> Sistem {device.upper()} uzerinde calisiyor.")
print(f">>> OpenCV OpenCL: {'AKTIF' if OPENCL_ACTIVE else 'PASIF'}")

if device == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

model = YOLO("yolo11l.pt").to(device)
try:
    model.fuse()
except Exception:
    pass

# =========================================================
# PLAKA DB (DEMO)
# =========================================================
KNOWN_PLATES = {
    5: "16 KCJ 5281"
}
VIP_ID = 5
vip_start_time = None

# =========================================================
# GLOBAL TRACKING (ARAÇ TEMELLİ GLOBAL ID)
# =========================================================
GLOBAL_TRACKS = {}        # gid -> track
PENDING_TRACKS = {}       # temp_key -> track (gid henüz yok)
TEMP_KEY_NEXT = -1
GLOBAL_NEXT_ID = 1

# Spot bağları (ID üretmez, sadece bağlar)
SPOT_OCCUPANT = {}        # spot_idx -> {"gid":gid, "last_frame":..., "last_center":(...)}
ID_TO_SPOT = {}           # gid -> spot_idx

# Park panel ve veri
GLOBAL_PARKED_REGISTRY = {}  # gid -> {"plate":..., "start_time":...}

# JSON optim
LAST_JSON_TEXT = None
JSON_WRITE_EVERY = 30  # ana dongu spike'larini azalt
SYSTEM_STATE_FILE = "system_state.json"

# Gercek-zaman akis
TARGET_FPS_OVERRIDE = 70.0  # canli akis ile uyumlu hedef
USE_ASYNC_JSON = True
ENABLE_FRAME_PACING = True
REALTIME_SOURCE_CLOCK = True
FRAME_GET_TIMEOUT = 0.08
READER_QUEUE_SIZE = 2

def set_system_state(is_running, reason=""):
    try:
        payload = {
            "running": bool(is_running),
            "reason": str(reason),
            "updated_at": int(time.time() * 1000),
        }
        with open(SYSTEM_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass

set_system_state(False, "booting")

# =========================================================
# 1) OTOMATİK BAŞLATMA
# =========================================================
def sistemi_otomatik_baslat():
    print("\n>>> [1/3] Sistem ve Dosyalar Kontrol Ediliyor...")
    if os.path.exists("kml_to_json.py"):
        print(">>> [2/3] Harita Verisi Donusturuluyor...")
        os.system("python kml_to_json.py")
    else:
        print(">>> [UYARI] 'kml_to_json.py' bulunamadi.")

    def sunucuyu_baslat():
        port = 3001
        try:
            from map_api import app as fastapi_app
            import uvicorn

            print(f">>> [3/3] FastAPI Aktif: http://localhost:{port}")
            cfg = uvicorn.Config(
                app=fastapi_app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )
            server = uvicorn.Server(cfg)
            server.run()
        except OSError:
            pass
        except Exception as e:
            print(f">>> [UYARI] FastAPI baslatilamadi ({e}). Klasik sunucuya geciliyor...")
            try:
                server_address = ("", port)

                class SessizHandler(SimpleHTTPRequestHandler):
                    def log_message(self, format, *args):
                        pass

                httpd = HTTPServer(server_address, SessizHandler)
                print(f">>> [3/3] Yerel Sunucu Aktif: http://localhost:{port}")
                httpd.serve_forever()
            except OSError:
                pass

    t = threading.Thread(target=sunucuyu_baslat, daemon=True)
    t.start()

    if not HEADLESS:
        time.sleep(1.2)
        url = "http://localhost:3001/"
        print(f">>> Tarayici Aciliyor: {url}\n")
        webbrowser.open(url)

sistemi_otomatik_baslat()

class LatestFrameGrabber:
    def __init__(self, source, queue_size=2, clock_fps=0.0, loop_file=False, enable_pacing=True):
        self.source = source
        self.loop_file = loop_file
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.queue = queue.Queue(maxsize=max(1, queue_size))
        self.stop_event = threading.Event()
        self.thread = None
        self.eos = False
        self.dropped = 0

        sfps = self.cap.get(cv2.CAP_PROP_FPS)
        if not sfps or sfps < 5 or sfps > 120:
            sfps = 25.0
        self.source_fps = float(sfps)

        fc = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.is_file_source = bool(fc and fc > 0)

        self.clock_fps = float(clock_fps) if clock_fps and clock_fps > 1.0 else self.source_fps
        self.frame_dt = 1.0 / max(1.0, self.clock_fps)
        self.use_pacing = bool(enable_pacing) and self.is_file_source and self.frame_dt > 0

    def is_opened(self):
        return bool(self.cap.isOpened())

    def start(self):
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return self

    def _reader_loop(self):
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.eos = True
                time.sleep(0.01)
                continue

            self.eos = False
            if self.use_pacing:
                next_tick += self.frame_dt
                remain = next_tick - time.perf_counter()
                if remain > 0:
                    time.sleep(remain)
                elif remain < -(self.frame_dt * 2.0):
                    next_tick = time.perf_counter()

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def get_latest(self, timeout=0.25):
        try:
            frame = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                frame = self.queue.get_nowait()
            except queue.Empty:
                break
        return frame

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()

class AsyncJsonWriter:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.pending_text = None
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, text):
        with self.lock:
            self.pending_text = text
        self.event.set()

    def _worker(self):
        while True:
            self.event.wait(timeout=0.5)
            self.event.clear()
            with self.lock:
                text = self.pending_text
                self.pending_text = None
            if self.stop_event.is_set() and text is None:
                break
            if text is None:
                continue
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        self.event.set()
        self.thread.join(timeout=1.0)

# =========================================================
# 2) AYARLAR
# =========================================================
FRAME_W, FRAME_H = 1280, 720

TOP_CUTOFF_RATIO = 0.35   # %35 üst / %65 alt (arka planda)
PLATE_LINE_RATIO = 0.50

# YOLO
YOLO_CLASSES = [2, 3, 5, 7]
YOLO_CONF = 0.45
YOLO_CONF_ARKA = 0.35
YOLO_CONF_ARKA_FAST = 0.30
YOLO_IOU = 0.55
YOLO_MAXDET = 64
YOLO_STRIDE = 1
RAW_MAP_TTL_FRAMES = 180
YOLO_ROI_MARGIN_ON = 220
YOLO_ROI_MARGIN_ARKA = 280

# Çizim/hold
PARK_BOX_HOLD_FRAMES = 12

# =========================================================
# BOX SMOOTHING (1 + 2)
# =========================================================
BOX_SMOOTH_ALPHA_BASE = 0.18
BOX_SMOOTH_ALPHA_MAX  = 0.85
BOX_SMOOTH_SPEED_REF  = 35.0

BOX_SNAP_IOU_THRESH   = 0.22
BOX_SNAP_CENTER_DIST  = 26.0
BOX_SNAP_AREA_RATIO   = 2.40

# =========================================================
# GLOBAL TRACK PARAMS (ID/BOX ATLAMASI ENGEL)
# =========================================================
NEW_TRACK_CONFIRM_HITS = 2          # gerçek araç için min isabet (ID üretmeden önce)
PENDING_MAX_MISSES = 3              # tentative track hızlı silinsin
TRACK_MAX_MISSES_MOVING = 10        # moving kaybolursa hızlı düş
TRACK_MAX_MISSES_PARKED = 999999    # park bağlı ID unutulmasın (spot BOS olunca temizleriz)
TRACK_PRUNE_TTL = 30000             # çok uzun kayıpları temizle (frame)
PARKED_STICK_FRAMES = 45            # park içi jitter ile spot dışına taşsa da parked say
EXIT_Y_FAST_DROP = 12               # ekran altından çıkışta hold yapma

# Eşleştirme kapıları
MATCH_MIN_IOU = 0.05
MATCH_MAX_DIST = 85                 # px
MATCH_MAX_DIST_PARKED = 120         # px
MATCH_SCORE_MIN = 0.12              # çok zayıf eşleşme kabul edilmesin

# Spot re-id: spotta kayıt var ama uzun süre görünmediyse körlemesine bağlama
SPOT_REID_GAP_FRAMES = 700          # ~10s @70fps
SPOT_FORCE_MIN_IOU = 0.08
SPOT_FORCE_MAX_DIST = 140

def choose_imgsz(roi_w, roi_h):
    m = max(roi_w, roi_h)
    if m < 640:
        return 416
    if m < 960:
        return 448
    return 512

# Park doluluk - park4 ile uyumlu
OCC_WEAK = 0.11
ONAY_SAYISI = 15
ANALYZE_EVERY = 3
OCC_SHARDS = 3
FAST_START_SECONDS = 1.0
FAST_START_ONAY = 1
VIEW_FAST_SECONDS = 3.0

# “Boşalan yer dolu takılı kalma” fix (temkinli)
CLEAR_LOW = 0.085
CLEAR_SHADOW = 0.13
BOS_ONAY = 10
YOLO_RECENT_TTL = 20
YOLO_MISS_CLEAR_FRAMES = 60

# Homography / ORB
MIN_MATCH_THRESH = 140
HOMOGRAPHY_INTERVAL = 30
ORB_TARGET_W = 640

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (a_area + b_area - inter + 1e-9))

def smooth_bbox_adaptive(prev_sb, curr_sb, speed_px):
    t = float(speed_px) / float(max(1e-6, BOX_SMOOTH_SPEED_REF))
    alpha = BOX_SMOOTH_ALPHA_BASE + (BOX_SMOOTH_ALPHA_MAX - BOX_SMOOTH_ALPHA_BASE) * clamp(t, 0.0, 1.0)
    b = 1.0 - alpha
    px1, py1, px2, py2 = prev_sb
    cx1, cy1, cx2, cy2 = curr_sb
    return (
        b * float(px1) + alpha * float(cx1),
        b * float(py1) + alpha * float(cy1),
        b * float(px2) + alpha * float(cx2),
        b * float(py2) + alpha * float(cy2),
    )

def should_snap(prev_sb, curr_sb, speed_px):
    if prev_sb is None:
        return True
    iou = bbox_iou(prev_sb, curr_sb)
    if iou < BOX_SNAP_IOU_THRESH:
        return True
    if float(speed_px) > BOX_SNAP_CENTER_DIST:
        return True
    px1, py1, px2, py2 = prev_sb
    cx1, cy1, cx2, cy2 = curr_sb
    p_area = max(1.0, (px2 - px1) * (py2 - py1))
    c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
    ratio = (c_area / p_area) if p_area > 0 else 999.0
    if ratio > BOX_SNAP_AREA_RATIO or ratio < (1.0 / BOX_SNAP_AREA_RATIO):
        return True
    return False

def center_of(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def dist2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy

def allocate_new_id():
    global GLOBAL_NEXT_ID
    gid = int(GLOBAL_NEXT_ID)
    GLOBAL_NEXT_ID += 1
    return gid

def bind_spot(spot_idx, gid, frame_idx, cx, cy):
    spot_idx = int(spot_idx)
    gid = int(gid)

    old_spot = ID_TO_SPOT.get(gid)
    if old_spot is not None and old_spot != spot_idx:
        SPOT_OCCUPANT.pop(old_spot, None)

    ID_TO_SPOT[gid] = spot_idx
    SPOT_OCCUPANT[spot_idx] = {
        "gid": gid,
        "last_frame": int(frame_idx),
        "last_center": (int(cx), int(cy)),
    }

def clear_spot(spot_idx):
    spot_idx = int(spot_idx)
    rec = SPOT_OCCUPANT.pop(spot_idx, None)
    if rec is not None:
        gid = int(rec.get("gid", -1))
        if gid in ID_TO_SPOT and ID_TO_SPOT[gid] == spot_idx:
            ID_TO_SPOT.pop(gid, None)

def track_is_moving(track):
    h = track.get("history")
    if not h or len(h) < 10:
        return True
    x0, y0 = h[0]
    x1, y1 = h[-1]
    d = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return d > 5.0

def make_track(view_label, bbox, cx, cy, raw_id, frame_idx, spot_idx):
    return {
        "confirmed": False,
        "gid": None,  # confirmed olunca dolar
        "view": str(view_label),
        "hits": 1,
        "age": 1,
        "last_seen": int(frame_idx),
        "missed": 0,
        "center": (int(cx), int(cy)),
        "v": (0.0, 0.0),
        "smooth_bbox": bbox,
        "last_bbox_area": float(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))),
        "history": deque(maxlen=20),
        "last_raw_id": int(raw_id) if raw_id is not None else -1,
        "spot": int(spot_idx) if spot_idx is not None else -1,
        "last_park_frame": int(frame_idx) if (spot_idx is not None and int(spot_idx) != -1) else -10**9,
    }

def track_update(track, det_bbox, cx, cy, raw_id, frame_idx, spot_idx, frame_w, frame_h):
    prev_center = track.get("center", (int(cx), int(cy)))
    speed_px = math.sqrt((int(cx) - int(prev_center[0])) ** 2 + (int(cy) - int(prev_center[1])) ** 2)

    prev_sb = track.get("smooth_bbox", None)
    curr_bbox = det_bbox

    snap_now = should_snap(prev_sb, curr_bbox, speed_px)
    if snap_now or prev_sb is None:
        sb = curr_bbox
    else:
        sb = smooth_bbox_adaptive(prev_sb, curr_bbox, speed_px)

    # Sınır güvenliği
    sx1, sy1, sx2, sy2 = sb
    sx1 = clamp(sx1, 0.0, float(frame_w - 2))
    sy1 = clamp(sy1, 0.0, float(frame_h - 2))
    sx2 = clamp(sx2, sx1 + 2.0, float(frame_w))
    sy2 = clamp(sy2, sy1 + 2.0, float(frame_h))
    sb = (sx1, sy1, sx2, sy2)

    # Basit hız vektörü (1 frame)
    vx = float(int(cx) - int(prev_center[0]))
    vy = float(int(cy) - int(prev_center[1]))

    track["age"] = int(track.get("age", 0)) + 1
    track["last_seen"] = int(frame_idx)
    track["missed"] = 0
    track["center"] = (int(cx), int(cy))
    track["v"] = (vx, vy)
    track["smooth_bbox"] = sb
    track["last_bbox_area"] = float(max(1.0, (sx2 - sx1) * (sy2 - sy1)))
    track["history"].append((int(cx), int(cy)))
    track["last_raw_id"] = int(raw_id) if raw_id is not None else track.get("last_raw_id", -1)

    if spot_idx is not None and int(spot_idx) != -1:
        track["spot"] = int(spot_idx)
        track["last_park_frame"] = int(frame_idx)
    else:
        # Park stick
        if (int(frame_idx) - int(track.get("last_park_frame", -10**9))) <= PARKED_STICK_FRAMES:
            pass
        else:
            track["spot"] = -1

def track_match_score(track, det, frame_idx):
    # yalnızca aynı view içinde
    if str(track.get("view", "")) != str(det["view"]):
        return None

    tb = track.get("smooth_bbox")
    if tb is None:
        return None

    iou = bbox_iou(tb, det["bbox"])
    tc = track.get("center", None)
    if tc is None:
        tc = center_of(tb)
    vx, vy = track.get("v", (0.0, 0.0))
    pred = (float(tc[0]) + float(vx), float(tc[1]) + float(vy))

    dc = det["center"]
    d = math.sqrt(dist2(pred, dc))

    # parkta ise biraz daha tolerans
    is_parked = (track.get("spot", -1) != -1) or ((frame_idx - track.get("last_park_frame", -10**9)) <= PARKED_STICK_FRAMES)
    max_d = MATCH_MAX_DIST_PARKED if is_parked else MATCH_MAX_DIST

    # Kapılar
    if iou < MATCH_MIN_IOU and d > max_d:
        return None
    if d > max_d * 1.6:
        return None

    d_norm = clamp(d / max(1.0, float(max_d)), 0.0, 1.0)
    score = (0.75 * iou) + (0.25 * (1.0 - d_norm))

    # aynı raw_id bonusu (kısa vadede swap engeli)
    if int(det.get("raw_id", -1)) != -1 and int(det.get("raw_id", -1)) == int(track.get("last_raw_id", -2)):
        score += 0.15

    # aynı spot bonusu
    if int(det.get("spot", -1)) != -1 and int(det.get("spot", -1)) == int(track.get("spot", -2)):
        score += 0.10

    return float(score)

# =========================================================
# KML / PARK VERİSİ
# =========================================================
def get_kml_coords(kml_file):
    coords_list = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        for pm in root.findall(".//kml:Placemark", ns):
            coord_text = pm.find(".//kml:coordinates", ns)
            if coord_text is not None:
                raw = coord_text.text.strip().split()
                coords_list.append(
                    np.array([[float(c.split(",")[1]), float(c.split(",")[0])] for c in raw])
                )
    except Exception:
        pass
    return coords_list

# =========================================================
# 3) HOMOGRAPHY NOKTALARI
# =========================================================
src_pts_on = np.array(
    [
        [40.226769, 28.845551],
        [40.226860, 28.845459],
        [40.226980, 28.845554],
        [40.227306, 28.845179],
        [40.226917, 28.845026],
        [40.227106, 28.845077],
    ]
)
dst_pts_on = np.array([[312, 680], [661, 469], [1193, 496], [1243, 301], [528, 284], [876, 283]])
H_ON, _ = cv2.findHomography(src_pts_on[:, [1, 0]], dst_pts_on, 0)

src_pts_arka = np.array(
    [
        [40.226661, 28.846109],
        [40.226470, 28.846383],
        [40.226419, 28.846297],
        [40.226381, 28.846253],
        [40.226563, 28.845982],
    ]
)
dst_pts_arka = np.array([[445, 505], [679, 268], [869, 272], [993, 278], [1153, 505]])
H_ARKA, _ = cv2.findHomography(src_pts_arka[:, [1, 0]], dst_pts_arka, 0)

orb = cv2.ORB_create(nfeatures=2000)
try:
    ref_on_img = cv2.imread("on_cephe_referans.jpg", 0)
    ref_arka_img = cv2.imread("arka_cephe_referans.jpg", 0)
    _, des1 = orb.detectAndCompute(ref_on_img, None)
    _, des2 = orb.detectAndCompute(ref_arka_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
except Exception:
    print("HATA: Referans resimler eksik!")
    sys.exit()

def get_active_homography(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w > ORB_TARGET_W:
        scale = ORB_TARGET_W / float(w)
        frame_small = cv2.resize(frame_bgr, (ORB_TARGET_W, int(h * scale)))
    else:
        frame_small = frame_bgr

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    _, des_f = orb.detectAndCompute(gray, None)
    if des_f is None:
        return None, "BELIRSIZ"
    m_on = len(bf.match(des1, des_f))
    m_arka = len(bf.match(des2, des_f))
    if max(m_on, m_arka) < MIN_MATCH_THRESH:
        return None, "GECIS"
    return (H_ON, "ON_CEPHE") if m_on > m_arka + 30 else (H_ARKA, "ARKA_CEPHE")

class AsyncHomographyWorker:
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pending_result = None
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame_bgr):
        if frame_bgr is None:
            return
        frame_copy = frame_bgr.copy()
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame_copy)
        except queue.Full:
            pass

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            result = get_active_homography(frame)
            with self.lock:
                self.pending_result = result

    def get_latest(self):
        with self.lock:
            result = self.pending_result
            self.pending_result = None
        return result

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

def compute_view_poly(kml_coords, H):
    pts = np.array([kml_coords[:, [1, 0]]], dtype="float32")
    try:
        poly = cv2.perspectiveTransform(pts, H)[0].astype(int)
        x_min, y_min = np.min(poly, axis=0)
        x_max, y_max = np.max(poly, axis=0)
        if x_max < 0 or x_min > FRAME_W or y_max < 0 or y_min > FRAME_H:
            return None
        poly[:, 0] = np.clip(poly[:, 0], 0, FRAME_W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, FRAME_H - 1)
        return poly
    except Exception:
        return None

def build_roi_from_polys(polys, margin=160):
    polys = [p for p in polys if p is not None]
    if not polys:
        return (0, 0, FRAME_W, FRAME_H)
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x1 = max(0, int(xs.min() - margin))
    y1 = max(0, int(ys.min() - margin))
    x2 = min(FRAME_W, int(xs.max() + margin))
    y2 = min(FRAME_H, int(ys.max() + margin))
    if (x2 - x1) < 80 or (y2 - y1) < 80:
        return (0, 0, FRAME_W, FRAME_H)
    return (x1, y1, x2, y2)

def precompute_crop_mask(poly):
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    x2, y2 = x + w, y + h
    x = max(0, x); y = max(0, y)
    x2 = min(FRAME_W, x2); y2 = min(FRAME_H, y2)
    if x2 <= x or y2 <= y:
        return None
    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    shifted = poly.copy()
    shifted[:, 0] -= x
    shifted[:, 1] -= y
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    area_contour = float(cv2.contourArea(poly.astype(np.int32)))
    mask_area_px = int(cv2.countNonZero(mask))
    if area_contour <= 0 or mask_area_px <= 0:
        return None
    return {"bbox": (x, y, x2, y2), "mask": mask, "area_contour": area_contour, "mask_area_px": mask_area_px}

def precompute_poly_bbox(poly):
    if poly is None:
        return None
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    return (x, y, x + w, y + h)

def masked_ratio_from_thresh_roi(thresh_roi, pack, ox, oy):
    bx1, by1, bx2, by2 = pack["bbox"]

    rx1 = bx1 - ox
    ry1 = by1 - oy
    rx2 = bx2 - ox
    ry2 = by2 - oy

    Ht, Wt = thresh_roi.shape[:2]
    ix1 = max(0, rx1); iy1 = max(0, ry1)
    ix2 = min(Wt, rx2); iy2 = min(Ht, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    mx1 = ix1 - rx1
    my1 = iy1 - ry1
    mx2 = mx1 + (ix2 - ix1)
    my2 = my1 + (iy2 - iy1)

    thr_crop = thresh_roi[iy1:iy2, ix1:ix2]
    mask_crop = pack["mask"][my1:my2, mx1:mx2]
    if thr_crop.size == 0 or mask_crop.size == 0:
        return None

    count = cv2.countNonZero(cv2.bitwise_and(thr_crop, thr_crop, mask=mask_crop))
    mask_nz = cv2.countNonZero(mask_crop)
    if mask_nz <= 0:
        return None

    clip_area = pack["area_contour"] * (mask_nz / float(pack["mask_area_px"]))
    if clip_area <= 0:
        return None

    return count / clip_area

# =========================================================
# 4) VIDEO / DB HAZIRLIK
# =========================================================
video_source = "gorukle.mp4"
json_writer = AsyncJsonWriter("otoparklar.json").start() if USE_ASYNC_JSON else None

probe = cv2.VideoCapture(video_source)
if not probe.isOpened():
    print(f"HATA: Video kaynagi acilamadi: {video_source}")
    sys.exit(1)
source_fps = probe.get(cv2.CAP_PROP_FPS)
if not source_fps or source_fps < 5 or source_fps > 120:
    source_fps = 25.0
probe.release()

target_fps = TARGET_FPS_OVERRIDE if TARGET_FPS_OVERRIDE > 0 else source_fps
clock_fps = min(source_fps, target_fps) if REALTIME_SOURCE_CLOCK else target_fps
grabber = LatestFrameGrabber(
    video_source,
    queue_size=READER_QUEUE_SIZE,
    clock_fps=clock_fps,
    loop_file=False,
    enable_pacing=ENABLE_FRAME_PACING,
).start()
homography_worker = AsyncHomographyWorker().start()
print(f">>> Video source: {video_source}")
print(f">>> Source FPS: {source_fps} | target_fps: {target_fps} | clock_fps: {clock_fps}")
print(f">>> Grabber opened: {grabber.is_opened()} | is_file_source: {grabber.is_file_source}")

if not grabber.is_opened():
    print(f"HATA: Video kaynagi acilamadi: {video_source}")
    sys.exit(1)

print(f">>> Hedef FPS: {target_fps:.1f}")

kml_parks = get_kml_coords("Birsav_akilli_park.kml")

# Poligonları 1 kez hesapla (HUGE hız)
polys_on = [compute_view_poly(k, H_ON) for k in kml_parks]
polys_arka = [compute_view_poly(k, H_ARKA) for k in kml_parks]

assigned_view = [None] * len(kml_parks)
for i in range(len(kml_parks)):
    p_on = polys_on[i]
    p_arka = polys_arka[i]
    if p_on is None and p_arka is None:
        assigned_view[i] = None
    elif p_on is None:
        assigned_view[i] = "ARKA_CEPHE"
    elif p_arka is None:
        assigned_view[i] = "ON_CEPHE"
    else:
        a_on = abs(cv2.contourArea(p_on.astype(np.int32)))
        a_arka = abs(cv2.contourArea(p_arka.astype(np.int32)))
        assigned_view[i] = "ON_CEPHE" if a_on >= a_arka else "ARKA_CEPHE"

packs_on = []
packs_arka = []
bboxes_on = []
bboxes_arka = []
for i in range(len(kml_parks)):
    if assigned_view[i] == "ON_CEPHE":
        packs_on.append(precompute_crop_mask(polys_on[i]) if polys_on[i] is not None else None)
        packs_arka.append(None)
        bboxes_on.append(precompute_poly_bbox(polys_on[i]) if polys_on[i] is not None else None)
        bboxes_arka.append(None)
    elif assigned_view[i] == "ARKA_CEPHE":
        packs_on.append(None)
        packs_arka.append(precompute_crop_mask(polys_arka[i]) if polys_arka[i] is not None else None)
        bboxes_on.append(None)
        bboxes_arka.append(precompute_poly_bbox(polys_arka[i]) if polys_arka[i] is not None else None)
    else:
        packs_on.append(None)
        packs_arka.append(None)
        bboxes_on.append(None)
        bboxes_arka.append(None)

# ON cephe sol dükkan/bina için ignore çizgisi
visible_on = [polys_on[i] for i in range(len(kml_parks)) if assigned_view[i] == "ON_CEPHE" and polys_on[i] is not None]
min_x_on = 0
if visible_on:
    min_x_on = min(int(np.min(p[:, 0])) for p in visible_on)
X_CUTOFF_ON = max(0, min_x_on - 40)
IGNORE_LEFT_X = X_CUTOFF_ON + 30

# YOLO ROI (hız)
YOLO_ROI_ON = build_roi_from_polys(visible_on, margin=YOLO_ROI_MARGIN_ON)
visible_arka = [polys_arka[i] for i in range(len(kml_parks)) if assigned_view[i] == "ARKA_CEPHE" and polys_arka[i] is not None]
YOLO_ROI_ARKA = build_roi_from_polys(visible_arka, margin=YOLO_ROI_MARGIN_ARKA)

# Occupancy ROI
OCC_ROI_ON = build_roi_from_polys(visible_on, margin=300)
OCC_ROI_ARKA = build_roi_from_polys(visible_arka, margin=300)

# Park DB
global_park_db = []
for i, kml_poly in enumerate(kml_parks):
    global_park_db.append(
        {
            "id": i,
            "status": "BOS",
            "prev_status": "BOS",
            "color": (0, 255, 0),
            "fill_hex": "#27AE60",
            "kml_coords": kml_poly,
            "coords_json": [{"lat": float(pt[0]), "lng": float(pt[1])} for pt in kml_poly],
            "poly_video": None,
            "mask_pack": None,
            "poly_bbox": None,
            "dolu_sayaci": 0,
            "bos_sayaci": 0,
            "last_vehicle_seen": -10**9,
        }
    )

# GPU warmup
if device == "cuda":
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            _ = model.predict(dummy, imgsz=640, device=device_arg, half=True, verbose=False)
        torch.cuda.synchronize()
    except Exception:
        pass

# =========================================================
# 5) ANA DÖNGÜ
# =========================================================
frame_sayac = 0
son_label = "BELIRSIZ"
active_H = None
last_yolo_results = None

last_view_label = None
prev_son_label = None
view_fast_until = 0.0

dyn_imgsz = 512
loop_t0 = time.perf_counter()
shutdown_reason = "stopped"
set_system_state(True, "running")
atexit.register(set_system_state, False, "process_exit")

while True:
    img = grabber.get_latest(timeout=FRAME_GET_TIMEOUT)

    # --- KESİN KORUMA: img None ise asla aşağı akma ---
    if img is None:
        # Debug: her 30 denemede bir yaz (konsolu boğmasın)
        if (frame_sayac % 30) == 0:
            print(f"[FRAME] None geldi | eos={grabber.eos} | dropped={getattr(grabber,'dropped',-1)}")

        if grabber.eos:
            print(">>> Video bitti. Sistem kapaniyor.")
            shutdown_reason = "video_finished"
            break

        # Kuyruk boş -> bir sonraki turu dene
        continue

    # Buradan sonrası: img artık kesinlikle dolu
    img = cv2.resize(img, (FRAME_W, FRAME_H))
    h, w = img.shape[:2]


    limit_line_y = int(h * TOP_CUTOFF_RATIO)
    plate_read_line = int(h * PLATE_LINE_RATIO)

    frame_sayac += 1
    now_ts = time.perf_counter()
    fast_start_mode = (now_ts - loop_t0) <= FAST_START_SECONDS

    # Cephe kontrol (seyrek, asenkron)
    if frame_sayac == 1 or frame_sayac % HOMOGRAPHY_INTERVAL == 1:
        homography_worker.submit(img)

    h_result = homography_worker.get_latest()
    if h_result is not None:
        new_H, new_label = h_result
        if new_H is not None:
            active_H, son_label = new_H, new_label
        elif new_label == "BELIRSIZ":
            active_H, son_label = None, new_label
        # GECIS ise son_label korunur

    if prev_son_label is None and son_label in ("ON_CEPHE", "ARKA_CEPHE"):
        prev_son_label = son_label
        view_fast_until = now_ts + VIEW_FAST_SECONDS
    elif son_label in ("ON_CEPHE", "ARKA_CEPHE") and prev_son_label != son_label:
        view_fast_until = now_ts + VIEW_FAST_SECONDS
        prev_son_label = son_label
    elif son_label in ("ON_CEPHE", "ARKA_CEPHE"):
        prev_son_label = son_label

    view_fast_mode = now_ts < view_fast_until

    # Aktif poligon/mask ve ROI seç
    if son_label == "ON_CEPHE":
        yolo_roi = YOLO_ROI_ON
        occ_roi = OCC_ROI_ON
        if last_view_label != son_label:
            for p in global_park_db:
                pid = p["id"]
                if assigned_view[pid] == "ON_CEPHE":
                    p["poly_video"] = polys_on[pid]
                    p["mask_pack"] = packs_on[pid]
                    p["poly_bbox"] = bboxes_on[pid]
                else:
                    p["poly_video"] = None
                    p["mask_pack"] = None
                    p["poly_bbox"] = None
            last_view_label = son_label
    elif son_label == "ARKA_CEPHE":
        yolo_roi = YOLO_ROI_ARKA
        occ_roi = OCC_ROI_ARKA
        if last_view_label != son_label:
            for p in global_park_db:
                pid = p["id"]
                if assigned_view[pid] == "ARKA_CEPHE":
                    p["poly_video"] = polys_arka[pid]
                    p["mask_pack"] = packs_arka[pid]
                    p["poly_bbox"] = bboxes_arka[pid]
                else:
                    p["poly_video"] = None
                    p["mask_pack"] = None
                    p["poly_bbox"] = None
            last_view_label = son_label
    else:
        yolo_roi = (0, 0, w, h)
        occ_roi = (0, 0, w, h)
        if last_view_label != son_label:
            for p in global_park_db:
                p["poly_video"] = None
                p["mask_pack"] = None
                p["poly_bbox"] = None
            last_view_label = son_label

    # YOLO ROI kırp
    rx1, ry1, rx2, ry2 = yolo_roi
    rx1 = max(0, min(rx1, w - 1))
    ry1 = max(0, min(ry1, h - 1))
    rx2 = max(rx1 + 1, min(rx2, w))
    ry2 = max(ry1 + 1, min(ry2, h))

    roi_img = img[ry1:ry2, rx1:rx2]
    if roi_img.size == 0:
        rx1, ry1, rx2, ry2 = (0, 0, w, h)
        roi_img = img

    # =====================================================
    # YOLO TRACK
    # =====================================================
    yolo_update = (frame_sayac % YOLO_STRIDE == 0) or (last_yolo_results is None)
    roi_h, roi_w = roi_img.shape[:2]
    dyn_imgsz = choose_imgsz(roi_w, roi_h)
    if son_label == "ARKA_CEPHE":
        yolo_conf_now = YOLO_CONF_ARKA_FAST if view_fast_mode else YOLO_CONF_ARKA
    else:
        yolo_conf_now = YOLO_CONF

    if yolo_update:
        try:
            with torch.inference_mode():
                last_yolo_results = model.track(
                    roi_img,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=YOLO_CLASSES,
                    conf=yolo_conf_now,
                    iou=YOLO_IOU,
                    imgsz=dyn_imgsz,
                    max_det=YOLO_MAXDET,
                    verbose=False,
                    device=device_arg,
                    half=(device == "cuda"),
                )
        except TypeError:
            with torch.inference_mode():
                last_yolo_results = model.track(
                    roi_img,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=YOLO_CLASSES,
                    conf=yolo_conf_now,
                    imgsz=dyn_imgsz,
                    verbose=False,
                )

    results = last_yolo_results if last_yolo_results is not None else []
    yolo_iter = results if isinstance(results, (list, tuple)) else [results]

    # =====================================================
    # DETECTIONS -> GLOBAL TRACK MATCH
    # =====================================================
    detections = []
    for result in yolo_iter:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        box_coords = boxes.xyxy.cpu().numpy().astype(float)
        raw_ids = boxes.id.cpu().numpy().astype(int)
        confs = None
        try:
            confs = boxes.conf.cpu().numpy().astype(float)
        except Exception:
            confs = np.ones((len(raw_ids),), dtype=float)

        for box, r_id, cf in zip(box_coords, raw_ids, confs):
            x1, y1, x2, y2 = box
            gx1, gy1 = x1 + rx1, y1 + ry1
            gx2, gy2 = x2 + rx1, y2 + ry1

            cx, cy = 0.5 * (gx1 + gx2), 0.5 * (gy1 + gy2)

            # ON cephe sol dükkan/bina ignore
            if son_label == "ON_CEPHE" and cx < IGNORE_LEFT_X:
                continue

            # Park içi spot tespiti
            spot_index = -1
            for p_idx, p_data in enumerate(global_park_db):
                poly = p_data["poly_video"]
                if poly is None:
                    continue
                bbox = p_data["poly_bbox"]
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    if cx < bx1 or cx > bx2 or cy < by1 or cy > by2:
                        continue
                is_inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
                if is_inside >= 0:
                    spot_index = p_idx
                    break

            # Yeni track açma gate:
            # - park içindeyse her yerde olur
            # - park dışındaysa sadece alt bölgede
            if spot_index == -1 and cy <= limit_line_y:
                continue

            det_bbox = (float(gx1), float(gy1), float(gx2), float(gy2))
            detections.append({
                "bbox": det_bbox,
                "center": (float(cx), float(cy)),
                "raw_id": int(r_id),
                "conf": float(cf),
                "spot": int(spot_index),
                "view": str(son_label),
            })

    # Basit dedup (aynı araca 2 bbox çıkarsa yeni ID fırlamasın)
    detections.sort(key=lambda d: d["conf"], reverse=True)
    deduped = []
    for d in detections:
        keep = True
        for k in deduped:
            if bbox_iou(d["bbox"], k["bbox"]) > 0.75:
                keep = False
                break
        if keep:
            deduped.append(d)
    detections = deduped

    # 1) SPOT OCCUPANT ile ön atama (park ID unutulmasın)
    assigned_det_idx = set()
    assigned_tracks = {}  # track_key (gid veya temp_key) -> det_idx
    used_track_keys = set()

    def get_track_by_key(key):
        if key in GLOBAL_TRACKS:
            return GLOBAL_TRACKS[key]
        if key in PENDING_TRACKS:
            return PENDING_TRACKS[key]
        return None

    for i, det in enumerate(detections):
        spot = int(det["spot"])
        if spot == -1:
            continue
        rec = SPOT_OCCUPANT.get(spot)
        if not rec:
            continue
        gid = int(rec.get("gid", -1))
        tr = GLOBAL_TRACKS.get(gid)
        if tr is None:
            # eski kayıt yetim -> temizle
            clear_spot(spot)
            continue

        gap = frame_sayac - int(tr.get("last_seen", -10**9))
        if gap > TRACK_PRUNE_TTL:
            clear_spot(spot)
            continue

        # Spot zorlaması: çok uzun kayıp varsa, benzemiyorsa zorlamayalım
        iou = bbox_iou(tr.get("smooth_bbox", det["bbox"]), det["bbox"])
        tc = tr.get("center", center_of(tr.get("smooth_bbox", det["bbox"])))
        d = math.sqrt(dist2((float(tc[0]), float(tc[1])), det["center"]))

        if gap > SPOT_REID_GAP_FRAMES:
            if iou < SPOT_FORCE_MIN_IOU and d > SPOT_FORCE_MAX_DIST:
                # Muhtemelen spot değişmiş/araç değişmiş -> spotu serbest bırak
                clear_spot(spot)
                continue

        # Aynı view şartı (front/back koordinat karışmasın)
        if str(tr.get("view")) != str(det["view"]):
            continue

        if gid not in used_track_keys and i not in assigned_det_idx:
            assigned_tracks[gid] = i
            used_track_keys.add(gid)
            assigned_det_idx.add(i)

    # 2) Greedy eşleştirme (kalan detler -> mevcut trackler)
    # Aday trackler (aynı view)
    candidate_keys = []
    for gid, tr in GLOBAL_TRACKS.items():
        if str(tr.get("view", "")) != str(son_label):
            continue
        # çok eski ise match'e sokma (ama prune ayrı)
        if (frame_sayac - int(tr.get("last_seen", -10**9))) > TRACK_PRUNE_TTL:
            continue
        candidate_keys.append(gid)

    for tk, tr in PENDING_TRACKS.items():
        if str(tr.get("view", "")) != str(son_label):
            continue
        candidate_keys.append(tk)

    pairs = []
    for tkey in candidate_keys:
        if tkey in used_track_keys:
            continue
        tr = get_track_by_key(tkey)
        if tr is None:
            continue
        for i, det in enumerate(detections):
            if i in assigned_det_idx:
                continue
            sc = track_match_score(tr, det, frame_sayac)
            if sc is None:
                continue
            if sc < MATCH_SCORE_MIN:
                continue
            pairs.append((sc, tkey, i))

    pairs.sort(key=lambda x: x[0], reverse=True)
    for sc, tkey, i in pairs:
        if tkey in used_track_keys:
            continue
        if i in assigned_det_idx:
            continue
        assigned_tracks[tkey] = i
        used_track_keys.add(tkey)
        assigned_det_idx.add(i)

    # 3) Unmatched det -> yeni PENDING track
for i, det in enumerate(detections):
    if i in assigned_det_idx:
        continue
    temp_key = int(TEMP_KEY_NEXT)
    TEMP_KEY_NEXT -= 1
    cx, cy = det["center"]
    tr = make_track(det["view"], det["bbox"], cx, cy, det["raw_id"], frame_sayac, det["spot"])
    PENDING_TRACKS[temp_key] = tr
    assigned_tracks[temp_key] = i
    used_track_keys.add(temp_key)
    assigned_det_idx.add(i)


    # 4) Matched track update + confirm
    current_frame_parked = []
    current_frame_moving = []
    seen_confirmed_ids = set()

    # önce matched olanları güncelle
    for tkey, det_idx in list(assigned_tracks.items()):
        det = detections[det_idx]
        cx, cy = det["center"]
        tr = get_track_by_key(tkey)
        if tr is None:
            continue

        # update
        track_update(tr, det["bbox"], cx, cy, det["raw_id"], frame_sayac, det["spot"], w, h)

        # confirm?
        if not tr.get("confirmed", False):
            tr["hits"] = int(tr.get("hits", 0)) + 1
            if tr["hits"] >= NEW_TRACK_CONFIRM_HITS:
                gid = allocate_new_id()
                tr["confirmed"] = True
                tr["gid"] = int(gid)
                # pending -> global taşı
                PENDING_TRACKS.pop(tkey, None)
                GLOBAL_TRACKS[gid] = tr
                # track key değişti; assigned_tracks içinde eski key kalabilir ama artık önemli değil
                tkey = gid

        if tr.get("confirmed", False):
            gid = int(tr["gid"])
            seen_confirmed_ids.add(gid)

            # Spot/park güncelle
            spot = int(tr.get("spot", -1))
            # Park say
            if spot != -1:
                global_park_db[spot]["last_vehicle_seen"] = frame_sayac
                bind_spot(spot, gid, frame_sayac, tr["center"][0], tr["center"][1])
                current_frame_parked.append(gid)

                # Park registry (plaka demo)
                park_plate = KNOWN_PLATES.get(gid, "[...]")
                if gid == VIP_ID:
                    if vip_start_time is None:
                        vip_start_time = time.time()
                    GLOBAL_PARKED_REGISTRY[gid] = {"plate": KNOWN_PLATES.get(gid, park_plate), "start_time": vip_start_time}
                else:
                    if gid not in GLOBAL_PARKED_REGISTRY:
                        GLOBAL_PARKED_REGISTRY[gid] = {"plate": park_plate, "start_time": None}
                    if park_plate != "[...]":
                        GLOBAL_PARKED_REGISTRY[gid]["plate"] = park_plate
            else:
                # hareketli liste
                if float(cy) > limit_line_y and track_is_moving(tr):
                    current_frame_moving.append(gid)

    # 5) Unmatched tracks miss++ ve prune/hold kararları
    # Pending tracks
    for tk in list(PENDING_TRACKS.keys()):
        tr = PENDING_TRACKS[tk]
        if tk in assigned_tracks:
            continue
        tr["missed"] = int(tr.get("missed", 0)) + 1
        if tr["missed"] > PENDING_MAX_MISSES:
            PENDING_TRACKS.pop(tk, None)

    # Global tracks
    for gid in list(GLOBAL_TRACKS.keys()):
        tr = GLOBAL_TRACKS[gid]
        if gid in used_track_keys:
            continue

        tr["missed"] = int(tr.get("missed", 0)) + 1

        # Park bağlı ise unutma (spot BOS olunca temizlenecek)
        spot = int(tr.get("spot", -1))
        is_parked = (spot != -1) or ((frame_sayac - int(tr.get("last_park_frame", -10**9))) <= PARKED_STICK_FRAMES)

        # Moving için hızlı prune
        if not is_parked and tr["missed"] > TRACK_MAX_MISSES_MOVING:
            # spot bağını temizle
            old_spot = ID_TO_SPOT.get(gid)
            if old_spot is not None:
                clear_spot(old_spot)
            GLOBAL_TRACKS.pop(gid, None)
            GLOBAL_PARKED_REGISTRY.pop(gid, None)
            continue

        # Çok eski kayıtları temizle (memory)
        if (frame_sayac - int(tr.get("last_seen", -10**9))) > TRACK_PRUNE_TTL and not is_parked:
            old_spot = ID_TO_SPOT.get(gid)
            if old_spot is not None:
                clear_spot(old_spot)
            GLOBAL_TRACKS.pop(gid, None)
            GLOBAL_PARKED_REGISTRY.pop(gid, None)

    # =====================================================
    # ÇİZİM (box titreme az + ID/box çiftlenme engelli)
    # =====================================================
    # 1) Bu frame görünen (missed==0) confirmed trackleri çiz
    for gid, tr in GLOBAL_TRACKS.items():
        if str(tr.get("view", "")) != str(son_label):
            continue

        missed = int(tr.get("missed", 0))
        if missed != 0:
            continue

        sb = tr.get("smooth_bbox")
        if sb is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in sb]
        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(x1 + 2, min(x2, w))
        y2 = max(y1 + 2, min(y2, h))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"ID: {gid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2) Hold çizimi (kısa miss), ama ekran altından çıkanlarda hold yok
    for gid, tr in GLOBAL_TRACKS.items():
        if str(tr.get("view", "")) != str(son_label):
            continue
        missed = int(tr.get("missed", 0))
        if missed <= 0 or missed > PARK_BOX_HOLD_FRAMES:
            continue

        sb = tr.get("smooth_bbox")
        if sb is None:
            continue

        hx1, hy1, hx2, hy2 = [int(v) for v in sb]
        # Ekran altından çıkış: box araçla birlikte kaybolsun (takılma yapmasın)
        if hy2 >= (h - EXIT_Y_FAST_DROP):
            continue

        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 240, 240), 2)
        cv2.putText(img, f"ID: {gid}", (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 240), 2)

    # =====================================================
    # PARK DOLULUK (ROI threshold)
    # =====================================================
    analyze_every_now = 1 if (fast_start_mode or view_fast_mode) else ANALYZE_EVERY
    occ_shards_now = 1 if (fast_start_mode or view_fast_mode) else OCC_SHARDS
    onay_now = FAST_START_ONAY if (fast_start_mode or view_fast_mode) else ONAY_SAYISI

    if (frame_sayac % analyze_every_now == 0) and son_label in ("ON_CEPHE", "ARKA_CEPHE"):
        if img is None:
            print("[FATAL] img None iken OCC bloğuna girildi. Yukarıdaki guard yanlış konumda!")
            continue
        ox1, oy1, ox2, oy2 = occ_roi
        ox1 = max(0, min(ox1, w - 1))
        oy1 = max(0, min(oy1, h - 1))
        ox2 = max(ox1 + 1, min(ox2, w))
        oy2 = max(oy1 + 1, min(oy2, h))

        occ_bgr = img[oy1:oy2, ox1:ox2]
        if occ_bgr.size == 0:
            thresh = None
        else:
            if OPENCL_ACTIVE:
                occ_u = cv2.UMat(occ_bgr)
                gray_u = cv2.cvtColor(occ_u, cv2.COLOR_BGR2GRAY)
                blur_u = cv2.GaussianBlur(gray_u, (5, 5), 1)
                thresh = cv2.adaptiveThreshold(
                    blur_u,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    25,
                    16,
                ).get()
            else:
                gray = cv2.cvtColor(occ_bgr, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(
                    cv2.GaussianBlur(gray, (5, 5), 1),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    25,
                    16,
                )

        should_write_json = (fast_start_mode or view_fast_mode) or (frame_sayac % JSON_WRITE_EVERY == 0)
        json_output = [] if should_write_json else None
        occ_shard = (frame_sayac // analyze_every_now) % occ_shards_now

        for p in global_park_db:
            poly = p["poly_video"]
            pack = p["mask_pack"]
            should_analyze_spot = (p["id"] % occ_shards_now) == occ_shard

            if poly is None or pack is None or thresh is None or (not should_analyze_spot):
                if should_write_json:
                    json_output.append({
                        "id": p["id"],
                        "status": p["status"],
                        "fill": p["fill_hex"],
                        "coords": p["coords_json"],
                    })
                continue

            ratio = masked_ratio_from_thresh_roi(thresh, pack, ox1, oy1)
            if ratio is not None:
                # dolu sayacı
                if ratio > OCC_WEAK:
                    p["dolu_sayaci"] += 1
                else:
                    p["dolu_sayaci"] = 0

                yolo_recent = (frame_sayac - p["last_vehicle_seen"]) <= YOLO_RECENT_TTL
                yolo_miss_long = (frame_sayac - p["last_vehicle_seen"]) >= YOLO_MISS_CLEAR_FRAMES

                if p["status"] == "DOLU":
                    if (not yolo_recent) and (ratio < CLEAR_LOW):
                        p["bos_sayaci"] += 2
                    elif yolo_miss_long and (ratio < CLEAR_SHADOW):
                        p["bos_sayaci"] += 1
                    else:
                        p["bos_sayaci"] = max(0, p["bos_sayaci"] - 1)
                else:
                    p["bos_sayaci"] = 0

                if p["status"] == "DOLU":
                    yeni_durum = "DOLU"
                    if p["bos_sayaci"] >= BOS_ONAY:
                        yeni_durum = "BOS"
                        p["dolu_sayaci"] = 0
                        p["bos_sayaci"] = 0
                else:
                    yeni_durum = "DOLU" if p["dolu_sayaci"] >= onay_now else "BOS"

                # Spot BOS olduysa occupant temizle (ID başka araca yapışmasın)
                prev = p.get("prev_status", p["status"])
                p["prev_status"] = yeni_durum
                if prev == "DOLU" and yeni_durum == "BOS":
                    clear_spot(p["id"])

                p["status"] = yeni_durum
                p["color"] = (0, 0, 255) if yeni_durum == "DOLU" else (0, 255, 0)
                p["fill_hex"] = "#E74C3C" if yeni_durum == "DOLU" else "#27AE60"

            if should_write_json:
                json_output.append({
                    "id": p["id"],
                    "status": p["status"],
                    "fill": p["fill_hex"],
                    "coords": p["coords_json"],
                })

        if should_write_json and json_output is not None:
            json_text = json.dumps(json_output, ensure_ascii=False)
            if json_text != LAST_JSON_TEXT:
                if json_writer is not None:
                    json_writer.submit(json_text)
                else:
                    with open("otoparklar.json", "w", encoding="utf-8") as f:
                        f.write(json_text)
                LAST_JSON_TEXT = json_text

    # =====================================================
    # POLY ÇİZ
    # =====================================================
    for p in global_park_db:
        if p["poly_video"] is not None:
            cv2.polylines(img, [p["poly_video"]], True, p["color"], 2)

    # =====================================================
    # PANELLER
    # =====================================================
    overlay = img.copy()

    # SOL ALT PANEL: PARK HALİNDE
    cv2.rectangle(overlay, (0, h - 250), (330, h), (0, 0, 0), -1)
    cv2.putText(overlay, "PARK HALINDE", (10, h - 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y_pos = h - 180
    if VIP_ID in GLOBAL_PARKED_REGISTRY:
        elapsed = int(time.time() - vip_start_time) if vip_start_time else 0
        timer_str = f"{elapsed // 60:02d}:{elapsed % 60:02d} dk"
        cv2.putText(
            overlay,
            f"ID: {VIP_ID} | {GLOBAL_PARKED_REGISTRY[VIP_ID]['plate']} | {timer_str}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_pos += 25

    for v_id in sorted(set(current_frame_parked)):
        if v_id == VIP_ID or y_pos > h - 20:
            continue
        p_text = GLOBAL_PARKED_REGISTRY.get(v_id, {}).get("plate", "[...]")
        cv2.putText(overlay, f"ID: {v_id} | {p_text}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25

    # SAĞ ALT PANEL: OKUNAN PLAKA
    cv2.rectangle(overlay, (w - 300, h - 250), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, "OKUNAN PLAKA", (w - 290, h - 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    row_idx = 0
    for v_id in sorted(set(current_frame_moving))[:8]:
        tr = GLOBAL_TRACKS.get(v_id)
        if tr is None:
            continue
        cy = tr.get("center", (0, 0))[1]
        display_plate = KNOWN_PLATES.get(v_id, "[...]") if cy > plate_read_line else "[...]"
        cv2.putText(overlay, f"ID: {v_id} | {display_plate}", (w - 290, h - 180 + row_idx * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        row_idx += 1

    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Üst bilgi: sadece cephe etiketi
    if son_label == "ON_CEPHE":
        view_text = "ON CEPHE"
        text_color = (0, 255, 0)
    elif son_label == "ARKA_CEPHE":
        view_text = "ARKA CEPHE"
        text_color = (0, 255, 255)
    else:
        view_text = ""
        text_color = (200, 200, 200)

    if view_text:
        cv2.rectangle(img, (0, 0), (340, 46), (0, 0, 0), -1)
        cv2.putText(img, view_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    if not HEADLESS:
        cv2.imshow("Birsav AI - Master Otonom", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            shutdown_reason = "user_exit"
            break

set_system_state(False, shutdown_reason)
homography_worker.stop()
grabber.stop()
if json_writer is not None:
    json_writer.stop()
if not HEADLESS:
    cv2.destroyAllWindows()
