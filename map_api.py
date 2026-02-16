from __future__ import annotations

import json
from pathlib import Path
from os import getenv
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse


DEBUG = getenv("DEBUG", "false").lower() == "true"
print(DEBUG)
BASE_DIR = Path(__file__).resolve().parent
MAP_HTML = BASE_DIR / "otopark_demo.html"
PANEL_HTML = BASE_DIR / "panel.html"
PARKING_JSON = BASE_DIR / "otoparklar.json"
SYSTEM_STATE_JSON = BASE_DIR / "system_state.json"


app = FastAPI(
    title="Birsav Park API",
    version="1.0.0",
    description="Harita ve park verileri icin FastAPI servisi",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_file_response(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadi: {path.name}")
    return FileResponse(
        path=str(path),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    # URL'e girildiginde harita otomatik acilsin.
    return RedirectResponse(url="/map", status_code=307)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/map", include_in_schema=False)
def map_page():
    if not MAP_HTML.exists():
        raise HTTPException(status_code=404, detail="otopark_demo.html bulunamadi")
    return FileResponse(str(MAP_HTML), media_type="text/html")


@app.get("/panel", include_in_schema=False)
def panel_page():
    if not PANEL_HTML.exists():
        raise HTTPException(status_code=404, detail="panel.html bulunamadi")
    return FileResponse(str(PANEL_HTML), media_type="text/html")


@app.get("/otopark_demo.html", include_in_schema=False)
def map_page_legacy():
    return map_page()


@app.get("/panel.html", include_in_schema=False)
def panel_page_legacy():
    return panel_page()


@app.get("/otoparklar.json", include_in_schema=False)
def parking_json_file():
    return _json_file_response(PARKING_JSON)


@app.get("/system_state.json", include_in_schema=False)
def system_state_json_file():
    return {"running": True, "reason": "api_active", "updated_at": int(__import__("time").time() * 1000)}


@app.get("/api/parking-status")
def api_parking_status():
    if not PARKING_JSON.exists():
        raise HTTPException(status_code=404, detail="otoparklar.json bulunamadi")
    with PARKING_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/system-state")
def api_system_state():
    if not SYSTEM_STATE_JSON.exists():
        raise HTTPException(status_code=404, detail="system_state.json bulunamadi")
    with SYSTEM_STATE_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)
