import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file jika ada
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# NASA FIRMS Config
NASA_FIRMS_MAP_KEY = os.getenv("NASA_FIRMS_MAP_KEY", "")
# Sensor: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, atau MODIS_NRT
FIRMS_DEFAULT_SENSOR = "VIIRS_NOAA20_NRT"

# Kalimantan Spatial Bounding Box (WGS84: MinLon, MinLat, MaxLon, MaxLat)
# Mencakup Kalbar, Kalteng, Kalsel, Kaltim, Kaltra
KALIMANTAN_BBOX = {
    "min_lon": 108.5,
    "min_lat": -4.3,
    "max_lon": 119.2,
    "max_lat": 4.5,
}

# Resolusi Grid: ~5 km (sekitar 0.045 derajat di ekuator)
GRID_RESOLUTION_DEG = 0.045  # ~5.0 km

# Batas Kategori Risiko (Skor 0 - 100)
RISK_THRESHOLDS = {
    "RENDAH": (0, 20),      # Aman, hijau
    "SEDANG": (20, 50),     # Waspada, kuning
    "TINGGI": (50, 100),    # Bahaya ekstrem, merah
}
