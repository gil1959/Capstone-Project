"""
Downloader Cuaca & Kelembapan Tanah dari Open-Meteo API.
Mendukung:
1. Data Historis (Archive API) untuk melatih model (2022-2023).
2. Data Real-Time / Ramalan (Forecast API) untuk inferensi peringatan dini harian.
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

WEATHER_VARIABLES = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "vapour_pressure_deficit_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]

class WeatherDownloader:
    def __init__(self):
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"

    def fetch_batch_weather(
        self,
        grid_coords: list, # list of tuples: (grid_id, lat, lon)
        start_date: str,
        end_date: str,
        is_forecast: bool = False,
        batch_size: int = 50,
        cache_dir: Path = None
    ) -> pd.DataFrame:
        """
        Mengunduh data cuaca & tanah secara batch (50 grid per request)
        dengan dukungan auto-checkpointing / cache per batch.
        """
        base_url = self.forecast_url if is_forecast else self.archive_url
        daily_vars = ",".join(WEATHER_VARIABLES)
        all_records = []

        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        total_batches = (len(grid_coords) + batch_size - 1) // batch_size
        print(f"[INFO] Mengunduh cuaca untuk {len(grid_coords):,} grid ({start_date} s/d {end_date}) dalam {total_batches} batch...")

        for batch_idx, i in enumerate(range(0, len(grid_coords), batch_size), start=1):
            batch_file = cache_dir / f"batch_{batch_idx:05d}.parquet" if cache_dir else None
            
            # Cek jika cache batch sudah ada
            if batch_file and batch_file.exists():
                df_b = pd.read_parquet(batch_file)
                all_records.extend(df_b.to_dict(orient="records"))
                continue

            batch = grid_coords[i:i + batch_size]
            lats_str = ",".join(str(item[1]) for item in batch)
            lons_str = ",".join(str(item[2]) for item in batch)

            params = {
                "latitude": lats_str,
                "longitude": lons_str,
                "daily": daily_vars,
                "timezone": "Asia/Jakarta",
            }
            if not is_forecast:
                params["start_date"] = start_date
                params["end_date"] = end_date

            batch_records = []
            max_retries = 10

            for attempt in range(max_retries):
                try:
                    resp = requests.get(base_url, params=params, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else [data]

                        for idx_item, res in enumerate(items):
                            if idx_item < len(batch):
                                gid = batch[idx_item][0]
                                daily = res.get("daily", {})
                                times = daily.get("time", [])

                                for t_idx, dt in enumerate(times):
                                    record = {
                                        "grid_id": gid,
                                        "date": dt,
                                        "temp_max": daily.get("temperature_2m_max", [None])[t_idx],
                                        "rain_sum": daily.get("precipitation_sum", [None])[t_idx],
                                        "wind_speed_max": daily.get("wind_speed_10m_max", [None])[t_idx],
                                        "vpd_max": daily.get("vapour_pressure_deficit_max", [None])[t_idx],
                                        "soil_moisture_0_7cm": daily.get("soil_moisture_0_to_7cm_mean", [None])[t_idx],
                                        "soil_moisture_7_28cm": daily.get("soil_moisture_7_to_28cm_mean", [None])[t_idx],
                                    }
                                    batch_records.append(record)
                        break
                    elif resp.status_code == 429:
                        err_text = resp.text.lower()
                        if "daily" in err_text:
                            print(f"\n[RATE LIMIT 429 - DAILY] Kuota HARIAN Open-Meteo untuk IP ini telah habis (10.000 limit tercapai).")
                            print(f"[SOLUSI CEPAT] Ganti koneksi sementara (misal: sambungkan ke Hotspot HP / Tethering / VPN), lalu jalankan ulang.")
                            print(f"[INFO] Karena titik cuaca kita sekarang hanya 16 batch, dengan IP baru prosesnya selesai dalam 40 detik!")
                            raise RuntimeError("Open-Meteo Daily Limit Exceeded. Silakan ganti IP (Hotspot HP / Tethering) untuk melanjutkan.")
                        elif "hourly" in err_text:
                            now_utc = datetime.now(timezone.utc)
                            next_hour_utc = (now_utc + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
                            wait_sec = max(15, int((next_hour_utc - now_utc).total_seconds()))
                            next_local = datetime.now() + timedelta(seconds=wait_sec)
                            print(f"\n[RATE LIMIT 429] Kuota per jam Open-Meteo habis pada batch {batch_idx}.")
                            print(f"[RESET OTOMATIS] Sesuai dokumentasi Open-Meteo, kuota reset pada {next_local.strftime('%H:%M:%S')} WIB.")
                            print(f"[WAIT] Script istirahat {wait_sec // 60} menit {wait_sec % 60} detik, lalu lanjut otomatis...")
                            time.sleep(wait_sec)
                        else:
                            print(f"[RATE LIMIT 429] Limit per menit tercapai pada batch {batch_idx}. Istirahat 65 detik...")
                            time.sleep(65)
                    else:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        print(f"  [WARN Batch {batch_idx} Attempt {attempt+1}] {last_err}")
                        time.sleep(2)
                except Exception as e:
                    if "Daily Limit Exceeded" in str(e):
                        raise e
                    last_err = str(e)
                    print(f"  [WARN Batch {batch_idx} Attempt {attempt+1}] Error: {last_err}")
                    time.sleep(2)

            if not batch_records:
                raise RuntimeError(
                    f"[ERROR] Gagal mengunduh cuaca batch {batch_idx}/{total_batches} setelah {max_retries} percobaan. "
                    f"Pesan error: {last_err}"
                )

            if batch_file and batch_records:
                pd.DataFrame(batch_records).to_parquet(batch_file)

            all_records.extend(batch_records)

            if batch_idx % 20 == 0 or batch_idx == total_batches:
                print(f"[PROGRESS CUACA] {batch_idx}/{total_batches} batch selesai.")

            time.sleep(0.3)  # Jeda aman agar server tidak rate limit

        df_weather = pd.DataFrame(all_records)
        print(f"[OK] Selesai mengunduh cuaca. Total baris data: {len(df_weather):,}.")
        return df_weather

    def generate_weather_anchor_mapping(self, gdf_grid: pd.DataFrame, resolution_deg: float = 0.25) -> tuple:
        """
        Membuat titik jangkar cuaca makro sesuai resolusi fisik ERA5 (~25 km)
        dan memetakan setiap kotak 5 km ke titik jangkar terdekat via KD-Tree.
        """
        import numpy as np
        from scipy.spatial import cKDTree

        min_lon, min_lat = gdf_grid["lon"].min(), gdf_grid["lat"].min()
        max_lon, max_lat = gdf_grid["lon"].max(), gdf_grid["lat"].max()

        lons = np.arange(min_lon - resolution_deg, max_lon + resolution_deg * 2, resolution_deg)
        lats = np.arange(min_lat - resolution_deg, max_lat + resolution_deg * 2, resolution_deg)
        mesh_lon, mesh_lat = np.meshgrid(lons, lats)
        candidate_anchors = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

        tree = cKDTree(candidate_anchors)
        dists, indices = tree.query(np.column_stack([gdf_grid["lat"], gdf_grid["lon"]]))

        # Saring hanya titik jangkar yang aktif memayungi daratan Kalimantan
        unique_indices = np.unique(indices)
        active_anchors = []
        anchor_id_map = {}

        for new_id, u_idx in enumerate(unique_indices, start=1):
            a_id = f"w_anchor_{new_id:04d}"
            anchor_id_map[u_idx] = a_id
            active_anchors.append({
                "anchor_id": a_id,
                "lat": round(float(candidate_anchors[u_idx][0]), 4),
                "lon": round(float(candidate_anchors[u_idx][1]), 4)
            })

        df_anchors = pd.DataFrame(active_anchors)

        mapping_records = []
        for g_id, u_idx, d in zip(gdf_grid["grid_id"], indices, dists):
            mapping_records.append({
                "grid_id": g_id,
                "anchor_id": anchor_id_map[u_idx],
                "dist_km": round(float(d) * 111.0, 2)
            })

        df_mapping = pd.DataFrame(mapping_records)
        return df_anchors, df_mapping

    def fetch_regional_weather(
        self,
        gdf_grid: pd.DataFrame,
        start_date: str,
        end_date: str,
        resolution_deg: float = 0.25,
        cache_dir: Path = None,
        batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Mengunduh cuaca untuk titik-titik jangkar regional (anti-rate limit)
        lalu memetakan kembali ke seluruh kotak 5 km Kalimantan dengan presisi spasial tinggi.
        """
        print(f"[INFO] Menyiapkan Weather Anchor Grid (resolusi spasial {resolution_deg}° ~25km)...")
        df_anchors, df_mapping = self.generate_weather_anchor_mapping(gdf_grid, resolution_deg)
        print(f"[OK] Terbentuk {len(df_anchors):,} titik pantau cuaca aktif (mewakili {len(gdf_grid):,} kotak 5 km).")
        print(f"[OK] Validasi spasial: Jarak rata-rata kotak ke titik pantau = {df_mapping['dist_km'].mean():.1f} km (Maksimal: {df_mapping['dist_km'].max():.1f} km).")

        # Unduh cuaca untuk titik jangkar (hanya ~16-25 batch!)
        anchor_coords = list(zip(df_anchors["anchor_id"], df_anchors["lat"], df_anchors["lon"]))
        df_anchor_weather = self.fetch_batch_weather(
            grid_coords=anchor_coords,
            start_date=start_date,
            end_date=end_date,
            is_forecast=False,
            batch_size=batch_size,
            cache_dir=cache_dir
        )

        if df_anchor_weather.empty:
            raise RuntimeError(
                "[ERROR] Data cuaca dari Open-Meteo kosong (karena terkena rate limit kuota per jam atau koneksi terputus).\n"
                "Silakan tunggu beberapa saat hingga jam berganti (kuota Open-Meteo reset otomatis setiap awal jam)."
            )

        print("[INFO] Memetakan kembali data cuaca ke seluruh kotak 5 km Kalimantan...")
        if "grid_id" in df_anchor_weather.columns:
            df_anchor_weather.rename(columns={"grid_id": "anchor_id"}, inplace=True)
        
        # Merge spasial
        df_full_weather = df_mapping[["grid_id", "anchor_id"]].merge(
            df_anchor_weather,
            on="anchor_id",
            how="inner"
        )
        df_full_weather.drop(columns=["anchor_id"], inplace=True)
        print(f"[OK] Selesai memetakan cuaca regional. Total baris data: {len(df_full_weather):,}.")
        return df_full_weather

