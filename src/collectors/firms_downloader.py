"""
Downloader & Processor untuk NASA FIRMS Hotspots.
Mendukung penarikan data NRT (Near Real-Time) harian dan arsip historis,
lalu memetakannya langsung ke Grid 5 km Kalimantan.
"""

import os
import sys
from pathlib import Path
import io
import time
import requests
import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    NASA_FIRMS_MAP_KEY,
    KALIMANTAN_BBOX,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

class FIRMSDownloader:
    def __init__(self, map_key: str = None):
        self.map_key = map_key or NASA_FIRMS_MAP_KEY
        # Bounding box Kalimantan
        self.bbox_str = f"{KALIMANTAN_BBOX['min_lon']},{KALIMANTAN_BBOX['min_lat']},{KALIMANTAN_BBOX['max_lon']},{KALIMANTAN_BBOX['max_lat']}"

    def download_nrt_hotspots(self, days: int = 5, sensor: str = "VIIRS_NOAA20_NRT") -> pd.DataFrame:
        """
        Mengunduh titik api Near-Real-Time dari NASA FIRMS (rentang 1 sampai 10 hari).
        Mendukung API key resmi, dan otomatis fallback ke direct open-access CSV jika key belum disetel.
        """
        print(f"[INFO] Mengunduh titik api FIRMS ({sensor}) untuk {days} hari terakhir...")
        df = None

        # Jalur 1: Menggunakan NASA FIRMS API Key resmi jika tersedia
        if self.map_key and len(self.map_key.strip()) > 5:
            try:
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.map_key}/{sensor}/{self.bbox_str}/{days}"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and not resp.text.startswith("Error"):
                    df = pd.read_csv(io.StringIO(resp.text))
                    print(f"[OK] Berhasil mengunduh via FIRMS API ({len(df):,} baris).")
            except Exception as e:
                print(f"[WARN] Akses FIRMS API Key kendala: {e}. Mengalihkan ke open feed publik...")

        # Jalur 2: Fallback ke open-access direct feed South East Asia (tanpa key)
        if df is None or df.empty:
            print("[INFO] Mengunduh feed publik open-access NASA FIRMS South East Asia...")
            feed_urls = [
                "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-nrt/csv/J1_VIIRS_C2_SouthEast_Asia_7d.csv",
                "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-nrt/csv/SUOMI_VIIRS_C2_SouthEast_Asia_7d.csv"
            ]
            for feed_url in feed_urls:
                try:
                    resp = requests.get(feed_url, timeout=35, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        df_raw = pd.read_csv(io.StringIO(resp.text))
                        # Filter koordinat sesuai Bounding Box Kalimantan
                        mask = (
                            (df_raw["latitude"] >= KALIMANTAN_BBOX["min_lat"]) &
                            (df_raw["latitude"] <= KALIMANTAN_BBOX["max_lat"]) &
                            (df_raw["longitude"] >= KALIMANTAN_BBOX["min_lon"]) &
                            (df_raw["longitude"] <= KALIMANTAN_BBOX["max_lon"])
                        )
                        df = df_raw[mask].copy()
                        print(f"[OK] Berhasil mengunduh via open feed NASA ({len(df):,} titik di Kalimantan).")
                        break
                except Exception as ex:
                    print(f"[WARN] Open feed {feed_url} gagal: {ex}")

        if df is None or df.empty:
            raise RuntimeError("Gagal mengunduh titik api dari seluruh sumber NASA FIRMS (API maupun open feed).")

        # Simpan file mentah
        raw_file = RAW_DATA_DIR / f"firms_{sensor}_latest_{days}d.csv"
        df.to_csv(raw_file, index=False)
        print(f"[SIMPAN] Data mentah disimpan di: {raw_file}")
        return df

    def download_historical_hotspots(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        sensor: str = "VIIRS_NOAA20_SP",
        chunk_days: int = 5
    ) -> pd.DataFrame:
        """
        Mengunduh titik api historis NASA FIRMS (arsip Standard Processing SP)
        dalam rentang tanggal tertentu dengan chunk bertahap (5 hari per request)
        dan sistem checkpoint/cache agar tahan putus koneksi.
        """
        cache_dir = RAW_DATA_DIR / f"firms_cache_{sensor}_{start_date[:4]}_{end_date[:4]}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)
        current_dt = dt_start

        # Bentuk daftar potongan tanggal
        date_chunks = []
        while current_dt <= dt_end:
            days_in_chunk = min(chunk_days, (dt_end - current_dt).days + 1)
            date_chunks.append((current_dt.strftime("%Y-%m-%d"), days_in_chunk))
            current_dt += pd.Timedelta(days=days_in_chunk)

        print(f"[INFO] Mengunduh arsip titik api FIRMS ({sensor}) dari {start_date} s/d {end_date}...")
        print(f"[INFO] Total {len(date_chunks)} chunk (interval {chunk_days} hari) dengan auto-checkpoint di: {cache_dir}")

        all_dfs = []
        downloaded_count = 0
        cached_count = 0

        for idx, (chunk_date, days_cnt) in enumerate(date_chunks, start=1):
            chunk_file = cache_dir / f"chunk_{chunk_date}_{days_cnt}d.parquet"
            if chunk_file.exists():
                df_chunk = pd.read_parquet(chunk_file)
                all_dfs.append(df_chunk)
                cached_count += 1
                continue

            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.map_key}/{sensor}/{self.bbox_str}/{days_cnt}/{chunk_date}"
            max_retries = 3
            df_chunk = None

            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        text = resp.text.strip()
                        if text and not text.startswith("No data") and "latitude" in text:
                            df_chunk = pd.read_csv(io.StringIO(text))
                        else:
                            df_chunk = pd.DataFrame()
                        df_chunk.to_parquet(chunk_file)
                        all_dfs.append(df_chunk)
                        downloaded_count += 1
                        break
                    elif resp.status_code == 429:
                        print(f"[WARN] Rate limit pada {chunk_date}. Menunggu 5 detik...")
                        time.sleep(5)
                    else:
                        time.sleep(1)
                except Exception as e:
                    time.sleep(2)

            if df_chunk is None:
                # Simpan dataframe kosong agar tidak looping error
                df_empty = pd.DataFrame()
                df_empty.to_parquet(chunk_file)
                all_dfs.append(df_empty)

            if idx % 10 == 0 or idx == len(date_chunks):
                print(f"[PROGRESS FIRMS] {idx}/{len(date_chunks)} chunk selesai (baru: {downloaded_count}, cache: {cached_count}).")

            time.sleep(0.15)

        # Gabungkan semua chunk
        valid_dfs = [d for d in all_dfs if not d.empty]
        if valid_dfs:
            df_combined = pd.concat(valid_dfs, ignore_index=True)
            dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df_combined.columns]
            if dedup_cols:
                df_combined.drop_duplicates(subset=dedup_cols, inplace=True)
        else:
            df_combined = pd.DataFrame()

        out_combined = RAW_DATA_DIR / f"firms_{sensor}_{start_date[:4]}_{end_date[:4]}.parquet"
        df_combined.to_parquet(out_combined)
        print(f"[OK] Selesai unduh FIRMS {sensor}. Total titik api mentah: {len(df_combined):,}.")
        print(f"[SIMPAN] Tersimpan di: {out_combined}")
        return df_combined


    def clean_and_snap_to_grid(self, df_hotspots: pd.DataFrame, grid_parquet_path: Path = None) -> pd.DataFrame:
        """
        Membersihkan data titik api (confidence & FRP) dan memetakannya ke Grid 5 km.
        """
        if grid_parquet_path is None:
            grid_parquet_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"

        if not grid_parquet_path.exists():
            raise FileNotFoundError(f"File grid {grid_parquet_path} belum ada! Jalankan grid_generator.py dulu.")

        print("[INFO] Membaca Master Grid 5 km Kalimantan...")
        gdf_grid = gpd.read_parquet(grid_parquet_path)

        # 1. Cleaning: Filter keyakinan (Confidence), FRP, dan DayNight
        print("[INFO] Membersihkan titik api mentah (V2 — filter lebih ketat)...")
        initial_len = len(df_hotspots)

        # V2: Deduplikasi eksplisit pada koordinat + waktu
        dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df_hotspots.columns]
        if dedup_cols:
            before_dedup = len(df_hotspots)
            df_hotspots = df_hotspots.drop_duplicates(subset=dedup_cols).copy()
            dropped_dup = before_dedup - len(df_hotspots)
            if dropped_dup > 0:
                print(f"  [DEDUP] Menghapus {dropped_dup:,} baris duplikat (lat+lon+date+time).")

        # V2: Preferensi deteksi siang hari (daynight = "D")
        # Deteksi malam hari lebih rentan false positive dari sumber panas non-api
        if "daynight" in df_hotspots.columns:
            day_count = (df_hotspots["daynight"].str.upper() == "D").sum()
            night_count = (df_hotspots["daynight"].str.upper() == "N").sum()
            # Hanya filter jika ada cukup data siang (>30%)
            if day_count > 0 and day_count / max(1, len(df_hotspots)) > 0.3:
                df_hotspots = df_hotspots[df_hotspots["daynight"].str.upper() == "D"].copy()
                print(f"  [DAYNIGHT] Memfilter deteksi siang hari saja: {day_count:,} siang, {night_count:,} malam dibuang.")

        # Filter confidence (sudah ada, tetap dipertahankan)
        if "confidence" in df_hotspots.columns:
            conf_str = df_hotspots["confidence"].astype(str).str.lower()
            numeric_conf = pd.to_numeric(df_hotspots["confidence"], errors="coerce")
            if numeric_conf.notna().any():
                df_clean = df_hotspots[numeric_conf >= 50].copy()
            else:
                df_clean = df_hotspots[conf_str.isin(["n", "h", "nominal", "high"])].copy()
        else:
            df_clean = df_hotspots.copy()

        # V2: Filter FRP >= 0.5 (lebih ketat — FRP mendekati 0 sering false detection)
        if "frp" in df_clean.columns:
            df_clean = df_clean[df_clean["frp"] >= 0.5]

        print(f"[CLEAN] Setelah filter V2: {len(df_clean):,} titik api (dari {initial_len:,}, terhapus {initial_len - len(df_clean):,}).")

        # 2. Spatial Join: Titik koordinat ke Poligon Kotak 5 km
        gdf_points = gpd.GeoDataFrame(
            df_clean,
            geometry=gpd.points_from_xy(df_clean["longitude"], df_clean["latitude"]),
            crs="EPSG:4326"
        )

        print("[INFO] Menempelkan titik api ke kotak grid Kalimantan...")
        if gdf_grid.crs is not None and gdf_points.crs != gdf_grid.crs:
            gdf_points = gdf_points.to_crs(gdf_grid.crs)
        joined = gpd.sjoin(gdf_points, gdf_grid[["grid_id", "geometry"]], how="inner", predicate="within")

        # Agregasi harian per grid
        summary = joined.groupby(["grid_id", "acq_date"]).agg(
            hotspot_count=("frp", "count"),
            max_frp=("frp", "max"),
            mean_frp=("frp", "mean")
        ).reset_index()

        summary.rename(columns={"acq_date": "date"}, inplace=True)
        print(f"[OK] Berhasil diagregasikan menjadi {len(summary):,} baris data harian per grid.")
        return summary


if __name__ == "__main__":
    downloader = FIRMSDownloader()
    df_raw = downloader.download_nrt_hotspots(days=5)
    if len(df_raw) > 0:
        df_summary = downloader.clean_and_snap_to_grid(df_raw)
        out_path = PROCESSED_DATA_DIR / "hotspots_snapped_5d.parquet"
        df_summary.to_parquet(out_path)
        print(f"[BERHASIL] Rangkuman hotspot per grid disimpan di: {out_path}")
        print(df_summary.head(10))
