"""
Script Pengunduh & Pembangun Dataset Latih Historis (Training Dataset) 2020 - 2025.
Sistem: "Forest Fire Risk Prediction and Early Warning System for Kalimantan Using Satellite Data and Machine Learning"

Alur Kerja:
1. Mengunduh data titik api NASA FIRMS historis (arsip VIIRS NOAA-20 SP) rentang 2020 - 2025.
2. Memfilter keyakinan (confidence) & FRP, lalu menempelkan ke Master Grid 5 km Kalimantan (spatial join).
3. Mengunduh data cuaca & kadar air tanah Open-Meteo Archive API rentang 2020 - 2025 dengan auto-checkpointing.
4. Menggabungkan data cuaca + riwayat api + data statis gambut & elevasi.
5. Rekayasa fitur (rolling rain 7d/14d, dry days count, peat dryness factor) & pembentukan target biner H+1.
6. Menyimpan dataset latih ke file Parquet siap pakai.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.collectors.firms_downloader import FIRMSDownloader
from src.collectors.weather_downloader import WeatherDownloader
from src.processing.feature_engineer import build_features

def build_learning_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    sample_grids: int = None,
    skip_firms_download: bool = False,
    skip_weather_download: bool = False
) -> pd.DataFrame:
    """
    Mengunduh data learning 2020-2025 dan menyusun dataset latih biner H+1.
    """
    print("\n" + "=" * 75)
    print("   PENGUNDUH & PEMBANGUN DATASET LATIH (LEARNING DATA) 2020 - 2025")
    print(f"   Periode: {start_date} s/d {end_date}")
    print("=" * 75 + "\n")

    # 1. Pastikan Master Grid Enriched (Gambut & Elevasi) Ada
    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not grid_path.exists():
        grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        if not grid_path.exists():
            raise FileNotFoundError("Master Grid 5km belum ditemukan! Jalankan grid_generator.py terlebih dahulu.")

    gdf_grid = gpd.read_parquet(grid_path)
    total_master_grids = len(gdf_grid)
    print(f"[OK] Master Grid 5 km dimuat ({total_master_grids:,} kotak daratan Kalimantan).")

    if sample_grids and sample_grids < total_master_grids:
        print(f"[SAMPLE] Mode pengujian aktif: Mengambil sampel {sample_grids:,} kotak representatif...")
        gdf_grid = gdf_grid.sample(sample_grids, random_state=42).copy()

    # 2. Unduh & Proses Titik Api NASA FIRMS Historis (2023 - 2024)
    hotspot_snapped_file = PROCESSED_DATA_DIR / f"hotspots_snapped_{start_date[:4]}_{end_date[:4]}.parquet"
    firms_dl = FIRMSDownloader()

    if hotspot_snapped_file.exists():
        print(f"[CACHE] Menggunakan data titik api yang sudah diproses: {hotspot_snapped_file}")
        df_hotspots = pd.read_parquet(hotspot_snapped_file)
    else:
        # Cek apakah sudah ada file mentah gabungan
        raw_firms_file = RAW_DATA_DIR / f"firms_VIIRS_NOAA20_SP_{start_date[:4]}_{end_date[:4]}.parquet"
        if raw_firms_file.exists():
            print(f"[CACHE] Ditemukan data mentah FIRMS: {raw_firms_file}")
            df_raw_firms = pd.read_parquet(raw_firms_file)
        else:
            df_raw_firms = firms_dl.download_historical_hotspots(
                start_date=start_date,
                end_date=end_date,
                sensor="VIIRS_NOAA20_SP",
                chunk_days=5
            )

        print("[INFO] Membersihkan dan memetakan titik api ke Master Grid 5 km...")
        df_hotspots = firms_dl.clean_and_snap_to_grid(df_raw_firms, grid_parquet_path=grid_path)
        df_hotspots.to_parquet(hotspot_snapped_file)
        print(f"[SIMPAN] Agregasi harian titik api tersimpan di: {hotspot_snapped_file}")

    # 3. Unduh & Proses Cuaca & Kelembapan Tanah Open-Meteo Archive API (2023 - 2024)
    weather_file = PROCESSED_DATA_DIR / f"weather_{start_date[:4]}_{end_date[:4]}.parquet"
    weather_dl = WeatherDownloader()

    if skip_weather_download and weather_file.exists():
        print(f"[SKIP] Menggunakan data cuaca yang sudah diproses: {weather_file}")
        df_weather = pd.read_parquet(weather_file)
    else:
        weather_cache_dir = RAW_DATA_DIR / f"weather_anchor_cache_{start_date[:4]}_{end_date[:4]}"

        df_weather = weather_dl.fetch_regional_weather(
            gdf_grid=gdf_grid,
            start_date=start_date,
            end_date=end_date,
            resolution_deg=0.25,
            cache_dir=weather_cache_dir,
            batch_size=50
        )
        df_weather.to_parquet(weather_file)
        print(f"[SIMPAN] Data cuaca lengkap tersimpan di: {weather_file}")

    # 4. Rekayasa Fitur & Bentuk Target Prediksi H+1
    print("\n[INFO] Menggabungkan data & merekayasa fitur prediktif...")
    df_training = build_features(
        df_weather=df_weather,
        df_hotspots=df_hotspots,
        df_static_grid=gdf_grid,
        is_training=True
    )

    # 5. Simpan Dataset Latih Siap Pakai
    out_file_versioned = PROCESSED_DATA_DIR / f"training_dataset_{start_date[:4]}_{end_date[:4]}.parquet"
    out_file_default = PROCESSED_DATA_DIR / "training_dataset.parquet"

    df_training.to_parquet(out_file_versioned)
    df_training.to_parquet(out_file_default)

    # Ringkasan Statistik Dataset Latih
    n_pos = int((df_training["target_fire_next_day"] == 1).sum())
    n_neg = int((df_training["target_fire_next_day"] == 0).sum())
    pos_pct = (n_pos / len(df_training) * 100.0) if len(df_training) > 0 else 0.0

    print("\n" + "=" * 75)
    print("   RINGKASAN DATASET LATIH (READY FOR XGBOOST TRAINING)")
    print("=" * 75)
    print(f"File Output Utama    : {out_file_default}")
    print(f"File Output Versi    : {out_file_versioned}")
    print(f"Rentang Waktu        : {df_training['date'].min().strftime('%Y-%m-%d')} s/d {df_training['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Baris Data     : {len(df_training):,} baris")
    print(f"Jumlah Kotak Grid    : {df_training['grid_id'].nunique():,} kotak")
    print(f"Kelas 1 (Api Besok)  : {n_pos:,} baris ({pos_pct:.2f}%)")
    print(f"Kelas 0 (Aman Besok) : {n_neg:,} baris ({100.0 - pos_pct:.2f}%)")
    print(f"Rasio Ketimpangan    : 1 Api berbanding {n_neg // max(1, n_pos)} Aman")
    print(f"Rekomendasi scale_pos_weight untuk XGBoost: {n_neg / max(1, n_pos):.1f}")
    print("=" * 75 + "\n")

    return df_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Unduh & Bangun Dataset Latih 2020 - 2025")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Tanggal awal (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="Tanggal akhir (YYYY-MM-DD)")
    parser.add_argument("--sample_grids", type=int, default=None, help="Jumlah sampel kotak (opsional, default: seluruh grid)")
    parser.add_argument("--skip_firms", action="store_true", help="Lewati unduh FIRMS jika file sudah ada")
    parser.add_argument("--skip_weather", action="store_true", help="Lewati unduh cuaca jika file sudah ada")
    
    args = parser.parse_args()

    build_learning_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_grids=args.sample_grids,
        skip_firms_download=args.skip_firms,
        skip_weather_download=args.skip_weather
    )
