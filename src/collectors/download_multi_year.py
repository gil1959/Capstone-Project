"""
Downloader & Processor Multi-Tahun (2020 - 2025) untuk Sistem Borneo Siaga.
Mendukung eksekusi per-tahun modular (2020, 2021, 2022, 2025),
memanfaatkan cache yang sudah ada untuk 2023-2024.
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


def download_year_data(
    year: int,
    gdf_grid: gpd.GeoDataFrame,
    firms_dl: FIRMSDownloader,
    weather_dl: WeatherDownloader,
    skip_existing: bool = True
):
    """Mengunduh dan memproses FIRMS dan Cuaca untuk satu tahun kalender."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not grid_path.exists():
        grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"

    print(f"\n{'='*70}\n   MEMPROSES DATA TAHUN {year} ({start_date} s/d {end_date})\n{'='*70}")

    # 1. Hotspots
    hs_file = PROCESSED_DATA_DIR / f"hotspots_snapped_{year}.parquet"
    if skip_existing and hs_file.exists():
        print(f"[CACHE] Hotspot tahun {year} sudah ada: {hs_file}")
    else:
        print(f"[FIRMS] Mengunduh hotspot {year}...")
        df_raw = firms_dl.download_historical_hotspots(
            start_date=start_date,
            end_date=end_date,
            sensor="VIIRS_NOAA20_SP",
            chunk_days=5
        )
        print(f"[FIRMS] Snapping titik api {year} ke grid...")
        df_hs = firms_dl.clean_and_snap_to_grid(df_raw, grid_parquet_path=grid_path)
        df_hs.to_parquet(hs_file)
        print(f"[OK] Hotspot {year} tersimpan: {len(df_hs):,} baris -> {hs_file}")

    # 2. Weather
    weath_file = PROCESSED_DATA_DIR / f"weather_{year}.parquet"
    if skip_existing and weath_file.exists():
        print(f"[CACHE] Cuaca tahun {year} sudah ada: {weath_file}")
    else:
        cache_dir = RAW_DATA_DIR / f"weather_anchor_cache_{year}"
        print(f"[WEATHER] Mengunduh cuaca {year} (Open-Meteo Archive API)...")
        df_weath = weather_dl.fetch_regional_weather(
            gdf_grid=gdf_grid,
            start_date=start_date,
            end_date=end_date,
            resolution_deg=0.25,
            cache_dir=cache_dir,
            batch_size=50
        )
        df_weath.to_parquet(weath_file)
        print(f"[OK] Cuaca {year} tersimpan: {len(df_weath):,} baris -> {weath_file}")


def split_existing_2023_2024():
    """Memecah cache 2023_2024 menjadi file per tahun (2023 dan 2024) jika belum ada."""
    for yr in [2023, 2024]:
        hs_yr = PROCESSED_DATA_DIR / f"hotspots_snapped_{yr}.parquet"
        weath_yr = PROCESSED_DATA_DIR / f"weather_{yr}.parquet"

        if not hs_yr.exists():
            combined_hs = PROCESSED_DATA_DIR / "hotspots_snapped_2023_2024.parquet"
            if combined_hs.exists():
                print(f"[SPLIT] Memisahkan hotspot {yr} dari 2023_2024...")
                df = pd.read_parquet(combined_hs)
                df["date"] = pd.to_datetime(df["date"])
                df_sub = df[df["date"].dt.year == yr].copy()
                df_sub.to_parquet(hs_yr)
                print(f"[OK] hotspots_snapped_{yr}.parquet tersimpan ({len(df_sub):,} baris).")

        if not weath_yr.exists():
            combined_weath = PROCESSED_DATA_DIR / "weather_2023_2024.parquet"
            if combined_weath.exists():
                print(f"[SPLIT] Memisahkan cuaca {yr} dari 2023_2024...")
                df = pd.read_parquet(combined_weath)
                df["date"] = pd.to_datetime(df["date"])
                df_sub = df[df["date"].dt.year == yr].copy()
                df_sub.to_parquet(weath_yr)
                print(f"[OK] weather_{yr}.parquet tersimpan ({len(df_sub):,} baris).")


def combine_all_years_and_build_dataset(years: list = [2020, 2021, 2022, 2023, 2024, 2025]):
    """
    Membangun fitur 40 kolom secara modular per-tahun (Memory-Friendly / Anti-OOM).
    Setiap tahun diproses secara terpisah dan disimpan ke training_dataset_{year}.parquet.
    """
    import gc
    print(f"\n{'='*70}\n   MEMBANGUN FITUR 40 KOLOM MULTI-TAHUN (MEMORY-FRIENDLY): {years}\n{'='*70}")

    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not grid_path.exists():
        grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
    gdf_grid = gpd.read_parquet(grid_path)

    yearly_datasets = []
    for yr in years:
        yr_dataset_file = PROCESSED_DATA_DIR / f"training_dataset_{yr}.parquet"
        if yr_dataset_file.exists():
            print(f"[CACHE] Dataset fitur tahun {yr} sudah ada: {yr_dataset_file}")
            yearly_datasets.append(yr_dataset_file)
            continue

        weath_file = PROCESSED_DATA_DIR / f"weather_{yr}.parquet"
        hs_file = PROCESSED_DATA_DIR / f"hotspots_snapped_{yr}.parquet"
        if not weath_file.exists() or not hs_file.exists():
            print(f"[WARN] Data tahun {yr} tidak lengkap (cuaca atau hotspot tidak ditemukan). Dilewati.")
            continue

        print(f"\n--- [FITUR V2] Memproses Dataset Tahun {yr} ({yr}-01-01 s/d {yr}-12-31) ---")
        df_w = pd.read_parquet(weath_file)
        df_h = pd.read_parquet(hs_file)

        df_feat = build_features(
            df_weather=df_w,
            df_hotspots=df_h,
            df_static_grid=gdf_grid,
            is_training=True
        )

        df_feat.to_parquet(yr_dataset_file)
        print(f"[OK] Selesai tahun {yr}: {len(df_feat):,} baris -> {yr_dataset_file}")
        yearly_datasets.append(yr_dataset_file)

        # Bebaskan memori
        del df_w, df_h, df_feat
        gc.collect()

    print(f"\n{'='*70}\n[BERHASIL] Seluruh {len(yearly_datasets)} tahun dataset fitur V2 siap digunakan!\n{'='*70}")
    return yearly_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-year Data Downloader 2020-2025")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2025], help="Tahun yang ingin diunduh")
    parser.add_argument("--combine_only", action="store_true", help="Hanya gabungkan tahun yang sudah ada")
    args = parser.parse_args()

    split_existing_2023_2024()

    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    gdf_grid = gpd.read_parquet(grid_path)
    firms_dl = FIRMSDownloader()
    weather_dl = WeatherDownloader()

    if not args.combine_only:
        for yr in args.years:
            download_year_data(yr, gdf_grid, firms_dl, weather_dl)

    all_available = sorted([yr for yr in range(2020, 2026) if (PROCESSED_DATA_DIR / f"weather_{yr}.parquet").exists()])
    print(f"\n[INFO] Seluruh tahun yang terdeteksi siap digabungkan: {all_available}")
    combine_all_years_and_build_dataset(all_available)
