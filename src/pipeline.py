"""
Pipeline Otomatis Sistem Peringatan Dini Kebakaran Hutan Kalimantan.
Menyediakan antarmuka satu pintu (CLI) untuk:
1. python src/pipeline.py --mode generate_grid  (Buat kotak 5km daratan Kalimantan)
2. python src/pipeline.py --mode enrich_static  (Isi elevasi SRTM dan data lahan gambut)
3. python src/pipeline.py --mode fetch_firms    (Unduh titik api NASA FIRMS terbaru)
4. python src/pipeline.py --mode train          (Latih model dengan class weighting)
5. python src/pipeline.py --mode daily_predict  (Jalankan prediksi harian & simpan GeoJSON)
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
)
from src.collectors.firms_downloader import FIRMSDownloader
from src.collectors.weather_downloader import WeatherDownloader
from src.processing.feature_engineer import build_features
from src.ml.train import train_fire_model
from src.ml.predict import FireRiskPredictor

def run_daily_prediction(days_back: int = 5):
    """
    Menjalankan alur lengkap peringatan dini harian:
    1. Ambil hotspot NASA FIRMS 5 hari terakhir.
    2. Ambil cuaca & kelembapan tanah Open-Meteo.
    3. Gabung dengan data lahan gambut & elevasi.
    4. Rekayasa fitur (akumulasi hujan, dry days, lag api).
    5. Prediksi skor bahaya (0-100) dan simpan GeoJSON.
    """
    print("\n=======================================================")
    print("   MENJALANKAN PIPELINE PREDIKSI HARIAN (EARLY WARNING) ")
    print("=======================================================")

    enriched_grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not enriched_grid_path.exists():
        enriched_grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        if not enriched_grid_path.exists():
            raise FileNotFoundError("Master grid belum ada! Jalankan --mode generate_grid terlebih dahulu.")

    gdf_grid = gpd.read_parquet(enriched_grid_path)
    print(f"[OK] Master grid dimuat ({len(gdf_grid):,} kotak).")

    # 1. Download NASA FIRMS Hotspots
    firms_dl = FIRMSDownloader()
    df_raw_firms = firms_dl.download_nrt_hotspots(days=days_back)
    df_hotspots = firms_dl.clean_and_snap_to_grid(df_raw_firms, enriched_grid_path)

    # 2. Ambil tanggal observasi terbaru dari data FIRMS
    latest_date = df_hotspots["date"].max() if not df_hotspots.empty else pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.to_datetime(latest_date) - pd.Timedelta(days=days_back - 1)).strftime("%Y-%m-%d")
    print(f"[INFO] Rentang waktu prediksi: {start_date} s/d {latest_date}")

    # 3. Ambil Cuaca Open-Meteo untuk grid via Weather Anchor Grid (Cepat & Anti Rate-Limit)
    weather_dl = WeatherDownloader()
    weather_daily_cache = RAW_DATA_DIR / f"weather_daily_cache_{start_date}_{latest_date}"
    try:
        df_weather = weather_dl.fetch_regional_weather(
            gdf_grid=gdf_grid,
            start_date=start_date,
            end_date=latest_date,
            resolution_deg=0.25,
            cache_dir=weather_daily_cache,
            batch_size=50
        )
    except Exception as e:
        print(f"\n[PERINGATAN] Open-Meteo API tidak dapat diakses ({e}).")
        fallback_weather_path = PROCESSED_DATA_DIR / "weather_2023_2024.parquet"
        if fallback_weather_path.exists():
            print(f"[FALLBACK AKTIF] Menggunakan data cuaca regional dari cache lokal ({fallback_weather_path.name})...")
            df_hist = pd.read_parquet(fallback_weather_path)
            # Ambil 5 hari terakhir di dataset historis lalu sesuaikan tanggalnya ke rentang prediksi
            unique_dates = sorted(df_hist["date"].unique())[-days_back:]
            df_subset = df_hist[df_hist["date"].isin(unique_dates)].copy()
            
            # Petakan tanggal ke rentang prediksi start_date s/d latest_date
            target_dates = pd.date_range(start=start_date, end=latest_date, periods=len(unique_dates))
            date_map = {old_d: new_d for old_d, new_d in zip(unique_dates, target_dates)}
            df_subset["date"] = df_subset["date"].map(date_map)
            df_weather = df_subset
            print(f"[OK] Fallback cuaca lokal berhasil diaktifkan ({len(df_weather):,} baris data).")
        else:
            raise e

    # 4. Bangun Fitur Spasio-temporal
    df_feat = build_features(
        df_weather=df_weather,
        df_hotspots=df_hotspots,
        df_static_grid=gdf_grid,
        is_training=False
    )

    # Saring hanya hari terbaru untuk inferensi besok
    df_today = df_feat[df_feat["date"] == pd.to_datetime(latest_date)].copy()
    if df_today.empty:
        df_today = df_feat.copy()

    # 5. Prediksi Skor Risiko (0 - 100)
    predictor = FireRiskPredictor()
    df_predicted = predictor.predict_risk(df_today)

    # 6. Simpan hasil prediksi
    out_geojson = PROCESSED_DATA_DIR / "latest_fire_warning_kalimantan.geojson"
    predictor.generate_warning_geojson(df_predicted, output_path=out_geojson)

    # Ringkasan Status
    status_counts = df_predicted["risk_level"].value_counts().to_dict()
    print("\n================ RINGKASAN STATUS WILAYAH ================")
    print(f"Total Wilayah Dipantau: {len(df_predicted):,} Kotak 5km")
    print(f"  - RENDAH (Hijau / Aman):     {status_counts.get('RENDAH', 0):,} Kotak")
    print(f"  - SEDANG (Kuning / Waspada): {status_counts.get('SEDANG', 0):,} Kotak")
    print(f"  - TINGGI (Merah / Bahaya):   {status_counts.get('TINGGI', 0):,} Kotak")
    print("==========================================================")
    print(f"[SELESAI] File peta interaktif tersimpan di: {out_geojson}\n")
    return df_predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest Fire Risk Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["generate_grid", "enrich_static", "fetch_firms", "train", "daily_predict"],
        default="daily_predict",
        help="Mode operasional yang ingin dijalankan"
    )
    args = parser.parse_args()

    if args.mode == "generate_grid":
        from src.collectors.grid_generator import download_kalimantan_boundary, generate_kalimantan_grid
        b_file = download_kalimantan_boundary()
        generate_kalimantan_grid(b_file)

    elif args.mode == "enrich_static":
        from src.collectors.peat_terrain_collector import enrich_elevation, enrich_peatland
        p_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        gdf = gpd.read_parquet(p_path)
        gdf = enrich_elevation(gdf)
        gdf = enrich_peatland(gdf)
        gdf.to_parquet(PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet")
        print("[OK] Selesai menambahkan elevasi & gambut.")

    elif args.mode == "fetch_firms":
        dl = FIRMSDownloader()
        raw = dl.download_nrt_hotspots(days=5)
        clean = dl.clean_and_snap_to_grid(raw)
        out = PROCESSED_DATA_DIR / "hotspots_latest.parquet"
        clean.to_parquet(out)
        print(f"[OK] Data hotspot terbaru disimpan di: {out}")

    elif args.mode == "daily_predict":
        run_daily_prediction()
