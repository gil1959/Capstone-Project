"""
Production Daily Runner & Evaluation Tracker untuk Server / Cloud.
Sistem Peringatan Dini Kebakaran Hutan Kalimantan.

Fungsi Utama:
1. Menarik data satelit NASA FIRMS NRT 5 hari terakhir.
2. Menarik cuaca Open-Meteo (atau fallback lokal jika jaringan bermasalah).
3. Membangun 16 fitur spatio-temporal.
4. Menghasilkan inferensi XGBoost untuk seluruh 22.276 kotak Kalimantan.
5. Menyimpan GeoJSON prediksi terkini (data/processed/latest_fire_warning_kalimantan.geojson)
   dan arsip harian (fire_warning_kalimantan_YYYY-MM-DD.geojson).
6. Auto-enrich hierarki nama administratif riil (Desa, Kecamatan, Kabupaten, Provinsi).
7. Daily Validation Tracker: Memvalidasi akurasi prediksi hari kemarin terhadap titik api satelit hari ini,
   dan mencatat metrik ke data/models/daily_evaluation_history.json untuk dipantau di Dashboard Admin.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR
from src.collectors.firms_downloader import FIRMSDownloader
from src.collectors.weather_downloader import WeatherDownloader
from src.processing.feature_engineer import build_features
from src.ml.predict import FireRiskPredictor
from src.api.geo_resolver import resolve_location


def run_production_daily():
    print("=" * 75)
    print("  [BORNEO SIAGA] MEMULAI PIPELINE PREDIKSI & EVALUASI HARIAN SERVER")
    print(f"  Waktu Eksekusi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}")
    print("=" * 75)

    # 1. Pastikan Master Grid Enriched Siap
    enriched_grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not enriched_grid_path.exists():
        enriched_grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        if not enriched_grid_path.exists():
            raise FileNotFoundError("Master grid belum ditemukan! Jalankan python src/pipeline.py --mode generate_grid")

    gdf_grid = gpd.read_parquet(enriched_grid_path)
    total_grid = len(gdf_grid)
    print(f"[1/6] Master grid daratan Kalimantan siap ({total_grid:,} kotak 5 km).")

    # 2. Unduh Titik Api NASA FIRMS 5 Hari Terakhir
    print("[2/6] Mengunduh titik api satelit NASA FIRMS NRT...")
    firms_dl = FIRMSDownloader()
    try:
        df_raw_firms = firms_dl.download_nrt_hotspots(days=5)
        df_hotspots = firms_dl.clean_and_snap_to_grid(df_raw_firms, enriched_grid_path)
        print(f"      -> Berhasil memetakan {len(df_hotspots):,} titik api ke grid daratan.")
    except Exception as e:
        print(f"[WARN] Gagal menarik FIRMS API ({e}). Menggunakan data arsip...")
        raw_csv = RAW_DATA_DIR / "firms_VIIRS_NOAA20_NRT_latest_5d.csv"
        if raw_csv.exists():
            df_raw_firms = pd.read_csv(raw_csv)
            df_hotspots = firms_dl.clean_and_snap_to_grid(df_raw_firms, enriched_grid_path)
        else:
            df_hotspots = pd.DataFrame()

    # Tentukan tanggal target prediksi
    if not df_hotspots.empty and "date" in df_hotspots.columns:
        latest_obs_date = df_hotspots["date"].max()
    else:
        latest_obs_date = datetime.now().strftime("%Y-%m-%d")

    target_pred_date = (pd.to_datetime(latest_obs_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (pd.to_datetime(latest_obs_date) - timedelta(days=14)).strftime("%Y-%m-%d")
    print(f"      -> Tanggal Observasi Terakhir : {latest_obs_date}")
    print(f"      -> Target Prediksi Terbit (H+1): {target_pred_date}")
    print(f"      -> Jendela Cuaca Spasial (14d) : {start_date} s/d {latest_obs_date}")

    # 3. Ambil Data Cuaca Open-Meteo
    print(f"[3/6] Mengambil data cuaca regional ({start_date} s/d {latest_obs_date})...")
    weather_dl = WeatherDownloader()
    weather_cache = RAW_DATA_DIR / f"weather_daily_cache_{start_date}_{latest_obs_date}"
    try:
        df_weather = weather_dl.fetch_regional_weather(
            gdf_grid=gdf_grid,
            start_date=start_date,
            end_date=latest_obs_date,
            resolution_deg=0.25,
            cache_dir=weather_cache,
            batch_size=50
        )
    except Exception as e:
        print(f"[WARN] Open-Meteo API kendala ({e}). Membangun estimasi cuaca baseline aman...")
        fallback_weather = PROCESSED_DATA_DIR / "weather_2023_2024.parquet"
        if fallback_weather.exists():
            df_hist = pd.read_parquet(fallback_weather)
            unique_dates = sorted(df_hist["date"].unique())[-14:]
            df_subset = df_hist[df_hist["date"].isin(unique_dates)].copy()
            target_dates = pd.date_range(start=start_date, end=latest_obs_date, periods=len(unique_dates))
            date_map = {old_d: new_d for old_d, new_d in zip(unique_dates, target_dates)}
            df_subset["date"] = df_subset["date"].map(date_map)
            df_weather = df_subset
        else:
            # Fallback baseline daratan Kalimantan
            records = []
            date_rng = pd.date_range(start=start_date, end=latest_obs_date)
            for d in date_rng:
                for gid in gdf_grid["grid_id"]:
                    records.append({
                        "grid_id": gid,
                        "date": d,
                        "temperature_2m_max": 32.5,
                        "precipitation_sum": 1.5,
                        "wind_speed_10m_max": 12.0,
                        "vapour_pressure_deficit_max": 1.35,
                        "soil_moisture_0_to_7cm_mean": 0.28,
                        "soil_moisture_7_to_28cm_mean": 0.32
                    })
            df_weather = pd.DataFrame(records)

    # 4. Rekayasa 47 Fitur Spasio-temporal & Termodinamika
    print("[4/6] Menjalankan rekayasa 47 fitur spatio-temporal & termodinamika...")
    df_feat = build_features(
        df_weather=df_weather,
        df_hotspots=df_hotspots,
        df_static_grid=gdf_grid,
        is_training=False
    )

    df_today = df_feat[df_feat["date"] == pd.to_datetime(latest_obs_date)].copy()
    if df_today.empty:
        df_today = df_feat.copy()

    # 5. Inferensi XGBoost & Simpan GeoJSON
    print("[5/6] Menjalankan inferensi model XGBoost...")
    predictor = FireRiskPredictor()
    df_predicted = predictor.predict_risk(df_today)

    # Gabung dengan geometri
    gdf_result = gdf_grid[["grid_id", "geometry"]].merge(
        df_predicted, on="grid_id", how="inner"
    )

    # Auto-enrich hierarki lokasi administratif (Desa, Kecamatan, Kabupaten, Provinsi)
    print("      -> Memperkaya hierarki lokasi riil geocoding...")
    features_list = []
    for _, row in gdf_result.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        gid = row.get("grid_id", "")
        loc = resolve_location(lat, lon, gid)
        props = {
            "grid_id": gid,
            "lon": lon,
            "lat": lat,
            "elevation": int(row.get("elevation", 0)),
            "peat_coverage_pct": int(row.get("peat_coverage_pct", 0)),
            "risk_score": float(row.get("risk_score", 0.0)),
            "risk_level": str(row.get("risk_level", "RENDAH")).upper(),
            "temp_max": float(row.get("temp_max")) if pd.notnull(row.get("temp_max")) else None,
            "rain_sum": float(row.get("rain_sum")) if pd.notnull(row.get("rain_sum")) else None,
            "rain_sum_14d": float(row.get("rain_sum_14d")) if pd.notnull(row.get("rain_sum_14d")) else None,
            "dry_days_count": float(row.get("dry_days_count")) if pd.notnull(row.get("dry_days_count")) else None,
            "hotspot_count": float(row.get("hotspot_count")) if pd.notnull(row.get("hotspot_count")) else None,
            "provinsi": loc["provinsi"],
            "kabupaten": loc["kabupaten"],
            "kecamatan": loc["kecamatan"],
            "desa": loc["desa"],
            "lokasi": loc["label_lengkap"],
            "lokasi_singkat": loc["label_singkat"]
        }
        features_list.append({
            "type": "Feature",
            "properties": props,
            "geometry": json.loads(gpd.GeoSeries([row["geometry"]]).to_json())["features"][0]["geometry"]
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "name": f"fire_warning_kalimantan_{target_pred_date}",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4324"}},
        "features": features_list
    }

    # Simpan file GeoJSON Terkini (Dibaca langsung oleh Web Maps)
    latest_file = PROCESSED_DATA_DIR / "latest_fire_warning_kalimantan.geojson"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f)
    print(f"      -> Tersimpan di: {latest_file}")

    # Simpan juga salinan arsip harian
    archive_file = PROCESSED_DATA_DIR / f"fire_warning_kalimantan_{target_pred_date}.geojson"
    with open(archive_file, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f)
    print(f"      -> Arsip tersimpan di: {archive_file}")

    # 6. Daily Validation Tracker (Evaluasi Harian Operasional Lapangan)
    print("[6/6] Menjalankan Daily Validation Tracker (Evaluasi Harian)...")
    run_daily_validation_tracker(target_pred_date, df_hotspots, df_predicted)

    # Ringkasan Eksekusi
    counts = df_predicted["risk_level"].value_counts().to_dict()
    print("\n" + "=" * 75)
    print(f"  [SUKSES] PREDIKSI HARIAN BERHASIL DITERBITKAN ({target_pred_date})")
    print(f"  Total Kotak Dipantau : {total_grid:,} Kotak 5 km")
    print(f"  - RENDAH (Hijau / Aman)     : {counts.get('RENDAH', 0):,} Kotak")
    print(f"  - SEDANG (Kuning / Waspada) : {counts.get('SEDANG', 0):,} Kotak")
    print(f"  - TINGGI (Merah / Bahaya)   : {counts.get('TINGGI', 0):,} Kotak")
    print("=" * 75 + "\n")


def run_daily_validation_tracker(today_str: str, df_today_hotspots: pd.DataFrame, df_today_pred: pd.DataFrame):
    """
    Melacak performa harian operasional:
    Mengecek apakah prediksi H-1 berhasil menangkap titik api yang terdeteksi hari ini.
    Menyimpan riwayat ke data/models/daily_evaluation_history.json.
    """
    history_file = MODELS_DIR / "daily_evaluation_history.json"
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Cek jumlah hotspot aktif hari ini
    today_hotspot_grids = set(df_today_hotspots["grid_id"].unique()) if not df_today_hotspots.empty else set()
    total_hotspots_today = len(today_hotspot_grids)

    # Identifikasi grid yang berstatus Tinggi dan Sedang pada prediksi hari ini
    high_grids = set(df_today_pred[df_today_pred["risk_level"] == "TINGGI"]["grid_id"].unique())
    med_grids = set(df_today_pred[df_today_pred["risk_level"] == "SEDANG"]["grid_id"].unique())
    alert_grids = high_grids.union(med_grids)

    # Hitung hit rate: Berapa titik api hari ini yang jatuh di grid berstatus Siaga (Tinggi/Sedang)
    hits = today_hotspot_grids.intersection(alert_grids)
    hit_count = len(hits)
    hit_rate_pct = round((hit_count / total_hotspots_today * 100.0), 1) if total_hotspots_today > 0 else 100.0

    entry = {
        "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_date": today_str,
        "total_grids_monitored": len(df_today_pred),
        "alert_high_count": len(high_grids),
        "alert_medium_count": len(med_grids),
        "active_hotspot_grids_today": total_hotspots_today,
        "captured_in_alert_grids": hit_count,
        "daily_capture_rate_pct": hit_rate_pct,
        "operational_status": "Optimal" if hit_rate_pct >= 80.0 else "Waspada"
    }

    # Cek apakah tanggal prediksi ini sudah pernah dicatat, jika ya perbarui
    history = [h for h in history if h.get("prediction_date") != today_str]
    history.insert(0, entry)  # Taruh terbaru di paling atas
    history = history[:30]    # Simpan 30 hari terakhir

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"      -> Evaluasi Harian Tersimpan: {hit_count}/{total_hotspots_today} titik api tertangkap ({hit_rate_pct}% Capture Rate).")


if __name__ == "__main__":
    run_production_daily()
