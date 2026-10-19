"""
Feature Engineering untuk Sistem Prediksi Kebakaran Hutan Kalimantan.
Menghitung:
1. Akumulasi kekeringan (Hari berturut-turut tanpa hujan, total hujan 7 & 14 hari).
2. Interaksi lahan gambut x kekeringan tanah.
3. Sejarah api (Lag 1 hari & titik api di kotak tetangga).
4. Target prediksi H+1 (Apakah besok ada titik api).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def build_features(
    df_weather: pd.DataFrame,
    df_hotspots: pd.DataFrame,
    df_static_grid: pd.DataFrame,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Menggabungkan seluruh sumber data dan merekayasa fitur-fitur prediktif.
    """
    print("[INFO] Memulai pembentukan fitur spasio-temporal...")

    # 1. Pastikan format tanggal datetime dan urutkan
    df_weather = df_weather.copy()
    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_weather.sort_values(by=["grid_id", "date"], inplace=True)

    # 2. Merge Hotspot hari ini
    if df_hotspots is not None and not df_hotspots.empty:
        df_hotspots = df_hotspots.copy()
        df_hotspots["date"] = pd.to_datetime(df_hotspots["date"])
        df = df_weather.merge(
            df_hotspots[["grid_id", "date", "hotspot_count", "max_frp"]],
            on=["grid_id", "date"],
            how="left"
        )
        df["hotspot_count"] = df["hotspot_count"].fillna(0).astype(int)
        df["max_frp"] = df["max_frp"].fillna(0.0)
    else:
        df = df_weather.copy()
        df["hotspot_count"] = 0
        df["max_frp"] = 0.0

    # 3. Merge Data Lahan Statis (Gambut & Elevasi)
    static_cols = [c for c in ["grid_id", "elevation", "peat_coverage_pct", "lat", "lon"] if c in df_static_grid.columns]
    df = df.merge(df_static_grid[static_cols], on="grid_id", how="left")
    df["peat_coverage_pct"] = df["peat_coverage_pct"].fillna(0.0)
    df["elevation"] = df["elevation"].fillna(20.0)

    # 4. Feature Engineering per Grid ID (Time-series operations)
    print("[INFO] Menghitung akumulasi kekeringan & riwayat cuaca...")
    
    # Hitung total hujan 7 hari & 14 hari ke belakang
    df["rain_sum_7d"] = df.groupby("grid_id")["rain_sum"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["rain_sum_14d"] = df.groupby("grid_id")["rain_sum"].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df["temp_max_7d_avg"] = df.groupby("grid_id")["temp_max"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Hitung hari tanpa hujan berturut-turut (Dry Days Count: hujan < 1.0 mm)
    def calc_dry_days(series):
        is_dry = (series < 1.0).astype(int)
        dry_streak = is_dry * (is_dry.groupby((~is_dry.astype(bool)).cumsum()).cumcount() + 1)
        return dry_streak

    df["dry_days_count"] = df.groupby("grid_id")["rain_sum"].transform(calc_dry_days)

    # Hotspot kemarin di grid yang sama (Lag 1d)
    df["hotspot_lag_1d"] = df.groupby("grid_id")["hotspot_count"].shift(1).fillna(0)

    # 5. Fitur Interaksi (Lahan Gambut x Kekeringan Tanah)
    # Makin tebal gambut dan makin rendah kadar air tanah, nilainya makin tinggi
    # Inversi kelembapan tanah: (1.0 - soil_moisture)
    df["peat_dryness_factor"] = (df["peat_coverage_pct"] / 100.0) * (1.0 - df["soil_moisture_7_28cm"].clip(0.0, 1.0))

    # 6. Target untuk Pelatihan Model (H+1: Apakah besok ada api?)
    if is_training:
        print("[INFO] Membentuk target prediksi H+1 (besok)...")
        # Geser ke atas 1 hari untuk target besok
        df["target_fire_next_day"] = df.groupby("grid_id")["hotspot_count"].shift(-1)
        # Biner: 1 jika besok ada >= 1 hotspot, 0 jika aman
        df["target_fire_next_day"] = (df["target_fire_next_day"] > 0).astype(int)
        # Buang baris terakhir untuk tiap grid karena target besoknya belum diketahui
        df = df.dropna(subset=["target_fire_next_day"]).copy()

    print(f"[OK] Fitur berhasil dibuat. Total baris: {len(df):,}, Jumlah kolom: {len(df.columns)}.")
    return df
