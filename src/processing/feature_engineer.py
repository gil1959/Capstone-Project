"""
Feature Engineering untuk Sistem Prediksi Kebakaran Hutan Kalimantan.
Versi 2.0 — Ditingkatkan dari 16 fitur menjadi ~40 fitur prediktif.

Menghitung:
1. Akumulasi kekeringan (Hari berturut-turut tanpa hujan, total hujan 7 & 14 hari).
2. Interaksi lahan gambut x kekeringan tanah (diperluas dengan VPD, temp, dry_days).
3. Sejarah api MULTI-HARI (Lag 1d, 3d, 7d, 14d & binary flag).
4. Fitur TETANGGA SPASIAL (api di 8 grid sekitar — fitur baru paling krusial).
5. Tren pengeringan & akumulasi panas (VPD, consecutive hot days, soil moisture trend).
6. Encoding musiman (month_sin, month_cos, is_dry_season, day_of_year).
7. Indeks kekeringan komposit (KBDI-like proxy).
8. Target prediksi H+1 (Apakah besok ada titik api).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _build_neighbor_map(df_static_grid: pd.DataFrame, max_dist_deg: float = 0.065) -> dict:
    """
    Membangun peta tetangga spasial (8 arah mata angin) untuk setiap grid
    menggunakan KD-Tree. Dijalankan sekali saja (~22K grid, <1 detik).
    max_dist_deg ≈ diagonal 1 cell (0.045 * sqrt(2) ≈ 0.0636), diberi margin.
    """
    from scipy.spatial import cKDTree

    coords = np.column_stack([df_static_grid["lat"].values, df_static_grid["lon"].values])
    grid_ids = df_static_grid["grid_id"].values

    tree = cKDTree(coords)
    neighbor_map = {}

    for i, gid in enumerate(grid_ids):
        indices = tree.query_ball_point(coords[i], r=max_dist_deg)
        neighbors = [grid_ids[j] for j in indices if j != i]
        neighbor_map[gid] = neighbors

    return neighbor_map


def _compute_spatial_neighbor_features(
    df: pd.DataFrame,
    neighbor_map: dict,
    hotspot_cols: list = None
) -> pd.DataFrame:
    """
    Menghitung fitur tetangga spasial secara efisien menggunakan edge-list merge.
    Untuk setiap grid dan tanggal, menghitung:
    - neighbor_fire_count: jumlah tetangga yang ada api hari ini
    - neighbor_max_frp: FRP maks di tetangga hari ini
    - neighbor_fire_count_3d: jumlah tetangga dengan api dalam 3 hari terakhir
    """
    print("  [SPATIAL] Menghitung fitur tetangga spasial (fire spread proximity)...")

    # Bangun edge list dari neighbor_map
    edges = []
    for gid, neighbors in neighbor_map.items():
        for n in neighbors:
            edges.append({"grid_id": gid, "neighbor_id": n})

    if not edges:
        df["neighbor_fire_count"] = 0
        df["neighbor_max_frp"] = 0.0
        df["neighbor_fire_count_3d"] = 0
        return df

    df_edges = pd.DataFrame(edges)

    # Siapkan data hotspot per grid per hari (hanya yang ADA titik api > 0 untuk efisiensi tinggi)
    daily_fire = df.loc[df["hotspot_count"] > 0, ["grid_id", "date", "hotspot_count", "max_frp"]].copy()
    daily_fire = daily_fire.rename(columns={
        "grid_id": "neighbor_id",
        "hotspot_count": "n_hs",
        "max_frp": "n_frp"
    })

    if daily_fire.empty:
        df["neighbor_fire_count"] = 0
        df["neighbor_max_frp"] = 0.0
        df["neighbor_total_hotspots"] = 0
        df["neighbor_fire_count_3d"] = 0
        return df

    # Merge edge list dengan data hotspot tetangga aktif
    merged = df_edges.merge(daily_fire, on="neighbor_id", how="inner")

    if merged.empty:
        df["neighbor_fire_count"] = 0
        df["neighbor_max_frp"] = 0.0
        df["neighbor_total_hotspots"] = 0
    else:
        # Agregasi per grid_id per date
        neighbor_agg = merged.groupby(["grid_id", "date"]).agg(
            neighbor_fire_count=("n_hs", "count"),
            neighbor_max_frp=("n_frp", "max"),
            neighbor_total_hotspots=("n_hs", "sum")
        ).reset_index()

        # Merge kembali ke df utama
        df = df.merge(neighbor_agg, on=["grid_id", "date"], how="left")
        df["neighbor_fire_count"] = df["neighbor_fire_count"].fillna(0).astype(int)
        df["neighbor_max_frp"] = df["neighbor_max_frp"].fillna(0.0)
        df["neighbor_total_hotspots"] = df["neighbor_total_hotspots"].fillna(0).astype(int)

    # Neighbor fire count 3d: rolling 3 hari pada neighbor_fire_count
    df.sort_values(by=["grid_id", "date"], inplace=True)
    df["neighbor_fire_count_3d"] = df.groupby("grid_id")["neighbor_fire_count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    ).fillna(0).astype(int)

    return df


def build_features(
    df_weather: pd.DataFrame,
    df_hotspots: pd.DataFrame,
    df_static_grid: pd.DataFrame,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Menggabungkan seluruh sumber data dan merekayasa fitur-fitur prediktif.
    Versi 2.0: 40 fitur (dari 16 fitur sebelumnya).
    """
    print("[INFO] Memulai pembentukan fitur spasio-temporal (V2 — 40 fitur)...")

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

    # ================================================================
    # 4. FITUR CUACA AKUMULASI (Rolling Windows — sudah ada + baru)
    # ================================================================
    print("  [WEATHER] Menghitung akumulasi kekeringan & cuaca multi-hari...")

    # Hujan akumulasi 7 & 14 hari (sudah ada)
    df["rain_sum_7d"] = df.groupby("grid_id")["rain_sum"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["rain_sum_14d"] = df.groupby("grid_id")["rain_sum"].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df["temp_max_7d_avg"] = df.groupby("grid_id")["temp_max"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # BARU: Suhu maks rata-rata 3 hari terakhir
    df["temp_max_3d_avg"] = df.groupby("grid_id")["temp_max"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Hari tanpa hujan berturut-turut (sudah ada)
    def calc_dry_days(series):
        is_dry = (series < 1.0).astype(int)
        dry_streak = is_dry * (is_dry.groupby((~is_dry.astype(bool)).cumsum()).cumcount() + 1)
        return dry_streak

    df["dry_days_count"] = df.groupby("grid_id")["rain_sum"].transform(calc_dry_days)

    # BARU: VPD akumulasi 7 & 14 hari
    df["vpd_7d_avg"] = df.groupby("grid_id")["vpd_max"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["vpd_14d_avg"] = df.groupby("grid_id")["vpd_max"].transform(lambda x: x.rolling(14, min_periods=1).mean())

    # BARU: Hari berturut suhu > 33°C (Consecutive Hot Days)
    def calc_hot_streak(series, threshold=33.0):
        is_hot = (series > threshold).astype(int)
        hot_streak = is_hot * (is_hot.groupby((~is_hot.astype(bool)).cumsum()).cumcount() + 1)
        return hot_streak

    df["consecutive_hot_days"] = df.groupby("grid_id")["temp_max"].transform(calc_hot_streak)

    # BARU: Tren pengeringan tanah (soil moisture 7d change)
    df["soil_moisture_7d_lag"] = df.groupby("grid_id")["soil_moisture_7_28cm"].shift(7)
    df["soil_moisture_diff_7d"] = df["soil_moisture_7_28cm"] - df["soil_moisture_7d_lag"].fillna(df["soil_moisture_7_28cm"])
    df.drop(columns=["soil_moisture_7d_lag"], inplace=True)

    # BARU: Soil moisture minimum 7 hari terakhir
    df["soil_moisture_min_7d"] = df.groupby("grid_id")["soil_moisture_7_28cm"].transform(
        lambda x: x.rolling(7, min_periods=1).min()
    )

    # BARU: Indeks kekeringan komposit (KBDI-like proxy)
    # KBDI proxy = (temp_max_7d_avg - 25) * dry_days_count / (rain_sum_14d + 1)
    df["kbdi_proxy"] = (
        (df["temp_max_7d_avg"] - 25.0).clip(lower=0.0) *
        df["dry_days_count"] /
        (df["rain_sum_14d"] + 1.0)
    ).round(3)

    # ================================================================
    # 5. FITUR RIWAYAT API MULTI-HARI (Mengatasi FN — fitur terpenting)
    # ================================================================
    print("  [FIRE HISTORY] Menghitung riwayat api multi-hari (lag 1d, 3d, 7d, 14d)...")

    # Hotspot kemarin (sudah ada)
    df["hotspot_lag_1d"] = df.groupby("grid_id")["hotspot_count"].shift(1).fillna(0)

    # BARU: Total hotspot 3, 7, 14 hari terakhir
    df["hotspot_count_3d"] = df.groupby("grid_id")["hotspot_count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    ).fillna(0)
    df["hotspot_count_7d"] = df.groupby("grid_id")["hotspot_count"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    ).fillna(0)
    df["hotspot_count_14d"] = df.groupby("grid_id")["hotspot_count"].transform(
        lambda x: x.rolling(14, min_periods=1).sum()
    ).fillna(0)

    # BARU: FRP maksimum 7 hari terakhir
    df["max_frp_7d"] = df.groupby("grid_id")["max_frp"].transform(
        lambda x: x.rolling(7, min_periods=1).max()
    ).fillna(0.0)

    # BARU: Binary flags — ada api dalam N hari terakhir?
    df["has_fire_last_3d"] = (df["hotspot_count_3d"] > 0).astype(int)
    df["has_fire_last_7d"] = (df["hotspot_count_7d"] > 0).astype(int)

    # ================================================================
    # 6. FITUR INTERAKSI GAMBUT (Diperluas)
    # ================================================================
    print("  [PEAT] Menghitung interaksi gambut x kekeringan...")

    # Fitur lama: peat_dryness_factor
    df["peat_dryness_factor"] = (df["peat_coverage_pct"] / 100.0) * (1.0 - df["soil_moisture_7_28cm"].clip(0.0, 1.0))

    # BARU: Interaksi gambut x VPD
    df["peat_x_vpd"] = (df["peat_coverage_pct"] / 100.0) * df["vpd_max"]

    # BARU: Interaksi gambut x hari kering
    df["peat_x_dry_days"] = (df["peat_coverage_pct"] / 100.0) * df["dry_days_count"]

    # BARU: Interaksi gambut x suhu tinggi
    df["peat_x_temp"] = (df["peat_coverage_pct"] / 100.0) * (df["temp_max"] - 30.0).clip(lower=0.0)

    # BARU (Solusi 2): Interaksi jalan raya x gambut & jalan x kekeringan
    if "distance_to_road_km" not in df.columns:
        if "distance_to_road_km" in df_static_grid.columns:
            road_map = dict(zip(df_static_grid["grid_id"], df_static_grid["distance_to_road_km"]))
            df["distance_to_road_km"] = df["grid_id"].map(road_map).fillna(5.0).astype(np.float32)
        else:
            df["distance_to_road_km"] = 5.0

    df["road_x_peat"] = ((df["peat_coverage_pct"].fillna(0) / 100.0) / (df["distance_to_road_km"] + 1.0)).astype(np.float32)
    df["road_x_dry_days"] = (df["dry_days_count"].fillna(0) / (df["distance_to_road_km"] + 1.0)).astype(np.float32)

    # ================================================================
    # 7. FITUR TEMPORAL / MUSIMAN (Baru)
    # ================================================================
    print("  [SEASONAL] Menambahkan encoding musim & bulan...")

    dt_series = pd.to_datetime(df["date"])
    month = dt_series.dt.month
    day_of_year = dt_series.dt.dayofyear

    # Encoding siklik bulan (agar Desember → Januari tetap dekat)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0).round(4)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0).round(4)

    # Flag musim kering (Agustus - Oktober = puncak kebakaran)
    df["is_dry_season"] = month.isin([7, 8, 9, 10]).astype(int)

    # Hari ke-N dalam setahun (normalisasi 0-1)
    df["day_of_year_norm"] = (day_of_year / 365.0).round(4)

    # ================================================================
    # 8. FITUR TETANGGA SPASIAL (Baru — fitur paling krusial)
    # ================================================================
    if "lat" in df_static_grid.columns and "lon" in df_static_grid.columns:
        try:
            neighbor_map = _build_neighbor_map(df_static_grid)
            n_with_neighbors = sum(1 for v in neighbor_map.values() if len(v) > 0)
            print(f"  [SPATIAL] Peta tetangga terbentuk: {n_with_neighbors:,} grid punya tetangga.")
            df = _compute_spatial_neighbor_features(df, neighbor_map)
        except Exception as e:
            print(f"  [WARN] Gagal menghitung fitur spasial ({e}). Menggunakan default 0.")
            df["neighbor_fire_count"] = 0
            df["neighbor_max_frp"] = 0.0
            df["neighbor_total_hotspots"] = 0
            df["neighbor_fire_count_3d"] = 0
    else:
        df["neighbor_fire_count"] = 0
        df["neighbor_max_frp"] = 0.0
        df["neighbor_total_hotspots"] = 0
        df["neighbor_fire_count_3d"] = 0

    # ================================================================
    # 9. TARGET PREDIKSI H+1 (Apakah besok ada api?)
    # ================================================================
    if is_training:
        print("  [TARGET] Membentuk target prediksi H+1 (besok)...")
        df["target_fire_next_day"] = df.groupby("grid_id")["hotspot_count"].shift(-1)
        df["target_fire_next_day"] = (df["target_fire_next_day"] > 0).astype(int)
        df = df.dropna(subset=["target_fire_next_day"]).copy()

    n_features = len([c for c in df.columns if c not in ["grid_id", "date", "target_fire_next_day", "lat", "lon", "geometry", "year", "month"]])
    print(f"[OK] Fitur V2 berhasil dibuat. Total baris: {len(df):,}, Jumlah fitur: {n_features}.")
    return df
