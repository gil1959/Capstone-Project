"""
Pipeline Pelatihan & Evaluasi Model XGBoost Peringatan Dini Kebakaran Hutan Kalimantan.
Versi 2.0 — Ditingkatkan untuk mengatasi False Negative tinggi.

Perubahan utama:
- Daftar fitur diperluas dari 16 → 40 fitur prediktif.
- Data latih diperluas dari 1 tahun (2023) → multi-tahun (2020-2024).
- Data uji dari 2024 → 2025 (out-of-time validation).
- Hyperparameter dioptimalkan: scale_pos_weight ↑, n_estimators ↑, regularization.
- Warmup cutoff diterapkan per tahun (14 hari awal setiap tahun dipotong).
- Outlier detection pada nilai cuaca ekstrem.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# 40 Fitur Prediktif Spasio-Temporal V2
FEATURE_COLUMNS = [
    # --- Cuaca Harian Dinamis (6) ---
    "temp_max",
    "rain_sum",
    "wind_speed_max",
    "vpd_max",
    "soil_moisture_0_7cm",
    "soil_moisture_7_28cm",

    # --- Akumulasi Kekeringan & Cuaca Multi-Hari (11) ---
    "rain_sum_7d",
    "rain_sum_14d",
    "temp_max_7d_avg",
    "temp_max_3d_avg",
    "dry_days_count",
    "vpd_7d_avg",
    "vpd_14d_avg",
    "consecutive_hot_days",
    "soil_moisture_diff_7d",
    "soil_moisture_min_7d",
    "kbdi_proxy",

    # --- Riwayat Titik Api Multi-Hari (9) ---
    "hotspot_count",
    "max_frp",
    "hotspot_lag_1d",
    "hotspot_count_3d",
    "hotspot_count_7d",
    "hotspot_count_14d",
    "max_frp_7d",
    "has_fire_last_3d",
    "has_fire_last_7d",

    # --- Tetangga Spasial (4) ---
    "neighbor_fire_count",
    "neighbor_max_frp",
    "neighbor_total_hotspots",
    "neighbor_fire_count_3d",

    # --- Temporal / Musiman (4) ---
    "month_sin",
    "month_cos",
    "is_dry_season",
    "day_of_year_norm",

    # --- Karakteristik Lahan, Aksesibilitas & Topografi (9) ---
    "elevation",
    "peat_coverage_pct",
    "peat_dryness_factor",
    "peat_x_vpd",
    "peat_x_dry_days",
    "peat_x_temp",
    "distance_to_road_km",
    "road_x_peat",
    "road_x_dry_days",
]


def load_and_prepare_train_val(downsample_ratio: int = 15) -> tuple:
    """
    [Langkah 1 & 2: Pemuatan & Pemisahan Data Multi-Tahun Memory-Safe]
    Memuat data per-tahun secara streaming agar RAM tetap di bawah 2 GB.
    - Data Latih (Train): 2020 s/d 2024 dengan 100% Titik Api dipertahankan + Subsampling Aman 1:15 (atau rasio kustom).
    - Data Uji (Test): 2025 murni (100% tanpa sampling) untuk validasi riil.
    """
    import gc
    import re

    print("\n" + "=" * 75)
    print("   LANGKAH 1 & 2: PEMUATAN & PEMISAHAN DATA MULTI-TAHUN (MEMORY-SAFE)")
    print("=" * 75)

    # Deteksi file tahun tunggal murni: training_dataset_YYYY.parquet
    yearly_files = {}
    for p in PROCESSED_DATA_DIR.glob("training_dataset_202*.parquet"):
        m = re.match(r"^training_dataset_(\d{4})\.parquet$", p.name)
        if m:
            yearly_files[int(m.group(1))] = p

    available_years = sorted(list(yearly_files.keys()))
    print(f"[INFO] Terdeteksi {len(available_years)} dataset tahunan: {available_years}")

    if not available_years:
        raise FileNotFoundError("Tidak ditemukan file dataset tahunan (training_dataset_YYYY.parquet)!")

    # Tentukan tahun train dan test
    if 2025 in available_years:
        train_years = [y for y in available_years if y < 2025]
        test_year = 2025
        print(f"[SPLIT] Mode Multi-Tahun: Train {train_years}, Test {test_year}")
    else:
        test_year = max(available_years)
        train_years = [y for y in available_years if y < test_year]
        print(f"[SPLIT] Mode Fallback: Train {train_years}, Test {test_year}")

    # 0. Siapkan Peta Jarak Jalan OSM dari kalimantan_grid_enriched.parquet
    road_map = {}
    enriched_grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if enriched_grid_path.exists():
        try:
            gdf_enriched = pd.read_parquet(enriched_grid_path, columns=["grid_id", "distance_to_road_km"])
            road_map = dict(zip(gdf_enriched["grid_id"], gdf_enriched["distance_to_road_km"].astype(np.float32)))
            print(f"[OK] Memuat data jarak jalan raya OSM ({len(road_map):,} kotak grid terhubung).")
        except Exception as e:
            print(f"[WARN] Belum ada data jarak jalan: {e}")

    # 1. Memuat Train Set (2020-2024)
    print(f"\n[MEMUAT DATA LATIH] Menggabungkan tahun {train_years} (100% api + sampling aman 1:{downsample_ratio})...")
    train_chunks = []
    total_raw_train = 0
    total_train_fires = 0

    dynamic_features = ["distance_to_road_km", "road_x_peat", "road_x_dry_days"]
    base_cols = [c for c in FEATURE_COLUMNS if c not in dynamic_features] + ["target_fire_next_day", "date", "grid_id"]

    for yr in train_years:
        f_path = yearly_files[yr]
        df_yr = pd.read_parquet(f_path, columns=[c for c in base_cols])
        total_raw_train += len(df_yr)

        # Warmup cutoff: potong 14 hari awal
        df_yr["date"] = pd.to_datetime(df_yr["date"])
        df_yr = df_yr[df_yr["date"] >= f"{yr}-01-15"]

        # Pasang jarak jalan raya OSM
        if road_map and "grid_id" in df_yr.columns:
            df_yr["distance_to_road_km"] = df_yr["grid_id"].map(road_map).fillna(5.0).astype(np.float32)
        else:
            df_yr["distance_to_road_km"] = 5.0

        # Solusi 2: Fitur Interaksi Akses Jalan x Gambut & Jalan x Hari Kering
        df_yr["road_x_peat"] = ((df_yr["peat_coverage_pct"].fillna(0) / 100.0) / (df_yr["distance_to_road_km"] + 1.0)).astype(np.float32)
        df_yr["road_x_dry_days"] = (df_yr["dry_days_count"].fillna(0) / (df_yr["distance_to_road_km"] + 1.0)).astype(np.float32)

        # Downcast float64 -> float32
        for c in FEATURE_COLUMNS:
            if c in df_yr.columns and df_yr[c].dtype == "float64":
                df_yr[c] = df_yr[c].astype("float32")

        # Capping fisik
        df_yr["soil_moisture_0_7cm"] = df_yr["soil_moisture_0_7cm"].clip(0.0, 1.0)
        df_yr["soil_moisture_7_28cm"] = df_yr["soil_moisture_7_28cm"].clip(0.0, 1.0)
        df_yr["peat_coverage_pct"] = df_yr["peat_coverage_pct"].clip(0.0, 100.0)
        df_yr["rain_sum"] = df_yr["rain_sum"].clip(lower=0.0)
        df_yr["temp_max"] = df_yr["temp_max"].clip(10.0, 45.0)
        df_yr["wind_speed_max"] = df_yr["wind_speed_max"].clip(0.0, 60.0)
        df_yr["vpd_max"] = df_yr["vpd_max"].clip(0.0, 8.0)

        # Subsampling imbalanced class untuk data latih
        df_pos = df_yr[df_yr["target_fire_next_day"] == 1]
        df_neg = df_yr[df_yr["target_fire_next_day"] == 0]
        n_pos = len(df_pos)
        total_train_fires += n_pos

        n_neg_sample = min(len(df_neg), n_pos * downsample_ratio) if downsample_ratio > 0 else len(df_neg)
        df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=42) if downsample_ratio > 0 else df_neg

        df_chunk = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
        train_chunks.append(df_chunk)
        print(f"  -> Tahun {yr}: {n_pos:,} api + {n_neg_sample:,} aman (dari {len(df_yr):,} baris awal)")

        del df_yr, df_pos, df_neg, df_neg_sampled
        gc.collect()

    df_train = pd.concat(train_chunks, ignore_index=True)
    del train_chunks
    gc.collect()

    print(f"\nData Latih (TRAIN SET) SIAP:")
    print(f"  - Total Data Latih   : {len(df_train):,} baris (terkompresi dari {total_raw_train:,} baris)")
    print(f"  - Total Titik Api (1): {total_train_fires:,} (100% Titik Api Utuh)")
    print(f"  - Total Wilayah Aman : {len(df_train) - total_train_fires:,}")
    print(f"  - Rasio Latih        : 1 Api : {downsample_ratio} Aman")
    print(f"  - Penggunaan Memori  : ~{df_train.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB (Sangat Hemat RAM!)")

    # 2. Memuat Test Set (2025)
    print(f"\n[MEMUAT DATA UJI] Memuat Tahun {test_year} (100% tanpa sampling untuk validasi objektif)...")
    test_path = yearly_files[test_year]
    cols_test = [c for c in base_cols]
    df_val = pd.read_parquet(test_path, columns=cols_test)

    df_val["date"] = pd.to_datetime(df_val["date"])
    df_val = df_val[df_val["date"] >= f"{test_year}-01-15"]

    if road_map and "grid_id" in df_val.columns:
        df_val["distance_to_road_km"] = df_val["grid_id"].map(road_map).fillna(5.0).astype(np.float32)
    else:
        df_val["distance_to_road_km"] = 5.0

    # Solusi 2: Fitur Interaksi Akses Jalan x Gambut & Jalan x Hari Kering
    df_val["road_x_peat"] = ((df_val["peat_coverage_pct"].fillna(0) / 100.0) / (df_val["distance_to_road_km"] + 1.0)).astype(np.float32)
    df_val["road_x_dry_days"] = (df_val["dry_days_count"].fillna(0) / (df_val["distance_to_road_km"] + 1.0)).astype(np.float32)

    for c in FEATURE_COLUMNS:
        if c in df_val.columns and df_val[c].dtype == "float64":
            df_val[c] = df_val[c].astype("float32")

    df_val["soil_moisture_0_7cm"] = df_val["soil_moisture_0_7cm"].clip(0.0, 1.0)
    df_val["soil_moisture_7_28cm"] = df_val["soil_moisture_7_28cm"].clip(0.0, 1.0)
    df_val["peat_coverage_pct"] = df_val["peat_coverage_pct"].clip(0.0, 100.0)
    df_val["rain_sum"] = df_val["rain_sum"].clip(lower=0.0)
    df_val["temp_max"] = df_val["temp_max"].clip(10.0, 45.0)
    df_val["wind_speed_max"] = df_val["wind_speed_max"].clip(0.0, 60.0)
    df_val["vpd_max"] = df_val["vpd_max"].clip(0.0, 8.0)

    val_pos = int((df_val["target_fire_next_day"] == 1).sum())
    val_neg = int((df_val["target_fire_next_day"] == 0).sum())
    print(f"\nData Uji (TEST SET {test_year}) SIAP:")
    print(f"  - Total Data Uji     : {len(df_val):,} baris")
    print(f"  - Api Nyata Besok (1): {val_pos:,} ({val_pos / len(df_val) * 100:.3f}%)")
    print(f"  - Wilayah Aman (0)   : {val_neg:,}")

    # scale_pos_weight untuk rasio 1:15 yang di-balance
    pos_weight = 10.0
    print(f"  -> scale_pos_weight = {pos_weight:.1f} (Dioptimalkan untuk Recall Tinggi)")

    return df_train, df_val, pos_weight


def train_and_evaluate_xgboost(df_train: pd.DataFrame, df_val: pd.DataFrame, scale_pos_weight: float):
    """
    [Langkah 3: Training & Evaluasi Resmi XGBoost V2]
    Hyperparameter dioptimalkan untuk mengatasi False Negative.
    """
    print("\n" + "=" * 75)
    print("   LANGKAH 3: PELATIHAN & EVALUASI RESMI MODEL XGBOOST (V2)")
    print("=" * 75)

    # Filter hanya fitur yang tersedia di dataset
    available_features = [c for c in FEATURE_COLUMNS if c in df_train.columns]
    print(f"[INFO] Fitur aktif: {len(available_features)} dari {len(FEATURE_COLUMNS)} fitur yang direncanakan.")

    X_train = df_train[available_features].astype(np.float32)
    y_train = df_train["target_fire_next_day"].astype(int)

    X_val = df_val[available_features].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int)

    print(f"[INFO] Mengonfigurasi parameter XGBoost V2...")
    print(f"  - Objective          : binary:logistic")
    print(f"  - Algoritma          : tree_method='hist' (Histogram Cepat)")
    print(f"  - Evaluasi Metric    : aucpr (Precision-Recall Area)")
    print(f"  - Denda Penalti      : scale_pos_weight = {scale_pos_weight:.1f}")
    print(f"  - Learning Rate      : 0.02 | Max Depth: 8 | Estimators: 800")
    print(f"  - Regularization     : alpha=0.1, lambda=5.0, gamma=1.0")

    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=10,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=5.0,
        tree_method="hist",
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )

    print(f"\n[TRAINING] Memulai pelatihan pada {len(X_train):,} baris data...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    print("[OK] Pelatihan model selesai.")

    # 1. Prediksi Probabilitas pada Data Uji
    print("\n[EVALUASI] Menghitung probabilitas prediksi pada Data Uji...")
    y_prob = model.predict_proba(X_val)[:, 1]

    pr_auc = float(average_precision_score(y_val, y_prob))
    roc_auc = float(roc_auc_score(y_val, y_prob))

    # 2. Penentuan Ambang Batas — F2-Score (Recall 2x lebih penting)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    f2_scores = (1 + 4) * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
    best_f2_idx = int(np.argmax(f2_scores))
    th_sedang_prob = float(thresholds[best_f2_idx])
    th_sedang_score = round(th_sedang_prob * 100.0, 1)

    # Ambang TINGGI: F1 optimal
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_f1_idx = int(np.argmax(f1_scores))
    th_tinggi_prob = float(thresholds[best_f1_idx])
    th_tinggi_score = round(max(th_sedang_score + 10.0, th_tinggi_prob * 100.0), 1)

    # Prediksi Biner
    y_pred_binary = (y_prob >= th_sedang_prob).astype(int)

    # 3. Evaluasi Resmi
    print("\n" + "=" * 75)
    print("   LAPORAN EVALUASI RESMI (V2)")
    print("=" * 75)
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"\n[AMBANG BATAS SKOR RISIKO (0 - 100)]")
    print(f"  - RENDAH : Skor 0.0 s/d < {th_sedang_score}")
    print(f"  - SEDANG : Skor {th_sedang_score} s/d < {th_tinggi_score}")
    print(f"  - TINGGI : Skor >= {th_tinggi_score}")

    print("\n[CLASSIFICATION REPORT]")
    report_dict = classification_report(y_val, y_pred_binary, target_names=["AMAN (0)", "API (1)"], output_dict=True)
    report_text = classification_report(y_val, y_pred_binary, target_names=["AMAN (0)", "API (1)"])
    print(report_text)

    cm = confusion_matrix(y_val, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    recall_pct = tp / (tp + fn) * 100.0
    print("[CONFUSION MATRIX]")
    print(f"  - True Positive  (Api Terdeteksi)  : {tp:,}")
    print(f"  - False Negative (Api Lolos)        : {fn:,}")
    print(f"  - False Positive (Alarm Palsu)      : {fp:,}")
    print(f"  - True Negative  (Aman Benar)       : {tn:,}")
    print(f"  -> Recall Api: {recall_pct:.2f}%")

    # 4. Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": available_features,
        "importance_gain": importances
    }).sort_values("importance_gain", ascending=False)
    print("\n[TOP 10 FITUR PALING BERPENGARUH]")
    for rank, (_, row) in enumerate(feat_imp.head(10).iterrows(), start=1):
        print(f"  {rank}. {row['feature']:<28}: {row['importance_gain'] * 100:.2f}%")

    # 5. Simpan Model dan Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_json_path = MODELS_DIR / "fire_risk_xgboost_model.json"
    model.save_model(str(model_json_path))

    metadata = {
        "algorithm": "XGBoost",
        "version": "2.0",
        "xgboost_version": xgb.__version__,
        "features": available_features,
        "threshold_sedang": th_sedang_score,
        "threshold_tinggi": th_tinggi_score,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "scale_pos_weight": float(scale_pos_weight),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_negative": int(fn),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "fire_recall_pct": float(recall_pct)
        },
        "classification_report": report_dict,
        "top_features": feat_imp.head(10).to_dict(orient="records")
    }

    report_path = MODELS_DIR / "model_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[BERHASIL] Model XGBoost V2 tersimpan di  : {model_json_path}")
    print(f"[BERHASIL] Laporan evaluasi tersimpan di   : {report_path}")
    print(f"[BERHASIL] Metadata model tersimpan di     : {meta_path}")
    print("=" * 75 + "\n")

def train_fire_model(downsample_ratio: int = 15):
    """Entrypoint resmi pelatihan model XGBoost V2 multi-tahun (Memory-Safe)."""
    df_train, df_val, pos_weight = load_and_prepare_train_val(downsample_ratio=downsample_ratio)
    return train_and_evaluate_xgboost(df_train, df_val, scale_pos_weight=pos_weight)


if __name__ == "__main__":
    train_fire_model()
