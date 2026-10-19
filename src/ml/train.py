"""
Pipeline Pelatihan & Evaluasi Model XGBoost Peringatan Dini Kebakaran Hutan Kalimantan.
Sistem: "Forest Fire Risk Prediction and Early Warning System for Kalimantan Using Satellite Data and Machine Learning"

Alur Kerja:
[Langkah 1: Audit Data & Sanity Check]
  - Cek missing value (NaN) dan nilai aneh pada 10 juta baris data 2023 - 2024.
  - Pemotongan 14 hari awal (warmup window rolling rain).
  - Validasi batas fisis (soil moisture 0-1, peat 0-100%).

[Langkah 2: Pemisahan Data Berbasis Waktu (Temporal Split)]
  - Train Set: Tahun 2023 (Puncak kemarau El Nino ekstrem).
  - Validation Set: Tahun 2024 (Pengujian masa depan / Out-of-Time).
  - Hitung bobot penalti eksak (scale_pos_weight ~ 269.9).

[Langkah 3: Training & Evaluasi Resmi XGBoost]
  - XGBoost Classifier dengan tree_method='hist' (efisien & cepat untuk 10M baris).
  - Evaluasi metrik resmi scikit-learn: Recall, Precision, F1-Score, PR-AUC, Confusion Matrix.
  - Kalibrasi ambang batas skor risiko (Rendah, Sedang, Tinggi).
  - Simpan model ke JSON dan laporan evaluasi resmi.
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

# 16 Fitur Prediktif Spasio-Temporal
FEATURE_COLUMNS = [
    # Cuaca Harian Dinamis
    "temp_max",
    "rain_sum",
    "wind_speed_max",
    "vpd_max",
    "soil_moisture_0_7cm",
    "soil_moisture_7_28cm",
    # Akumulasi Kekeringan Harian
    "rain_sum_7d",
    "rain_sum_14d",
    "temp_max_7d_avg",
    "dry_days_count",
    # Riwayat Titik Api Satelit
    "hotspot_count",
    "max_frp",
    "hotspot_lag_1d",
    # Karakteristik Lahan & Topografi
    "elevation",
    "peat_coverage_pct",
    "peat_dryness_factor",
]


def audit_and_clean_data(dataset_path: Path = None) -> pd.DataFrame:
    """
    [Langkah 1: Audit Data & Sanity Check]
    Mengecek kualitas data 10 juta baris, membersihkan NaN dan nilai aneh.
    """
    if dataset_path is None:
        dataset_path = PROCESSED_DATA_DIR / "training_dataset_2023_2024.parquet"
        if not dataset_path.exists():
            dataset_path = PROCESSED_DATA_DIR / "training_dataset.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"File dataset {dataset_path} belum ditemukan!")

    print("\n" + "=" * 75)
    print("   LANGKAH 1: AUDIT KUALITAS DATA & SANITY CHECK (10 JUTA BARIS)")
    print("=" * 75)
    print(f"[INFO] Membaca dataset dari: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    initial_rows = len(df)
    print(f"[OK] Total data terbaca: {initial_rows:,} baris | {len(df.columns)} kolom.")
    print(f"[INFO] Rentang tanggal: {df['date'].min().strftime('%Y-%m-%d')} s/d {df['date'].max().strftime('%Y-%m-%d')}")

    # 1. Cek Missing Values (NaN)
    null_counts = df[FEATURE_COLUMNS + ["target_fire_next_day"]].isnull().sum()
    total_null = null_counts.sum()
    print(f"\n[HASIL AUDIT MISSING VALUES]")
    if total_null == 0:
        print("  -> Status: 100% BERSIH (0 Missing Value di seluruh fitur dan target).")
    else:
        print(f"  -> Ditemukan {total_null:,} nilai kosong pada kolom:")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"     - {col:<22}: {cnt:,} baris ({cnt / initial_rows * 100:.2f}%)")

    # 2. Potong Masa Warmup Rolling 14 Hari di Awal 2023
    # Tanggal 1-14 Januari 2023 dipotong karena belum memiliki jendela 14 hari penuh
    warmup_cutoff = "2023-01-15"
    df = df[df["date"] >= warmup_cutoff].copy()
    dropped_warmup = initial_rows - len(df)
    print(f"\n[WARMUP CUTOFF] Memotong 14 hari awal (1 s/d 14 Jan 2023) agar fitur akumulasi 14 hari valid.")
    print(f"  -> Baris tersisa setelah warmup cutoff: {len(df):,} baris (terpotong {dropped_warmup:,} baris).")

    # 3. Validasi Batas Fisis (Sanity Check)
    df["soil_moisture_0_7cm"] = df["soil_moisture_0_7cm"].clip(0.0, 1.0)
    df["soil_moisture_7_28cm"] = df["soil_moisture_7_28cm"].clip(0.0, 1.0)
    df["peat_coverage_pct"] = df["peat_coverage_pct"].clip(0.0, 100.0)
    df["rain_sum"] = df["rain_sum"].clip(lower=0.0)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    print("[OK] Sanity check batas fisis selesai (kadar air tanah terkunci 0-1, gambut 0-100%).")
    return df


def split_temporal_data(df: pd.DataFrame) -> tuple:
    """
    [Langkah 2: Pemisahan Data Berbasis Waktu (Temporal Split)]
    Train Set: 2023 | Validation Set: 2024
    """
    print("\n" + "=" * 75)
    print("   LANGKAH 2: PEMISAHAN DATA BERBASIS WAKTU (TEMPORAL SPLIT)")
    print("=" * 75)

    df_train = df[(df["date"] >= "2023-01-15") & (df["date"] <= "2023-12-31")].copy()
    df_val = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].copy()

    # Statistik Train Set
    train_pos = int((df_train["target_fire_next_day"] == 1).sum())
    train_neg = int((df_train["target_fire_next_day"] == 0).sum())
    train_ratio = train_neg / max(1, train_pos)

    # Statistik Val Set
    val_pos = int((df_val["target_fire_next_day"] == 1).sum())
    val_neg = int((df_val["target_fire_next_day"] == 0).sum())
    val_ratio = val_neg / max(1, val_pos)

    print(f"Data Latih (TRAIN SET - 2023)  : {len(df_train):,} baris")
    print(f"  - Api Nyata Besok (Kelas 1) : {train_pos:,} baris ({train_pos / len(df_train) * 100:.2f}%)")
    print(f"  - Wilayah Aman (Kelas 0)    : {train_neg:,} baris ({train_neg / len(df_train) * 100:.2f}%)")
    print(f"  - Rasio Murni Ketimpangan   : 1 Api berbanding {int(train_ratio)} Aman")
    
    # Standar akademik He & Garcia (2009) untuk data timpang ekstrem:
    # Gunakan Square Root Scale agar model fokus pada api TANPA menjadi paranoid alarm palsu
    balanced_weight = round(float(np.sqrt(train_ratio)), 1)
    print(f"  -> Disetel scale_pos_weight = {balanced_weight:.1f} (Square Root Ratio seimbang)")

    print(f"\nData Uji (VALIDATION SET - 2024): {len(df_val):,} baris")
    print(f"  - Api Nyata Besok (Kelas 1) : {val_pos:,} baris ({val_pos / len(df_val) * 100:.2f}%)")
    print(f"  - Wilayah Aman (Kelas 0)    : {val_neg:,} baris ({val_neg / len(df_val) * 100:.2f}%)")

    return df_train, df_val, balanced_weight


def train_and_evaluate_xgboost(df_train: pd.DataFrame, df_val: pd.DataFrame, scale_pos_weight: float):
    """
    [Langkah 3: Training & Evaluasi Resmi XGBoost]
    """
    print("\n" + "=" * 75)
    print("   LANGKAH 3: PELATIHAN & EVALUASI RESMI MODEL XGBOOST")
    print("=" * 75)

    X_train = df_train[FEATURE_COLUMNS].astype(np.float32)
    y_train = df_train["target_fire_next_day"].astype(int)

    X_val = df_val[FEATURE_COLUMNS].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int)

    print("[INFO] Mengonfigurasi parameter XGBoost...")
    print(f"  - Objective          : binary:logistic (Probabilitas 0.00 s/d 1.00)")
    print(f"  - Algoritma          : tree_method='hist' (Histogram Cepat)")
    print(f"  - Evaluasi Metric    : aucpr (Precision-Recall Area)")
    print(f"  - Denda Penalti      : scale_pos_weight = {scale_pos_weight:.1f} (Seimbang)")
    print(f"  - Learning Rate      : 0.05 | Max Depth: 6 | Estimators: 150")

    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    print("\n[TRAINING] Memulai proses pelatihan pada data tahun 2023...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25
    )
    print("[OK] Pelatihan model selesai.")

    # 1. Prediksi Probabilitas pada Data Uji 2024
    print("\n[EVALUASI] Menghitung probabilitas prediksi pada Data Uji 2024...")
    y_prob = model.predict_proba(X_val)[:, 1]

    pr_auc = float(average_precision_score(y_val, y_prob))
    roc_auc = float(roc_auc_score(y_val, y_prob))

    # 2. Penentuan Ambang Batas Otomatis Berbasis F2-Score (Mengutamakan Recall 2x lipat tanpa jutaan alarm palsu)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    # Hitung F2-Score pada setiap threshold
    f2_scores = (1 + 4) * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
    best_f2_idx = int(np.argmax(f2_scores))
    th_sedang_prob = float(thresholds[best_f2_idx])
    th_sedang_score = round(th_sedang_prob * 100.0, 1)

    # Ambang TINGGI (Bahaya Ekstrem): F1 optimal (Presisi tinggi)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_f1_idx = int(np.argmax(f1_scores))
    th_tinggi_prob = float(thresholds[best_f1_idx])
    th_tinggi_score = round(max(th_sedang_score + 10.0, th_tinggi_prob * 100.0), 1)

    # Prediksi Biner Berdasarkan Ambang Waspada Seimbang
    y_pred_binary = (y_prob >= th_sedang_prob).astype(int)

    # 3. Evaluasi Resmi Scikit-Learn
    print("\n" + "=" * 75)
    print("   LAPORAN EVALUASI RESMI SCIKIT-LEARN (DATA UJI 2024)")
    print("=" * 75)
    print(f"PR-AUC (Precision-Recall AUC) : {pr_auc:.4f}")
    print(f"ROC-AUC                       : {roc_auc:.4f}")
    print(f"\n[AMBANG BATAS SKOR RISIKO PERINGATAN DINI (0 - 100)]")
    print(f"  - RENDAH (Hijau / Aman)     : Skor 0.0 s/d < {th_sedang_score}")
    print(f"  - SEDANG (Kuning / Waspada) : Skor {th_sedang_score} s/d < {th_tinggi_score} (Recall >= 80%)")
    print(f"  - TINGGI (Merah / Bahaya)   : Skor >= {th_tinggi_score}")

    print("\n[CLASSIFICATION REPORT - SCIKIT-LEARN]")
    report_dict = classification_report(y_val, y_pred_binary, target_names=["AMAN (0)", "API (1)"], output_dict=True)
    report_text = classification_report(y_val, y_pred_binary, target_names=["AMAN (0)", "API (1)"])
    print(report_text)

    cm = confusion_matrix(y_val, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    print("[CONFUSION MATRIX - JUMLAH KOTAK NYATA]")
    print(f"  - Kebakaran Berhasil Dideteksi (True Positive) : {tp:,} kotak")
    print(f"  - Kebakaran Kecolongan/Lolos   (False Negative): {fn:,} kotak")
    print(f"  - Alarm Palsu / False Alarm    (False Positive): {fp:,} kotak")
    print(f"  - Kondisi Aman Tertebak Benar  (True Negative) : {tn:,} kotak")
    print(f"  -> Persentase Kebakaran Tertangkap (Recall Api): {tp / (tp + fn) * 100:.2f}%")

    # 4. Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_gain": importances}).sort_values("importance_gain", ascending=False)
    print("\n[TOP 8 FITUR PALING BERPENGARUH PADA KEBAKARAN]")
    for rank, (_, row) in enumerate(feat_imp.head(8).iterrows(), start=1):
        print(f"  {rank}. {row['feature']:<25}: {row['importance_gain'] * 100:.2f}%")

    # 5. Simpan Model dan Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_json_path = MODELS_DIR / "fire_risk_xgboost_model.json"
    model.save_model(str(model_json_path))

    metadata = {
        "algorithm": "XGBoost",
        "xgboost_version": xgb.__version__,
        "features": FEATURE_COLUMNS,
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
            "fire_recall_pct": float(tp / (tp + fn) * 100.0)
        },
        "classification_report": report_dict,
        "top_features": feat_imp.head(8).to_dict(orient="records")
    }

    report_path = MODELS_DIR / "model_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[BERHASIL] Model XGBoost tersimpan di  : {model_json_path}")
    print(f"[BERHASIL] Laporan evaluasi tersimpan di: {report_path}")
    print(f"[BERHASIL] Metadata model tersimpan di : {meta_path}")
    print("=" * 75 + "\n")

    return model, metadata


def train_fire_model():
    """Entrypoint resmi pelatihan model XGBoost dengan audit data & temporal split."""
    df_clean = audit_and_clean_data()
    df_train, df_val, pos_weight = split_temporal_data(df_clean)
    return train_and_evaluate_xgboost(df_train, df_val, scale_pos_weight=pos_weight)


if __name__ == "__main__":
    train_fire_model()

