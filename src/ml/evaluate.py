"""
Script Evaluasi Resmi Model XGBoost Peringatan Dini Kebakaran Hutan Kalimantan.
Menggunakan metrik resmi scikit-learn pada data uji tahun 2024 (Out-of-Time Validation):
- Classification Report (Precision, Recall, F1-Score, Support)
- Confusion Matrix (True Positive, False Positive, True Negative, False Negative)
- PR-AUC (Precision-Recall Area Under Curve) & ROC-AUC
- Analisis Multi-Ambang Batas Risiko
- Top Fitur Paling Berpengaruh (Feature Importance Gain)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def run_evaluation():
    model_path = MODELS_DIR / "fire_risk_xgboost_model.json"
    meta_path = MODELS_DIR / "model_metadata.json"
    dataset_path = PROCESSED_DATA_DIR / "training_dataset_2023_2024.parquet"

    if not model_path.exists():
        print(f"[ERROR] Model {model_path} belum ditemukan! Silakan jalankan 'python src/ml/train.py' terlebih dahulu.")
        return

    # 1. Muat Metadata & Model
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        report_path = MODELS_DIR / "model_evaluation_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

    features = meta.get("features", [
        "temp_max", "rain_sum", "wind_speed_max", "vpd_max",
        "soil_moisture_0_7cm", "soil_moisture_7_28cm",
        "rain_sum_7d", "rain_sum_14d", "temp_max_7d_avg", "dry_days_count",
        "hotspot_count", "max_frp", "hotspot_lag_1d",
        "elevation", "peat_coverage_pct", "peat_dryness_factor"
    ])

    th_sedang_score = float(meta.get("threshold_sedang", 35.0))
    th_tinggi_score = float(meta.get("threshold_tinggi", 65.0))
    th_sedang_prob = th_sedang_score / 100.0
    th_tinggi_prob = th_tinggi_score / 100.0

    model = xgb.Booster()
    model.load_model(str(model_path))

    print("\n" + "=" * 80)
    print("      LAPORAN EVALUASI RESMI MODEL EARLY WARNING KEBAKARAN HUTAN")
    print("                (DATA UJI VALIDASI TEMPORAL TAHUN 2024)")
    print("=" * 80)

    # 2. Muat Data Uji Tahun 2024
    print(f"[INFO] Memuat Data Uji 2024 dari: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df_val = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].copy()

    X_val = df_val[features].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int)

    total_val = len(y_val)
    total_pos = int((y_val == 1).sum())
    total_neg = int((y_val == 0).sum())

    print(f"[OK] Total Data Uji 2024 : {total_val:,} baris kotak harian (5km x 5km)")
    print(f"     - Kebakaran Nyata Besok (1): {total_pos:,} kotak ({total_pos / total_val * 100:.2f}%)")
    print(f"     - Wilayah Aman (0)         : {total_neg:,} kotak ({total_neg / total_val * 100:.2f}%)")

    # 3. Hitung Prediksi Probabilitas
    print("[INFO] Menghitung inferensi probabilitas XGBoost...")
    dmat = xgb.DMatrix(X_val)
    y_prob = model.predict(dmat)

    roc_auc = float(roc_auc_score(y_val, y_prob))
    pr_auc = float(average_precision_score(y_val, y_prob))

    # 4. Ringkasan Metrik Global AUC
    print("\n" + "=" * 80)
    print("1. METRIK DISKRIMINASI GLOBAL (SCIKIT-LEARN AUC)")
    print("=" * 80)
    print(f"  - ROC-AUC Score : {roc_auc:.4f}  (Kemampuan memisahkan kotak aman vs api)")
    print(f"  - PR-AUC Score  : {pr_auc:.4f}  (Precision-Recall Area under Curve)")
    baseline_pr = total_pos / total_val
    print(f"  - Baseline PR   : {baseline_pr:.4f} (Model meningkat {pr_auc / max(1e-6, baseline_pr):.1f}x lipat dari tebakan acak)")

    # 5. Evaluasi Ambang Waspada (Sedang)
    y_pred_sedang = (y_prob >= th_sedang_prob).astype(int)
    cm_sedang = confusion_matrix(y_val, y_pred_sedang)
    tn, fp, fn, tp = cm_sedang.ravel()
    recall_api = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    prec_api = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"2. EVALUASI RESMI PADA TINGKAT RISIKO WASPADA / SEDANG (SKOR >= {th_sedang_score:.1f})")
    print("=" * 80)
    print(classification_report(y_val, y_pred_sedang, target_names=["AMAN (0)", "API (1)"], digits=4))

    print("Confusion Matrix Resmi Scikit-Learn [[TN, FP], [FN, TP]]:")
    print(f"  [[{tn:,},  {fp:,}],")
    print(f"   [{fn:,},  {tp:,}]]")
    print("\nRincian Operasional:")
    print(f"  - Api Nyata Berhasil Ditangkap (True Positive) : {tp:,} dari {total_pos:,} api ({recall_api:.2f}% Recall)")
    print(f"  - Api Kecolongan / Lolos       (False Negative): {fn:,} api")
    print(f"  - Alarm Palsu Lapangan         (False Positive): {fp:,} kotak dari {total_neg:,} kotak aman")
    print(f"  - Wilayah Aman Terkonfirmasi   (True Negative) : {tn:,} kotak")

    # 6. Analisis Sensitivitas Ambang Batas Operasional
    print("\n" + "=" * 80)
    print("3. TABEL RESPON OPERASIONAL PADA BERBAGAI TINGKAT AMBANG BATAS SKOR RISIKO")
    print("=" * 80)
    header = f"{'Ambang Skor':<12} | {'Recall Api':<12} | {'Precision':<12} | {'F1-Score':<10} | {'Api Tertangkap':<16} | {'Alarm Palsu':<14}"
    print(header)
    print("-" * 80)

    test_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    for th in test_thresholds:
        p_bin = (y_prob >= th).astype(int)
        c_mat = confusion_matrix(y_val, p_bin)
        _tn, _fp, _fn, _tp = c_mat.ravel()
        _rec = (_tp / (_tp + _fn)) * 100.0 if (_tp + _fn) > 0 else 0.0
        _prec = (_tp / (_tp + _fp)) * 100.0 if (_tp + _fp) > 0 else 0.0
        _f1 = (2 * _prec * _rec) / (_prec + _rec) / 100.0 if (_prec + _rec) > 0 else 0.0
        print(f"Skor >= {int(th*100):<4}  | {_rec:>10.2f} % | {_prec:>10.2f} % | {_f1:>10.4f} | {_tp:>6,} / {total_pos:,} | {_fp:>12,}")

    # 7. Top Fitur Berpengaruh
    print("\n" + "=" * 80)
    print("4. TOP 8 FAKTOR PREDRAWAT PALING BERPENGARUH PADA MODEL (XGBOOST GAIN)")
    print("=" * 80)
    score_dict = model.get_score(importance_type="gain")
    total_gain = sum(score_dict.values()) if score_dict else 1.0
    sorted_gain = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, g) in enumerate(sorted_gain[:8], start=1):
        pct = (g / total_gain) * 100.0
        print(f"  {rank}. {feat:<24}: {pct:.2f}% pengaruh")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_evaluation()
