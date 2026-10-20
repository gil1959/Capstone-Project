"""
Menghitung titik-titik kurva ROC dan Precision-Recall resmi Scikit-Learn
langsung dari hasil inferensi model XGBoost pada data uji tahun 2024.
Hasilnya disimpan ke data/models/roc_pr_curve_data.json agar dapat digambar
secara presisi 100% dan titik ambang (Sedang & Tinggi) persis berada di atas kurva.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

def compute_curves():
    model_path = MODELS_DIR / "fire_risk_xgboost_model.json"
    dataset_path = PROCESSED_DATA_DIR / "training_dataset_2023_2024.parquet"
    if not dataset_path.exists():
        dataset_path = PROCESSED_DATA_DIR / "training_dataset.parquet"

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    th_sedang_score = float(meta.get("threshold_sedang", 28.4))
    th_tinggi_score = float(meta.get("threshold_tinggi", 38.4))
    th_sedang_prob = th_sedang_score / 100.0
    th_tinggi_prob = th_tinggi_score / 100.0

    print("[1/5] Memuat model XGBoost...")
    model = xgb.Booster()
    model.load_model(str(model_path))

    print(f"[2/5] Membaca data uji 2024 dari: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df_val = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].copy()

    X_val = df_val[features].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int).values

    print(f"[3/5] Inferensi probabilitas pada {len(y_val):,} sampel data uji...")
    dmat = xgb.DMatrix(X_val)
    y_prob = model.predict(dmat)

    roc_auc = float(roc_auc_score(y_val, y_prob))
    pr_auc = float(average_precision_score(y_val, y_prob))
    print(f"      -> ROC-AUC: {roc_auc:.4f}")
    print(f"      -> PR-AUC : {pr_auc:.4f}")

    print("[4/5] Menghitung kurva ROC dan Precision-Recall scikit-learn...")
    fpr_arr, tpr_arr, roc_th = roc_curve(y_val, y_prob)
    prec_arr, rec_arr, pr_th = precision_recall_curve(y_val, y_prob)

    # Cari titik exact untuk Ambang Sedang (threshold 28.4%)
    idx_roc_sedang = int(np.argmin(np.abs(roc_th - th_sedang_prob)))
    pt_roc_sedang = {
        "fpr": float(fpr_arr[idx_roc_sedang]),
        "tpr": float(tpr_arr[idx_roc_sedang]),
        "threshold": float(roc_th[idx_roc_sedang])
    }
    idx_pr_sedang = int(np.argmin(np.abs(pr_th - th_sedang_prob)))
    pt_pr_sedang = {
        "recall": float(rec_arr[idx_pr_sedang]),
        "precision": float(prec_arr[idx_pr_sedang]),
        "threshold": float(pr_th[idx_pr_sedang])
    }

    # Cari titik exact untuk Ambang Tinggi (threshold 38.4%)
    idx_roc_tinggi = int(np.argmin(np.abs(roc_th - th_tinggi_prob)))
    pt_roc_tinggi = {
        "fpr": float(fpr_arr[idx_roc_tinggi]),
        "tpr": float(tpr_arr[idx_roc_tinggi]),
        "threshold": float(roc_th[idx_roc_tinggi])
    }
    idx_pr_tinggi = int(np.argmin(np.abs(pr_th - th_tinggi_prob)))
    pt_pr_tinggi = {
        "recall": float(rec_arr[idx_pr_tinggi]),
        "precision": float(prec_arr[idx_pr_tinggi]),
        "threshold": float(pr_th[idx_pr_tinggi])
    }

    print("Titik Ambang Sedang (Kuning):")
    print(f"  ROC: FPR={pt_roc_sedang['fpr']:.5f}, TPR={pt_roc_sedang['tpr']:.5f} (th={pt_roc_sedang['threshold']:.4f})")
    print(f"  PR : Recall={pt_pr_sedang['recall']:.5f}, Precision={pt_pr_sedang['precision']:.5f} (th={pt_pr_sedang['threshold']:.4f})")

    print("Titik Ambang Tinggi (Merah):")
    print(f"  ROC: FPR={pt_roc_tinggi['fpr']:.5f}, TPR={pt_roc_tinggi['tpr']:.5f} (th={pt_roc_tinggi['threshold']:.4f})")
    print(f"  PR : Recall={pt_pr_tinggi['recall']:.5f}, Precision={pt_pr_tinggi['precision']:.5f} (th={pt_pr_tinggi['threshold']:.4f})")

    # Downsample kurva agar ringan dan cepat dirender (e.g. 500 titik terdistribusi merata)
    # Gunakan interpolasi linier pada grid linspace agar bentuk aslinya 100% terjaga
    sample_fpr = np.linspace(0, 1, 500)
    # np.interp butuh x monotonik naik (fpr monotonik naik)
    sample_tpr = np.interp(sample_fpr, fpr_arr, tpr_arr)
    # Pastikan batas 0 dan 1
    sample_tpr[0] = 0.0
    sample_tpr[-1] = 1.0

    # Untuk PR curve: recall turun dari 1 ke 0 di array precision_recall_curve
    # Balik array agar recall monotonik naik untuk interp
    idx_sort_pr = np.argsort(rec_arr)
    rec_sorted = rec_arr[idx_sort_pr]
    prec_sorted = prec_arr[idx_sort_pr]

    sample_recall = np.linspace(0.001, 1.0, 500)
    sample_prec = np.interp(sample_recall, rec_sorted, prec_sorted)

    baseline_prevalence = float((y_val == 1).sum() / len(y_val))

    output_data = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "baseline_prevalence": round(baseline_prevalence, 6),
        "roc_curve": {
            "fpr": [round(float(v), 5) for v in sample_fpr],
            "tpr": [round(float(v), 5) for v in sample_tpr]
        },
        "pr_curve": {
            "recall": [round(float(v), 5) for v in sample_recall],
            "precision": [round(float(v), 5) for v in sample_prec]
        },
        "threshold_sedang": {
            "score": th_sedang_score,
            "roc_point": pt_roc_sedang,
            "pr_point": pt_pr_sedang
        },
        "threshold_tinggi": {
            "score": th_tinggi_score,
            "roc_point": pt_roc_tinggi,
            "pr_point": pt_pr_tinggi
        }
    }

    out_file = MODELS_DIR / "roc_pr_curve_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"[5/5] Data kurva tersimpan di: {out_file}")

if __name__ == "__main__":
    compute_curves()
