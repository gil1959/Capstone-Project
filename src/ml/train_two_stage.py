"""
Sistem Peringatan Dini Kebakaran Hutan Kalimantan (Borneo Siaga)
Arsitektur 2 Tahap: Model Detektor (Stage 1) + Model Penjaga Alarm Palsu (Stage 2)

KONSEP KERJA DUA TAHAP:
1. TAHAP 1 (Detektor / Radar Api - XGBoost Penalti 7.5):
   - Menyapu seluruh 7,8 juta kotak harian pulau Kalimantan.
   - Target utama: RECALL TINGGI (Jangan ada api nyata yang lolos!).
2. TAHAP 2 (Model Penjaga / Verifikator Alarm Palsu - LightGBM Hard-Negative):
   - Dilatih KHUSUS pada data "kondisi panas tapi ternyata tidak terbakar".
   - Menguji kandidat yang dicurigai oleh Tahap 1.
3. LOGIKA JALUR CEPAT (VIP Fast-Track):
   - Api yang sudah jelas terbakar (Skor Stage 1 >= 70%) LANGSUNG LOLOS (tidak boleh diveto).
   - Kasus yang ragu-ragu (Skor 50% - 69%) WAJIB diverifikasi oleh Stage 2.
4. FILTER TETANGGA SPASIAL & AMBANG MUSIMAN DINAMIS:
   - Menyaring alarm terpencil dan menyapu bersih alarm palsu di musim hujan.

Cara Penggunaan:
  # Mode Simulasi / Preview (Aman, tidak menyimpan/menimpa model sistem):
  python src/ml/train_two_stage.py

  # Mode Simpan Permanen (Hanya jika Anda setuju dengan hasilnya):
  python src/ml/train_two_stage.py --save
"""

import sys
import gc
import json
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.ml.train import FEATURE_COLUMNS, load_and_prepare_train_val


def parse_args():
    parser = argparse.ArgumentParser(description="Latih Sistem 2 Tahap (Detektor + Penjaga Alarm Palsu)")
    parser.add_argument(
        "--penalty-s1",
        type=float,
        default=7.5,
        help="Penalti Stage 1 Detektor (scale_pos_weight). Default: 7.5 (Peka menangkap api)"
    )
    parser.add_argument(
        "--penalty-s2",
        type=float,
        default=1.5,
        help="Penalti Stage 2 Penjaga (scale_pos_weight). Default: 1.5 (Fokus teliti memangkas alarm palsu)"
    )
    parser.add_argument(
        "--downsample-ratio",
        type=int,
        default=15,
        help="Rasio sampling data latih non-api (1 api : N aman). Default: 15"
    )
    parser.add_argument(
        "--fast-track-threshold",
        type=float,
        default=0.70,
        help="Ambang batas Jalur Cepat (VIP) Stage 1 yang langsung lolos tanpa verifikasi. Default: 0.70"
    )
    parser.add_argument(
        "--verifier-threshold",
        type=float,
        default=0.28,
        help="Ambang batas kelulusan verifikasi Stage 2 untuk kasus ragu-ragu. Default: 0.28"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Simpan model 2 tahap ke folder data/models dan perbarui metadata sistem"
    )
    return parser.parse_args()


def train_two_stage_pipeline(args):
    print("\n" + "=" * 90)
    print("   PELATIHAN SISTEM 2 TAHAP: DETEKTOR API + PENJAGA ALARM PALSU (BORNEO SIAGA)")
    print("=" * 90)
    print(f"   - Tahap 1 (Detektor Api)  : XGBoost (Penalti = {args.penalty_s1:.1f}) -> Target: Anti Api Lolos")
    print(f"   - Tahap 2 (Penjaga Alarm) : LightGBM Hard-Negative (Penalti = {args.penalty_s2:.1f}) -> Target: Pangkas Alarm Palsu")
    print(f"   - Jalur Cepat (VIP)       : Skor >= {int(args.fast_track_threshold*100)}% langsung lolos tanpa verifikasi")
    print(f"   - Status Eksekusi         : {'SIMPAN PERMANEN (--save AKTIF)' if args.save else 'SIMULASI / PREVIEW (Aman, tidak menimpa)'}")
    print("=" * 90)

    # 1. Pemuatan Data Multi-Tahun
    df_train, df_val, _ = load_and_prepare_train_val(downsample_ratio=args.downsample_ratio)

    available_features = [c for c in FEATURE_COLUMNS if c in df_train.columns]
    print(f"\n[INFO] Fitur spatio-temporal aktif: {len(available_features)} fitur.")

    X_train = df_train[available_features].astype(np.float32)
    y_train = df_train["target_fire_next_day"].astype(int)

    X_val = df_val[available_features].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int).values

    total_fires_val = int(y_val.sum())
    total_val_rows = len(y_val)
    total_days_val = len(df_val["date"].unique())

    val_grid_ids = df_val["grid_id"].values
    val_dates = df_val["date"].values
    val_dt = pd.to_datetime(val_dates)
    val_months = val_dt.month
    mask_kemarau = np.isin(val_months, [7, 8, 9, 10])
    mask_hujan = ~mask_kemarau

    # Sampel validasi cepat untuk early stopping
    val_pos_idx = np.where(y_val == 1)[0]
    val_neg_idx = np.where(y_val == 0)[0]
    sample_neg_idx = np.random.choice(val_neg_idx, size=min(len(val_neg_idx), len(val_pos_idx) * 4), replace=False)
    quick_val_idx = np.concatenate([val_pos_idx, sample_neg_idx])
    X_val_quick = X_val.iloc[quick_val_idx]
    y_val_quick = y_val[quick_val_idx]

    # Helper inferensi memori-aman
    def chunked_predict(model, X, is_xgb=False, chunk_size=500000):
        preds = []
        for start_idx in range(0, len(X), chunk_size):
            end_idx = min(start_idx + chunk_size, len(X))
            chunk = X.iloc[start_idx:end_idx]
            if is_xgb:
                dmat = xgb.DMatrix(chunk)
                preds.append(model.get_booster().predict(dmat).astype(np.float32))
            else:
                preds.append(model.predict_proba(chunk)[:, 1].astype(np.float32))
        return np.concatenate(preds)

    # =========================================================================
    # 2. TAHAP 1: Melatih Model Detektor (XGBoost Penalti Tinggi)
    # =========================================================================
    print("\n" + "-" * 85)
    print(f"[TAHAP 1] Melatih Model Detektor Utama (XGBoost, Penalti = {args.penalty_s1:.1f})...")
    print("          Target: Memindai seluruh data & memastikan api nyata tidak ada yang lolos.")
    print("-" * 85)

    model_s1 = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=7,
        scale_pos_weight=args.penalty_s1,
        min_child_weight=8,
        gamma=1.0,
        reg_alpha=0.2,
        reg_lambda=4.0,
        tree_method="hist",
        subsample=0.75,
        colsample_bytree=0.75,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=25
    )
    model_s1.fit(X_train, y_train, eval_set=[(X_val_quick, y_val_quick)], verbose=False)
    print("[OK] Model Detektor (Tahap 1) selesai dilatih.")

    # =========================================================================
    # 3. MENYIAPKAN DATA KHUSUS TAHAP 2 (Hard Negatives / Alarm Palsu Panas)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[TAHAP 2] Mengidentifikasi 'Hard Negatives' (Kasus Panas yang Sering Bikin Salah Paham)...")
    print("-" * 85)

    p_train_s1 = model_s1.predict_proba(X_train)[:, 1]
    # Saring hanya data yang dicurigai oleh Tahap 1 (p >= 0.40)
    suspect_mask_train = (p_train_s1 >= 0.40)
    X_train_s2 = X_train[suspect_mask_train]
    y_train_s2 = y_train[suspect_mask_train]

    n_fires_s2 = int(y_train_s2.sum())
    n_false_s2 = int((y_train_s2 == 0).sum())
    print(f"  -> Data Latih Khusus Model Penjaga: {len(X_train_s2):,} baris.")
    print(f"     Berisi: {n_fires_s2:,} api nyata vs {n_false_s2:,} alarm palsu panas.")
    print("     (Model Penjaga dilatih KHUSUS untuk membedakan dua kondisi ini!).")

    # =========================================================================
    # 4. TAHAP 2: Melatih Model Penjaga Alarm Palsu (LightGBM Verifier)
    # =========================================================================
    print("\n" + "-" * 85)
    print(f"[TAHAP 2] Melatih Model Penjaga / Verifikator (LightGBM, Penalti = {args.penalty_s2:.1f})...")
    print("          Target: Menghabisi alarm palsu yang dicurigai salah oleh Tahap 1.")
    print("-" * 85)

    model_s2 = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        scale_pos_weight=args.penalty_s2,
        min_child_samples=30,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=4.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_s2.fit(X_train_s2, y_train_s2)
    print("[OK] Model Penjaga (Tahap 2) selesai dilatih.")

    # =========================================================================
    # 5. INFERENSI PADA DATA UJI 2025 PENUH (7,8 Juta Baris)
    # =========================================================================
    print("\n" + "-" * 85)
    print(f"[EVALUASI] Menjalankan Inferensi 2 Tahap pada {total_val_rows:,} baris data uji 2025...")
    print("-" * 85)

    print("      -> [Tahap 1] Memindai seluruh 7,8 juta kotak daratan Kalimantan...")
    prob_s1 = chunked_predict(model_s1, X_val, is_xgb=True)
    pr_auc_s1 = average_precision_score(y_val, prob_s1)
    roc_auc_s1 = roc_auc_score(y_val, prob_s1)

    # Identifikasi kotak yang dicurigai oleh Tahap 1 (p >= 0.45)
    suspect_indices = np.where(prob_s1 >= 0.45)[0]
    print(f"      -> [Tahap 1] Menemukan {len(suspect_indices):,} kotak mencurigakan ({len(X_val) - len(suspect_indices):,} kotak jelas aman).")

    print(f"      -> [Tahap 2] Model Penjaga memeriksa ke {len(suspect_indices):,} kotak mencurigakan...")
    prob_s2 = np.zeros(len(X_val), dtype=np.float32)
    if len(suspect_indices) > 0:
        X_suspect = X_val.iloc[suspect_indices]
        p_s2_suspect = chunked_predict(model_s2, X_suspect, is_xgb=False)
        prob_s2[suspect_indices] = p_s2_suspect

    # =========================================================================
    # 6. LOGIKA KEPUTUSAN SINERGIS 2 TAHAP (Dengan Jalur Cepat / VIP)
    # =========================================================================
    # Kasus A: Api yang sangat jelas (Skor Stage 1 >= fast_track) -> Lolos Langsung
    # Kasus B: Kasus ragu-ragu (Skor Stage 1 antara 50% s/d 69%) -> Lolos HANYA JIKA lolos Stage 2
    prob_final = prob_s1.copy()

    # Veto Filter: Jika di zona ragu-ragu (< fast_track_threshold) dan Stage 2 bilang tidak yakin (< verifier_threshold)
    mask_doubt = (prob_s1 >= 0.45) & (prob_s1 < args.fast_track_threshold)
    mask_vetoed = mask_doubt & (prob_s2 < args.verifier_threshold)

    # Pangkas probabilitas kotak yang terkena veto agar jatuh ke status RENDAH
    prob_final[mask_vetoed] = prob_final[mask_vetoed] * (prob_s2[mask_vetoed] / args.verifier_threshold)

    pr_auc_final = average_precision_score(y_val, prob_final)
    roc_auc_final = roc_auc_score(y_val, prob_final)

    print("\n" + "=" * 90)
    print("                    PERBANDINGAN METRIK GLOBAL DATA UJI 2025")
    print("=" * 90)
    print(f"  1. Tahap 1 Saja (XGBoost Standalone Penalti 7.5) : ROC-AUC = {roc_auc_s1:.4f} | PR-AUC = {pr_auc_s1:.4f}")
    print(f"  2. SISTEM 2 TAHAP (Detektor + Penjaga Alarm)     : ROC-AUC = {roc_auc_final:.4f} | PR-AUC = {pr_auc_final:.4f}")
    print("=" * 90)

    # =========================================================================
    # 7. TABEL 1: PERBANDINGAN SEBELUM VS SESUDAH MODEL PENJAGA (Skor 60%)
    # =========================================================================
    header = f"{'Metode':<24} | {'Recall Api':<12} | {'Presisi':<10} | {'Api Tertangkap':<16} | {'Api Kecolongan':<16} | {'Alarm Palsu Total':<18} | {'Alarm / Hari':<12}"
    print("\n" + "=" * 115)
    print("  TABEL 1: DAMPAK MODEL PENJAGA PADA AMBANG SKOR 60% (SEBELUM FILTER TETANGGA)")
    print("=" * 115)
    print(header)
    print("-" * 115)

    # Stage 1 Murni pada 60%
    pred_s1_60 = (prob_s1 >= 0.60).astype(int)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_val, pred_s1_60).ravel()
    rec1 = (tp1 / (tp1 + fn1)) * 100.0
    prec1 = (tp1 / (tp1 + fp1)) * 100.0
    print(f"{'Tahap 1 Saja (Detektor)':<24} | {rec1:>10.2f} % | {prec1:>8.2f} % | {tp1:>7,} titik      | {fn1:>7,} titik      | {fp1:>12,} kotak      | {fp1/total_days_val:>8.1f} /hari")

    # Sistem 2 Tahap pada 60%
    pred_final_60 = (prob_final >= 0.60).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_val, pred_final_60).ravel()
    rec2 = (tp2 / (tp2 + fn2)) * 100.0
    prec2 = (tp2 / (tp2 + fp2)) * 100.0
    print(f"{'Sistem 2 Tahap (Lengkap)':<24} | {rec2:>10.2f} % | {prec2:>8.2f} % | {tp2:>7,} titik      | {fn2:>7,} titik      | {fp2:>12,} kotak      | {fp2/total_days_val:>8.1f} /hari")
    print("-" * 115)
    fp_saved_stage2 = fp1 - fp2
    print(f"  [HASIL MODEL PENJAGA]: Berhasil memotong {fp_saved_stage2:,} alarm palsu (turun {(fp_saved_stage2/fp1)*100:.1f}%) sambil menjaga {tp2:,} api nyata tetap tertangkap!")
    print("=" * 115)

    # =========================================================================
    # 8. TABEL 2: RESPONS OPERASIONAL DENGAN FILTER TETANGGA SPASIAL AKTIF
    # =========================================================================
    print("\n" + "=" * 115)
    print("  TABEL 2: HASIL AKHIR SISTEM 2 TAHAP + FILTER TETANGGA SPASIAL AKTIF")
    print("=" * 115)
    print(f"{'Ambang Skor':<18} | {'Recall Api':<12} | {'Presisi':<10} | {'Api Tertangkap':<16} | {'Api Kecolongan':<16} | {'Alarm Palsu Total':<18} | {'Alarm / Hari':<12}")
    print("-" * 115)

    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    gdf_coords = pd.read_parquet(grid_path, columns=["grid_id", "lat", "lon"])
    coords = np.column_stack([gdf_coords["lat"].values, gdf_coords["lon"].values])
    tree = cKDTree(coords)
    neighbor_map = tree.query_ball_tree(tree, r=0.08)
    gid_to_idx = {gid: idx for idx, gid in enumerate(gdf_coords["grid_id"])}

    thresholds_to_test = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    summary_filtered = {}
    filtered_preds_dict = {}

    for th in thresholds_to_test:
        pred_bin = (prob_final >= th).astype(int)
        pred_filtered = pred_bin.copy()
        score_int = int(round(th * 100))

        for d in pd.unique(val_dates):
            mask_day = (val_dates == d)
            day_indices = np.where(mask_day)[0]
            active_gids = set(val_grid_ids[day_indices[pred_bin[day_indices] == 1]])

            for idx in day_indices:
                if pred_bin[idx] == 1:
                    gid = val_grid_ids[idx]
                    g_idx = gid_to_idx.get(gid)
                    if g_idx is not None:
                        n_indices = neighbor_map[g_idx]
                        has_active_neighbor = any(
                            gdf_coords.iloc[n]["grid_id"] in active_gids
                            for n in n_indices if n != g_idx
                        )
                        if not has_active_neighbor:
                            pred_filtered[idx] = 0

        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_val, pred_filtered).ravel()
        rec_f = (tp_f / (tp_f + fn_f)) * 100.0 if (tp_f + fn_f) > 0 else 0.0
        prec_f = (tp_f / (tp_f + fp_f)) * 100.0 if (tp_f + fp_f) > 0 else 0.0
        fp_daily_f = fp_f / total_days_val

        label = f"Skor >= {score_int}%"
        print(f"{label:<18} | {rec_f:>10.2f} % | {prec_f:>8.2f} % | {tp_f:>7,} titik      | {fn_f:>7,} titik      | {fp_f:>12,} kotak      | {fp_daily_f:>8.1f} /hari")

        summary_filtered[score_int] = {
            "threshold": float(th),
            "recall": round(float(rec_f), 2),
            "precision": round(float(prec_f), 2),
            "tp": int(tp_f),
            "fn": int(fn_f),
            "fp": int(fp_f),
            "tn": int(tn_f),
            "fp_daily": round(float(fp_daily_f), 1)
        }
        filtered_preds_dict[score_int] = pred_filtered
    print("=" * 115)

    # =========================================================================
    # 9. TABEL 3: EVALUASI AMBANG BATAS MUSIMAN DINAMIS (Kemarau vs Hujan)
    # =========================================================================
    print("\n" + "=" * 115)
    print("  TABEL 3: EVALUASI AMBANG BATAS MUSIMAN DINAMIS (KEMARAU 60% VS HUJAN 65%)")
    print("=" * 115)
    days_kemarau = len(np.unique(val_dates[mask_kemarau]))
    days_hujan = len(np.unique(val_dates[mask_hujan]))
    fires_kemarau = int(y_val[mask_kemarau].sum())
    fires_hujan = int(y_val[mask_hujan].sum())

    # Kemarau 60% vs Hujan 65% (Masuk Akal!)
    pred_dyn = np.zeros(len(y_val), dtype=int)
    pred_dyn[mask_kemarau] = filtered_preds_dict[60][mask_kemarau]
    pred_dyn[mask_hujan] = filtered_preds_dict[65][mask_hujan]

    tn_dyn, fp_dyn, fn_dyn, tp_dyn = confusion_matrix(y_val, pred_dyn).ravel()
    rec_dyn = (tp_dyn / (tp_dyn + fn_dyn)) * 100.0
    prec_dyn = (tp_dyn / (tp_dyn + fp_dyn)) * 100.0

    tp_k = int(((pred_dyn[mask_kemarau] == 1) & (y_val[mask_kemarau] == 1)).sum())
    fp_k = int(((pred_dyn[mask_kemarau] == 1) & (y_val[mask_kemarau] == 0)).sum())
    rec_k = (tp_k / fires_kemarau) * 100.0 if fires_kemarau > 0 else 0.0

    tp_h = int(((pred_dyn[mask_hujan] == 1) & (y_val[mask_hujan] == 1)).sum())
    fp_h = int(((pred_dyn[mask_hujan] == 1) & (y_val[mask_hujan] == 0)).sum())
    rec_h = (tp_h / fires_hujan) * 100.0 if fires_hujan > 0 else 0.0

    fp_saved_dyn = summary_filtered[60]["fp"] - fp_dyn

    print(f"  1. Musim Kemarau (Juli s/d Oktober - {days_kemarau} Hari, Ambang Skor 60%):")
    print(f"     - Api Tertangkap      : {tp_k:,} dari {fires_kemarau:,} titik api ({rec_k:.2f}% Recall)")
    print(f"     - Alarm Palsu Kemarau : {fp_k:,} kotak ({fp_k / days_kemarau:.1f} kotak/hari)")
    print(f"\n  2. Musim Hujan/Transisi (November s/d Juni - {days_hujan} Hari, Ambang Skor 65%):")
    print(f"     - Api Tertangkap      : {tp_h:,} dari {fires_hujan:,} titik api ({rec_h:.2f}% Recall)")
    print(f"     - Alarm Palsu Hujan   : {fp_h:,} kotak ({fp_h / days_hujan:.1f} kotak/hari)")
    print(f"\n  3. TOTAL OPERASIONAL SETAHUN PENUH DENGAN SISTEM 2 TAHAP + AMBANG DINAMIS:")
    print(f"     - Total Api Tertangkap     : {tp_dyn:,} dari {total_fires_val:,} titik ({rec_dyn:.2f}% Recall)")
    print(f"     - Total Alarm Palsu        : {fp_dyn:,} kotak (Rata-rata {fp_dyn / total_days_val:.1f} kotak/hari)")
    print(f"     - Presisi Operasional      : {prec_dyn:.2f}%")
    print(f"     - ALARM PALSU TERPANGKAS   : {fp_saved_dyn:,} kotak berhasil dieliminasi berkat ambang musiman dinamis!")
    print("=" * 115)

    # =========================================================================
    # 10. PENYIMPANAN MODEL (--save)
    # =========================================================================
    if args.save:
        print("\n" + "=" * 85)
        print("[PENYIMPANAN] Menyimpan model sistem 2 tahap ke folder data/models...")
        print("=" * 85)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        xgb_file = MODELS_DIR / "fire_risk_xgboost_model.json"
        lgb_file = MODELS_DIR / "fire_risk_verifier_model.txt"
        meta_file = MODELS_DIR / "model_metadata.json"
        rep_file = MODELS_DIR / "model_evaluation_report.json"

        # Simpan Stage 1 XGBoost
        model_s1.save_model(str(xgb_file))
        print(f"  [1/2] Model Detektor (Stage 1 XGBoost) tersimpan di  : {xgb_file.name}")

        # Simpan Stage 2 Verifier LightGBM
        model_s2.booster_.save_model(str(lgb_file))
        print(f"  [2/2] Model Penjaga (Stage 2 Verifier) tersimpan di : {lgb_file.name}")

        metadata = {
            "algorithm": "Two-Stage Cascade (Stage 1 XGBoost Detector + Stage 2 LightGBM Verifier)",
            "version": "2.3-two-stage-cascade",
            "features": available_features,
            "feature_count": len(available_features),
            "stage1_config": {
                "model_type": "xgboost",
                "scale_pos_weight": float(args.penalty_s1),
                "fast_track_threshold": float(args.fast_track_threshold)
            },
            "stage2_config": {
                "model_type": "lightgbm_verifier",
                "scale_pos_weight": float(args.penalty_s2),
                "verifier_threshold": float(args.verifier_threshold)
            },
            "downsample_ratio": int(args.downsample_ratio),
            "threshold_sedang": 60.0,
            "threshold_tinggi": 85.0,
            "seasonal_dynamic_thresholds": {
                "dry_season_months": [7, 8, 9, 10],
                "dry_threshold_sedang": 60.0,
                "dry_threshold_tinggi": 85.0,
                "wet_threshold_sedang": 65.0,
                "wet_threshold_tinggi": 90.0
            },
            "metrics_global": {
                "roc_auc_two_stage": round(float(roc_auc_final), 4),
                "pr_auc_two_stage": round(float(pr_auc_final), 4),
                "roc_auc_stage1": round(float(roc_auc_s1), 4),
                "pr_auc_stage1": round(float(pr_auc_s1), 4)
            },
            "confusion_matrix_dynamic": {
                "true_positive": int(tp_dyn),
                "false_negative": int(fn_dyn),
                "false_positive": int(fp_dyn),
                "true_negative": int(tn_dyn),
                "fire_recall_pct": round(float(rec_dyn), 2),
                "precision_pct": round(float(prec_dyn), 2)
            },
            "threshold_sensitivity_filtered": summary_filtered
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  [OK] Metadata sistem diperbarui di : {meta_file.name}")
        print(f"  [OK] Laporan evaluasi disimpan di  : {rep_file.name}")
        print("\n[SUKSES] Seluruh model Sistem 2 Tahap dan metadata berhasil disimpan!")
    else:
        print("\n[INFO] Mode simulasi selesai tanpa penyimpanan.")
        print("       Untuk melatih dan MENYIMPAN model ini ke sistem, jalankan:")
        print("       python src/ml/train_two_stage.py --save\n")


if __name__ == "__main__":
    args = parse_args()
    train_two_stage_pipeline(args)
