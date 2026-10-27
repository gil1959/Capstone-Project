"""
Script Pelatihan & Evaluasi Sinergi Cerdas (Smart Ensemble: XGBoost + LightGBM + Random Forest)
Sistem Peringatan Dini Kebakaran Hutan Kalimantan (Borneo Siaga).

STRATEGI SINERGI CERDAS (BUKAN RATA-RATA BIASA):
1. XGBoost + LightGBM sebagai "Radar Penjaga Api (Dual Boosters)":
   - XGBoost (pembelahan depth-wise) dan LightGBM (pembelahan leaf-wise) dilatih dengan penalti sensitif (default 7.5).
   - Keduanya saling melengkapi: jika salah satu model mendeteksi api nyata, kebakaran TIDAK AKAN LOLOS!
2. Random Forest sebagai "Hakim Verifikasi Konsensus (Anti Alarm Palsu)":
   - 120 pohon independen bertindak sebagai filter verifikasi stabilitas fisik lahan.
   - Jika konsensus pohon RF menilai wilayah sangat aman (p_rf < 0.20) pada alarm yang masih samar, alarm palsu langsung dipangkas!
3. Filter Tetangga Spasial:
   - Menyaring kotak terpencil yang berdiri sendirian.
4. Ambang Batas Musiman Dinamis:
   - Kemarau (Juli-Oktober): Ambang 60% (sensitif agar api tertangkap).
   - Hujan (November-Juni): Ambang 75% (ketat untuk menyapu bersih alarm palsu saat lahan basah).

Cara Penggunaan:
  # Mode Sinergi Cerdas (Preview / Simulasi):
  python src/ml/train_ensemble.py

  # Mode Sinergi Cerdas (Simpan Permanen ke data/models):
  python src/ml/train_ensemble.py --save

  # Opsi Kustomisasi:
  python src/ml/train_ensemble.py --penalty 7.5 --downsample-ratio 15 --save
  python src/ml/train_ensemble.py --only-xgb --penalty 7.5
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import joblib

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.ml.train import FEATURE_COLUMNS, load_and_prepare_train_val


def parse_args():
    parser = argparse.ArgumentParser(description="Latih Sinergi Cerdas XGBoost + LightGBM + Random Forest")
    parser.add_argument(
        "--penalty",
        type=float,
        default=7.5,
        help="Bobot penalti minoritas (scale_pos_weight). Default: 7.5 (menjaga agar api nyata tidak lolos)"
    )
    parser.add_argument(
        "--downsample-ratio",
        type=int,
        default=15,
        help="Rasio sampling data latih non-api (1 api : N aman). Default: 15"
    )
    parser.add_argument(
        "--mode",
        choices=["smart", "blend"],
        default="smart",
        help="Mode ensemble: 'smart' (Dual Booster + RF Verifier) atau 'blend' (Rata-rata biasa). Default: smart"
    )
    parser.add_argument(
        "--only-xgb",
        action="store_true",
        help="Latih HANYA model XGBoost murni (lewati LightGBM dan Random Forest)."
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=350,
        help="Jumlah pohon XGBoost. Default: 350"
    )
    parser.add_argument(
        "--lgb-estimators",
        type=int,
        default=350,
        help="Jumlah pohon LightGBM. Default: 350"
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=120,
        help="Jumlah pohon Random Forest. Default: 120"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Simpan model ke folder data/models dan perbarui metadata sistem"
    )
    return parser.parse_args()


def train_ensemble_pipeline(args):
    print("\n" + "=" * 90)
    print("   PELATIHAN MODEL SINERGI CERDAS: XGBOOST + LIGHTGBM + RANDOM FOREST")
    print("   Sistem Peringatan Dini Kebakaran Hutan Kalimantan (Borneo Siaga)")
    print(f"   - Mode Ensemble     : {'XGBoost Standalone' if args.only_xgb else ('Sinergi Cerdas (Dual Booster + RF Verifier)' if args.mode == 'smart' else 'Rata-Rata Linier (Blend)')}")
    print(f"   - Penalti Deteksi   : {args.penalty:.1f} (Dioptimalkan agar api nyata tidak lolos)")
    print(f"   - Rasio Data Latih  : 1 Api : {args.downsample_ratio} Aman (Multi-Tahun 2020-2024)")
    print(f"   - Status Eksekusi   : {'SIMPAN PERMANEN (--save AKTIF)' if args.save else 'SIMULASI / PREVIEW (Aman, tidak menimpa)'}")
    print("=" * 90)

    # 1. Pemuatan Data Latih (2020-2024) dan Uji (2025)
    df_train, df_val, _ = load_and_prepare_train_val(downsample_ratio=args.downsample_ratio)

    available_features = [c for c in FEATURE_COLUMNS if c in df_train.columns]
    print(f"\n[INFO] Fitur prediktif spatio-temporal aktif: {len(available_features)} fitur.")
    if "road_x_peat" in available_features and "road_x_dry_days" in available_features:
        print("       [OK] Fitur interaksi jalan raya OSM (Solusi 2) berhasil diaktifkan.")

    X_train = df_train[available_features].astype(np.float32)
    y_train = df_train["target_fire_next_day"].astype(int)

    X_val = df_val[available_features].astype(np.float32)
    y_val = df_val["target_fire_next_day"].astype(int).values

    total_fires_val = int(y_val.sum())
    total_val_rows = len(y_val)
    total_days_val = len(df_val["date"].unique())
    print(f"[INFO] Data Uji 2025 Penuh : {total_val_rows:,} kotak harian ({total_days_val} hari pemantauan)")
    print(f"       Titik Api Nyata (1) : {total_fires_val:,} kotak ({total_fires_val/total_val_rows*100:.3f}%)")

    # Siapkan sampel validasi cepat untuk early stopping XGBoost & LightGBM
    val_pos_idx = np.where(y_val == 1)[0]
    val_neg_idx = np.where(y_val == 0)[0]
    sample_neg_idx = np.random.choice(val_neg_idx, size=min(len(val_neg_idx), len(val_pos_idx) * 4), replace=False)
    quick_val_idx = np.concatenate([val_pos_idx, sample_neg_idx])
    np.random.shuffle(quick_val_idx)

    X_val_quick = X_val.iloc[quick_val_idx]
    y_val_quick = y_val[quick_val_idx]

    # Helper inferensi memori-aman per chunk
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
    # 2. Pelatihan Model 1: XGBoost (Depth-Wise Gradient Boosting)
    # =========================================================================
    print("\n" + "-" * 85)
    print(f"[1/3] Melatih Model 1: XGBoost (Penalti scale_pos_weight = {args.penalty:.1f})...")
    print("-" * 85)

    model_xgb = xgb.XGBClassifier(
        n_estimators=args.xgb_estimators,
        learning_rate=0.03,
        max_depth=7,
        scale_pos_weight=args.penalty,
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
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val_quick, y_val_quick)],
        verbose=100
    )
    print("[OK] XGBoost selesai dilatih.")

    print("      -> Menghitung probabilitas inferensi XGBoost pada data 2025...")
    prob_xgb = chunked_predict(model_xgb, X_val, is_xgb=True)
    roc_xgb = roc_auc_score(y_val, prob_xgb)
    pr_xgb = average_precision_score(y_val, prob_xgb)

    if args.only_xgb:
        prob_ensemble = prob_xgb
        roc_ens = roc_xgb
        pr_ens = pr_xgb
        print("\n" + "=" * 90)
        print(f"  HASIL XGBOOST STANDALONE : ROC-AUC = {roc_xgb:.4f} | PR-AUC = {pr_xgb:.4f}")
        print("=" * 90)
    else:
        # =========================================================================
        # 3. Pelatihan Model 2: LightGBM (Leaf-Wise Gradient Boosting)
        # =========================================================================
        print("\n" + "-" * 85)
        print(f"[2/3] Melatih Model 2: LightGBM (Penalti scale_pos_weight = {args.penalty:.1f})...")
        print("-" * 85)

        model_lgb = lgb.LGBMClassifier(
            n_estimators=args.lgb_estimators,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            scale_pos_weight=args.penalty,
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
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val_quick, y_val_quick)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )
        print("[OK] LightGBM selesai dilatih.")

        print("      -> Menghitung probabilitas inferensi LightGBM...")
        prob_lgb = chunked_predict(model_lgb, X_val, is_xgb=False)
        roc_lgb = roc_auc_score(y_val, prob_lgb)
        pr_lgb = average_precision_score(y_val, prob_lgb)

        # =========================================================================
        # 4. Pelatihan Model 3: Random Forest (Bagging Consensus Validator)
        # =========================================================================
        print("\n" + "-" * 85)
        print(f"[3/3] Melatih Model 3: Random Forest ({args.rf_estimators} pohon keputusan - Hakim Verifikasi)...")
        print("-" * 85)

        model_rf = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model_rf.fit(X_train, y_train)
        print("[OK] Random Forest selesai dilatih.")

        print("      -> Menghitung probabilitas inferensi Random Forest...")
        prob_rf = chunked_predict(model_rf, X_val, is_xgb=False)
        roc_rf = roc_auc_score(y_val, prob_rf)
        pr_rf = average_precision_score(y_val, prob_rf)

        # =========================================================================
        # 5. Penggabungan Model: Sinergi Cerdas vs Blend
        # =========================================================================
        if args.mode == "smart":
            print("\n[SINERGI CERDAS AKTIF]:")
            print("  1. Radar Penjaga Api : Menggabungkan XGBoost + LightGBM secara maksimal agar api tidak lolos.")
            print("  2. Hakim Verifikasi  : Random Forest memotong alarm palsu yang tidak didukung kondisi fisik lahan.")

            # Dual Booster: Ambil kewaspadaan tertinggi antara XGBoost dan LightGBM
            prob_boosters = np.maximum(prob_xgb, prob_lgb)
            prob_ensemble = prob_boosters.copy()

            # Hakim Verifikasi RF:
            # Jika konsensus pohon RF < 0.22 (sangat yakin aman) dan booster belum super yakin (< 0.72), pangkas skor
            mask_veto_rf = (prob_rf < 0.22) & (prob_boosters < 0.72)
            prob_ensemble[mask_veto_rf] *= (prob_rf[mask_veto_rf] / 0.22)
            
            # Jika RF juga mendeteksi bahaya (prob_rf >= 0.40), beri dorongan konfirmasi positif
            mask_boost_rf = (prob_rf >= 0.40)
            prob_ensemble[mask_boost_rf] = np.clip(prob_ensemble[mask_boost_rf] * 1.08, 0.0, 1.0)
        else:
            # Mode Rata-Rata Linier Biasa
            prob_ensemble = (0.45 * prob_xgb) + (0.35 * prob_lgb) + (0.20 * prob_rf)

        roc_ens = roc_auc_score(y_val, prob_ensemble)
        pr_ens = average_precision_score(y_val, prob_ensemble)

        print("\n" + "=" * 90)
        print("                  PERBANDINGAN METRIK GLOBAL DATA UJI 2025")
        print("=" * 90)
        print(f"  1. XGBoost Standalone       : ROC-AUC = {roc_xgb:.4f} | PR-AUC = {pr_xgb:.4f}")
        print(f"  2. LightGBM Standalone      : ROC-AUC = {roc_lgb:.4f} | PR-AUC = {pr_lgb:.4f}")
        print(f"  3. Random Forest Standalone  : ROC-AUC = {roc_rf:.4f} | PR-AUC = {pr_rf:.4f}")
        print(f"  4. SINERGI CERDAS (GABUNGAN): ROC-AUC = {roc_ens:.4f} | PR-AUC = {pr_ens:.4f}")
        print("=" * 90)

    # =========================================================================
    # 6. TABEL 1: Respons Operasional Sebelum Filter Tetangga
    # =========================================================================
    thresholds_to_test = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    summary_eval = {}

    print("\n" + "=" * 105)
    print("  TABEL 1: RESPONS OPERASIONAL MODEL (SEBELUM FILTER TETANGGA & AMBANG STATIS)")
    print("=" * 105)
    header = f"{'Ambang Skor':<14} | {'Recall Api':<12} | {'Presisi':<10} | {'Api Tertangkap':<16} | {'Api Kecolongan':<16} | {'Alarm Palsu Total':<18} | {'Alarm / Hari':<12}"
    print(header)
    print("-" * 105)

    for th in thresholds_to_test:
        pred_bin = (prob_ensemble >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred_bin).ravel()
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        fp_daily = fp / total_days_val
        score_int = int(round(th * 100))

        label = f"Skor >= {score_int}%"
        print(f"{label:<14} | {rec:>10.2f} % | {prec:>8.2f} % | {tp:>7,} titik      | {fn:>7,} titik      | {fp:>12,} kotak      | {fp_daily:>8.1f} /hari")

        summary_eval[score_int] = {
            "threshold": float(th),
            "recall": round(float(rec), 2),
            "precision": round(float(prec), 2),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "fp_daily": round(float(fp_daily), 1)
        }
    print("=" * 105)

    # =========================================================================
    # 7. TABEL 2: Respons Operasional dengan Filter Tetangga Spasial
    # =========================================================================
    print("\n" + "=" * 105)
    print("  TABEL 2: RESPONS OPERASIONAL + FILTER TETANGGA SPASIAL AKTIF")
    print("=" * 105)
    print(header)
    print("-" * 105)

    from scipy.spatial import cKDTree
    grid_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    gdf_coords = pd.read_parquet(grid_path, columns=["grid_id", "lat", "lon"])
    coords = np.column_stack([gdf_coords["lat"].values, gdf_coords["lon"].values])
    tree = cKDTree(coords)
    neighbor_map = tree.query_ball_tree(tree, r=0.08)  # radius ~8.8 km
    gid_to_idx = {gid: idx for idx, gid in enumerate(gdf_coords["grid_id"])}

    val_grid_ids = df_val["grid_id"].values
    val_dates = df_val["date"].values

    summary_filtered = {}
    filtered_preds_dict = {}

    for th in thresholds_to_test:
        pred_bin = (prob_ensemble >= th).astype(int)
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
        print(f"{label:<14} | {rec_f:>10.2f} % | {prec_f:>8.2f} % | {tp_f:>7,} titik      | {fn_f:>7,} titik      | {fp_f:>12,} kotak      | {fp_daily_f:>8.1f} /hari")

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
    print("=" * 105)

    # =========================================================================
    # 8. TABEL 3: Solusi 1 - Ambang Batas Musiman Dinamis (Kemarau vs Hujan)
    # =========================================================================
    print("\n" + "=" * 105)
    print("  TABEL 3: EVALUASI AMBANG BATAS MUSIMAN DINAMIS (KEMARAU VS HUJAN)")
    print("=" * 105)
    val_dt = pd.to_datetime(val_dates)
    val_months = val_dt.month
    mask_kemarau = np.isin(val_months, [7, 8, 9, 10])
    mask_hujan = ~mask_kemarau

    days_kemarau = len(np.unique(val_dates[mask_kemarau]))
    days_hujan = len(np.unique(val_dates[mask_hujan]))
    fires_kemarau = int(y_val[mask_kemarau].sum())
    fires_hujan = int(y_val[mask_hujan].sum())

    # Kemarau pakai Skor 60%, Hujan pakai Skor 75%
    pred_filtered_60 = filtered_preds_dict[60]
    pred_filtered_75 = filtered_preds_dict[75]

    pred_dynamic = np.zeros(len(y_val), dtype=int)
    pred_dynamic[mask_kemarau] = pred_filtered_60[mask_kemarau]
    pred_dynamic[mask_hujan] = pred_filtered_75[mask_hujan]

    tn_dyn, fp_dyn, fn_dyn, tp_dyn = confusion_matrix(y_val, pred_dynamic).ravel()
    rec_dyn = (tp_dyn / (tp_dyn + fn_dyn)) * 100.0
    prec_dyn = (tp_dyn / (tp_dyn + fp_dyn)) * 100.0

    tp_k = int(((pred_dynamic[mask_kemarau] == 1) & (y_val[mask_kemarau] == 1)).sum())
    fn_k = int(((pred_dynamic[mask_kemarau] == 0) & (y_val[mask_kemarau] == 1)).sum())
    fp_k = int(((pred_dynamic[mask_kemarau] == 1) & (y_val[mask_kemarau] == 0)).sum())
    rec_k = (tp_k / (tp_k + fn_k)) * 100.0 if (tp_k + fn_k) > 0 else 0.0

    tp_h = int(((pred_dynamic[mask_hujan] == 1) & (y_val[mask_hujan] == 1)).sum())
    fn_h = int(((pred_dynamic[mask_hujan] == 0) & (y_val[mask_hujan] == 1)).sum())
    fp_h = int(((pred_dynamic[mask_hujan] == 1) & (y_val[mask_hujan] == 0)).sum())
    rec_h = (tp_h / (tp_h + fn_h)) * 100.0 if (tp_h + fn_h) > 0 else 0.0

    fp_static_60 = summary_filtered[60]["fp"]
    fp_saved = fp_static_60 - fp_dyn

    print(f"  1. Musim Kemarau (Juli s/d Oktober - {days_kemarau} Hari, Ambang Skor 60%):")
    print(f"     - Api Tertangkap      : {tp_k:,} dari {fires_kemarau:,} titik api ({rec_k:.2f}% Recall)")
    print(f"     - Alarm Palsu Kemarau : {fp_k:,} kotak ({fp_k / days_kemarau:.1f} kotak/hari)")
    print(f"\n  2. Musim Hujan/Transisi (November s/d Juni - {days_hujan} Hari, Ambang Skor 75%):")
    print(f"     - Api Tertangkap      : {tp_h:,} dari {fires_hujan:,} titik api ({rec_h:.2f}% Recall)")
    print(f"     - Alarm Palsu Hujan   : {fp_h:,} kotak ({fp_h / days_hujan:.1f} kotak/hari)")
    print(f"\n  3. TOTAL OPERASIONAL SETAHUN PENUH DENGAN AMBANG DINAMIS:")
    print(f"     - Total Api Tertangkap     : {tp_dyn:,} dari {total_fires_val:,} titik ({rec_dyn:.2f}% Recall)")
    print(f"     - Total Alarm Palsu        : {fp_dyn:,} kotak (Rata-rata {fp_dyn / total_days_val:.1f} kotak/hari)")
    print(f"     - Presisi Operasional      : {prec_dyn:.2f}%")
    print(f"     - ALARM PALSU TERPANGKAS   : {fp_saved:,} kotak berhasil dieliminasi berkat ambang musiman dinamis!")
    print("=" * 105)

    # =========================================================================
    # 9. Penyimpanan Model Permanen (--save)
    # =========================================================================
    if args.save:
        print("\n" + "=" * 85)
        print("[PENYIMPANAN] Menyimpan model ke folder data/models...")
        print("=" * 85)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        xgb_file = MODELS_DIR / "fire_risk_xgboost_model.json"
        lgb_file = MODELS_DIR / "fire_risk_lgb_model.txt"
        rf_file = MODELS_DIR / "fire_risk_rf_model.joblib"
        meta_file = MODELS_DIR / "model_metadata.json"
        rep_file = MODELS_DIR / "model_evaluation_report.json"

        if args.only_xgb:
            model_xgb.save_model(str(xgb_file))
            print(f"  [1/1] Model XGBoost murni tersimpan di : {xgb_file.name}")
            ens_weights = {"xgboost": 1.0}
            algo_name = "XGBoost Standalone (V2.2 Enriched)"
            ens_mode = "standalone"
        else:
            model_xgb.save_model(str(xgb_file))
            print(f"  [1/3] Model XGBoost tersimpan di       : {xgb_file.name}")
            model_lgb.booster_.save_model(str(lgb_file))
            print(f"  [2/3] Model LightGBM tersimpan di      : {lgb_file.name}")
            joblib.dump(model_rf, rf_file, compress=3)
            print(f"  [3/3] Model Random Forest tersimpan di : {rf_file.name}")
            ens_weights = {"xgboost": 0.45, "lightgbm": 0.35, "random_forest": 0.20}
            algo_name = f"Smart Ensemble ({args.mode.upper()}: XGBoost + LightGBM + Random Forest)"
            ens_mode = args.mode

        metadata = {
            "algorithm": algo_name,
            "version": "2.2-smart-ensemble",
            "ensemble_mode": ens_mode,
            "features": available_features,
            "feature_count": len(available_features),
            "ensemble_weights": ens_weights,
            "scale_pos_weight": float(args.penalty),
            "downsample_ratio": int(args.downsample_ratio),
            "threshold_sedang": 60.0,
            "threshold_tinggi": 85.0,
            "seasonal_dynamic_thresholds": {
                "dry_season_months": [7, 8, 9, 10],
                "dry_threshold_sedang": 60.0,
                "dry_threshold_tinggi": 85.0,
                "wet_threshold_sedang": 75.0,
                "wet_threshold_tinggi": 90.0
            },
            "metrics_global": {
                "roc_auc_ensemble": round(float(roc_ens), 4),
                "pr_auc_ensemble": round(float(pr_ens), 4),
                "roc_auc_xgb": round(float(roc_xgb), 4),
                "pr_auc_xgb": round(float(pr_xgb), 4)
            },
            "confusion_matrix_dynamic": {
                "true_positive": int(tp_dyn),
                "false_negative": int(fn_dyn),
                "false_positive": int(fp_dyn),
                "true_negative": int(tn_dyn),
                "fire_recall_pct": round(float(rec_dyn), 2),
                "precision_pct": round(float(prec_dyn), 2)
            },
            "threshold_sensitivity_static": summary_eval,
            "threshold_sensitivity_filtered": summary_filtered
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  [OK] Metadata sistem diperbarui di     : {meta_file.name}")
        print(f"  [OK] Laporan evaluasi disimpan di      : {rep_file.name}")
        print("\n[SUKSES] Model dan metadata berhasil diperbarui!")
    else:
        print("\n[INFO] Mode simulasi selesai tanpa penyimpanan.")
        print("       Untuk melatih dan MENYIMPAN model ini ke sistem, jalankan:")
        print(f"       python src/ml/train_ensemble.py --penalty {args.penalty:.1f} --save\n")


if __name__ == "__main__":
    args = parse_args()
    train_ensemble_pipeline(args)
