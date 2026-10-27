"""
Inference Engine untuk Sistem Early Warning Kebakaran Hutan Kalimantan.
Mendukung Model XGBoost:
1. Skor Risiko Kontinu (0.0 sampai 100.0) per Grid.
2. Status Peringatan Dini: RENDAH (Hijau), SEDANG (Kuning), TINGGI (Merah).
3. Export GeoJSON / GeoDataFrame siap tampil di Peta Web.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

class FireRiskPredictor:
    def __init__(self, model_path: Path = None, meta_path: Path = None):
        self.meta_path = meta_path or (MODELS_DIR / "model_metadata.json")

        if not self.meta_path.exists():
            raise FileNotFoundError("Metadata model belum ditemukan! Jalankan pelatihan terlebih dahulu.")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.th_sedang = self.metadata.get("threshold_sedang", 60.0)
        self.th_tinggi = self.metadata.get("threshold_tinggi", 85.0)

        # 1. Muat Model XGBoost
        self.xgb_model = None
        xgb_path = MODELS_DIR / "fire_risk_xgboost_model.json"
        if xgb_path.exists():
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                print("[OK] Model XGBoost aktif.")
            except Exception as e:
                print(f"[WARN] Gagal memuat XGBoost: {e}")

        # 2. Muat Model LightGBM
        self.lgb_model = None
        lgb_path = MODELS_DIR / "fire_risk_lgb_model.txt"
        if lgb_path.exists():
            try:
                import lightgbm as lgb
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                print("[OK] Model LightGBM aktif.")
            except Exception as e:
                print(f"[WARN] Gagal memuat LightGBM: {e}")

        # 3. Muat Model Random Forest
        self.rf_model = None
        rf_path = MODELS_DIR / "fire_risk_rf_model.joblib"
        if rf_path.exists():
            try:
                import joblib
                self.rf_model = joblib.load(rf_path)
                print("[OK] Model Random Forest aktif.")
            except Exception as e:
                print(f"[WARN] Gagal memuat Random Forest: {e}")

        # 4. Muat Model Verifier (Two-Stage Cascade)
        self.verifier_model = None
        verifier_path = MODELS_DIR / "fire_risk_verifier_model.txt"
        if verifier_path.exists():
            try:
                import lightgbm as lgb
                self.verifier_model = lgb.Booster(model_file=str(verifier_path))
                print("[OK] Model Penjaga (Stage 2 Verifier) aktif.")
            except Exception as e:
                print(f"[WARN] Gagal memuat model verifier: {e}")

        # Konfigurasi bobot ensemble
        weights_conf = self.metadata.get("ensemble_weights", {})
        active_models = []
        if self.xgb_model is not None: active_models.append("xgboost")
        if self.lgb_model is not None: active_models.append("lightgbm")
        if self.rf_model is not None: active_models.append("random_forest")

        if not active_models:
            raise FileNotFoundError("Tidak ada model aktif yang dapat dimuat dari data/models!")

        # Normalisasi bobot sesuai model yang aktif
        total_w = sum(weights_conf.get(m, 1.0) for m in active_models)
        self.weights = {m: weights_conf.get(m, 1.0) / total_w for m in active_models}
        print(f"[OK] Mode Prediksi Aktif: {self.metadata.get('algorithm', 'Ensemble')}")
        print(f"     Ambang Batas Standar: Sedang={self.th_sedang}, Tinggi={self.th_tinggi}")

    def predict_risk(self, df_features: pd.DataFrame, filter_isolated: bool = True, dynamic_season: bool = True) -> pd.DataFrame:
        """
        Menghitung Skor Risiko Kontinu dan Kategori Peringatan untuk setiap baris grid.
        Mendukung Two-Stage Cascade, Trio Ensemble, dan Ambang Batas Musiman Dinamis.
        """
        df_in = df_features.copy()

        # Pastikan fitur jarak jalan dan interaksi terisi jika belum ada
        if "distance_to_road_km" not in df_in.columns:
            df_in["distance_to_road_km"] = 5.0
        # Pastikan 4 fitur termodinamika terhitung jika ada di daftar fitur model
        if "compound_drought_index" not in df_in.columns and all(k in df_in.columns for k in ["temp_max", "vpd_max", "dry_days_count", "soil_moisture_0_7cm"]):
            sm = df_in["soil_moisture_0_7cm"].clip(lower=0.05)
            df_in["compound_drought_index"] = (
                (df_in["temp_max"] / 32.0).clip(lower=0.5) *
                (df_in["vpd_max"] / 1.5).clip(lower=0.5) *
                np.log1p(df_in["dry_days_count"]) / (sm + 0.1)
            ).astype(np.float32)

        if "soil_drying_gradient" not in df_in.columns and all(k in df_in.columns for k in ["soil_moisture_7_28cm", "soil_moisture_0_7cm"]):
            df_in["soil_drying_gradient"] = (df_in["soil_moisture_7_28cm"] - df_in["soil_moisture_0_7cm"]).clip(lower=0.0).astype(np.float32)

        if "fire_cluster_momentum" not in df_in.columns:
            h3d = df_in["hotspot_count_3d"] if "hotspot_count_3d" in df_in.columns else 0.0
            nb3d = df_in["neighbor_fire_count_3d"] if "neighbor_fire_count_3d" in df_in.columns else 0.0
            df_in["fire_cluster_momentum"] = (h3d * 2.0 + nb3d).astype(np.float32)

        if "peat_road_hazard" not in df_in.columns:
            peat = (df_in["peat_coverage_pct"] / 100.0) if "peat_coverage_pct" in df_in.columns else 0.0
            dist = df_in["distance_to_road_km"] if "distance_to_road_km" in df_in.columns else 5.0
            dd = df_in["dry_days_count"] if "dry_days_count" in df_in.columns else 0.0
            df_in["peat_road_hazard"] = (peat * np.exp(-dist / 6.0) * np.log1p(dd)).astype(np.float32)

        available_cols = [c for c in self.features if c in df_in.columns]
        X = df_in[available_cols].fillna(0).astype(np.float32)

        algo_name = self.metadata.get("algorithm", "")

        # Mode A: Two-Stage Cascade (Detektor + Penjaga Alarm Palsu)
        if "Two-Stage" in algo_name and self.xgb_model is not None:
            dmat = xgb.DMatrix(X)
            raw_prob = self.xgb_model.predict(dmat)
            if self.verifier_model is not None:
                p_s2 = self.verifier_model.predict(X)
                # VIP Fast-Track: skor >= 0.70 lolos langsung, < 0.70 diverifikasi
                mask_doubt = (raw_prob >= 0.45) & (raw_prob < 0.70)
                mask_veto = mask_doubt & (p_s2 < 0.28)
                raw_prob[mask_veto] = raw_prob[mask_veto] * (p_s2[mask_veto] / 0.28)

        # Mode B: Smart Synergy Ensemble
        elif self.metadata.get("ensemble_mode") == "smart" and self.xgb_model is not None and self.lgb_model is not None:
            dmat = xgb.DMatrix(X)
            p_xgb = self.xgb_model.predict(dmat)
            p_lgb = self.lgb_model.predict(X)
            prob_boosters = np.maximum(p_xgb, p_lgb)
            raw_prob = prob_boosters.copy()

            if self.rf_model is not None:
                p_rf = self.rf_model.predict_proba(X)[:, 1]
                mask_veto = (p_rf < 0.22) & (prob_boosters < 0.72)
                raw_prob[mask_veto] *= (p_rf[mask_veto] / 0.22)
                mask_boost = (p_rf >= 0.40)
                raw_prob[mask_boost] = np.clip(raw_prob[mask_boost] * 1.08, 0.0, 1.0)

        # Mode C: Blend atau Standalone Fallback
        else:
            raw_prob = np.zeros(len(df_in), dtype=np.float32)
            if "xgboost" in self.weights and self.xgb_model is not None:
                dmat = xgb.DMatrix(X)
                raw_prob += (self.weights["xgboost"] * self.xgb_model.predict(dmat))
            if "lightgbm" in self.weights and self.lgb_model is not None:
                raw_prob += (self.weights["lightgbm"] * self.lgb_model.predict(X))
            if "random_forest" in self.weights and self.rf_model is not None:
                raw_prob += (self.weights["random_forest"] * self.rf_model.predict_proba(X)[:, 1])

        df_out = df_in.copy()
        # Konversi probabilitas ke skala Skor Risiko 0.0 - 100.0
        df_out["risk_score"] = (raw_prob * 100.0).round(1)

        # Solusi 1: Ambang Batas Musiman Dinamis
        # Musim Kemarau (Juli s/d Oktober): Sensitif (Sedang >= 60, Tinggi >= 85)
        # Musim Hujan / Transisi (November s/d Juni): Ketat (Sedang >= 75, Tinggi >= 90)
        if dynamic_season and "date" in df_out.columns:
            dates = pd.to_datetime(df_out["date"])
            months = dates.dt.month.values
            is_dry = np.isin(months, [7, 8, 9, 10])

            # Ambang dinamis per baris
            th_sedang_arr = np.where(is_dry, 60.0, 75.0)
            th_tinggi_arr = np.where(is_dry, 85.0, 90.0)

            scores = df_out["risk_score"].values
            risk_levels = np.full(len(scores), "RENDAH", dtype=object)
            risk_levels[scores >= th_sedang_arr] = "SEDANG"
            risk_levels[scores >= th_tinggi_arr] = "TINGGI"
            df_out["risk_level"] = risk_levels
        else:
            def assign_status(score):
                if score >= self.th_tinggi:
                    return "TINGGI"
                elif score >= self.th_sedang:
                    return "SEDANG"
                else:
                    return "RENDAH"
            df_out["risk_level"] = df_out["risk_score"].apply(assign_status)

        # Solusi Filter Tetangga (Penyaring Alarm Palsu Sendirian)
        if filter_isolated and "lat" in df_out.columns and "lon" in df_out.columns:
            df_out = self.apply_spatial_neighbor_filter(df_out)

        return df_out

    def apply_spatial_neighbor_filter(self, df_predicted: pd.DataFrame, search_radius_deg: float = 0.08) -> pd.DataFrame:
        """
        [Filter Tetangga Spasial]
        Menyaring kotak terpencil yang berdiri sendirian tanpa ada tetangga aktif di sekelilingnya.
        Jika kotak berstatus SEDANG/TINGGI tetapi di sekelilingnya (8 kotak terdekat ~8.8 km) semuanya RENDAH,
        maka status kotak tersebut diturunkan menjadi RENDAH untuk memangkas alarm palsu.
        """
        try:
            from scipy.spatial import cKDTree
            coords = np.column_stack([df_predicted["lat"].values, df_predicted["lon"].values])
            tree = cKDTree(coords)
            neighbors = tree.query_ball_tree(tree, r=search_radius_deg)

            active_indices = set(df_predicted[df_predicted["risk_level"].isin(["SEDANG", "TINGGI"])].index)
            filtered_levels = list(df_predicted["risk_level"])
            cleaned_count = 0

            for idx in active_indices:
                adj = [n for n in neighbors[idx] if n != idx]
                has_active_neighbor = any(n in active_indices for n in adj)
                if not has_active_neighbor:
                    filtered_levels[idx] = "RENDAH"
                    cleaned_count += 1

            df_out = df_predicted.copy()
            df_out["risk_level"] = filtered_levels
            if cleaned_count > 0:
                print(f"[FILTER TETANGGA] Berhasil menyaring {cleaned_count:,} kotak alarm palsu terpencil (sendirian).")
            return df_out
        except Exception as e:
            print(f"[WARN] Filter spasial tetangga dilewati ({e}).")
            return df_predicted

    def generate_warning_geojson(self, df_predicted: pd.DataFrame, output_path: Path = None) -> Path:
        """
        Menggabungkan hasil prediksi dengan poligon grid 5km dan menyimpan file GeoJSON untuk Web GIS.
        """
        grid_parquet = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        gdf_grid = gpd.read_parquet(grid_parquet)

        # Merge hasil prediksi ke poligon grid
        merged = gdf_grid.merge(
            df_predicted[["grid_id", "risk_score", "risk_level"]],
            on="grid_id",
            how="inner"
        )

        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "fire_risk_early_warning_latest.geojson"

        merged.to_file(output_path, driver="GeoJSON")
        print(f"[OK] Peta risiko kebakaran berhasil disimpan di: {output_path}")
        return output_path
