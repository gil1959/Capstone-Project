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
        if model_path is None:
            xgb_path = MODELS_DIR / "fire_risk_xgboost_model.json"
            if xgb_path.exists():
                self.model_path = xgb_path
            else:
                self.model_path = MODELS_DIR / "fire_risk_lgb_model.txt"
        else:
            self.model_path = Path(model_path)

        self.meta_path = meta_path or (MODELS_DIR / "model_metadata.json")

        if not self.model_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Model belum dilatih! Jalankan train.py terlebih dahulu.")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.th_sedang = self.metadata.get("threshold_sedang", 15.0)
        self.th_tinggi = self.metadata.get("threshold_tinggi", 35.0)

        # Muat model
        if str(self.model_path).endswith(".json"):
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            self.model_type = "xgboost"
        else:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.model_path))
            self.model_type = "lightgbm"

        print(f"[OK] Model ({self.model_type.upper()}) berhasil dimuat.")
        print(f"     Ambang Batas: Sedang={self.th_sedang}, Tinggi={self.th_tinggi}")

    def predict_risk(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Menghitung Skor Risiko Kontinu dan Kategori Peringatan untuk setiap baris grid.
        """
        X = df_features[self.features].fillna(0).astype(np.float32)

        if self.model_type == "xgboost":
            dmat = xgb.DMatrix(X)
            raw_prob = self.model.predict(dmat)
        else:
            raw_prob = self.model.predict(X)

        df_out = df_features.copy()
        # Konversi probabilitas ke skala Skor Risiko 0.0 - 100.0
        df_out["risk_score"] = (raw_prob * 100.0).round(1)

        def assign_status(score):
            if score >= self.th_tinggi:
                return "TINGGI"
            elif score >= self.th_sedang:
                return "SEDANG"
            else:
                return "RENDAH"

        df_out["risk_level"] = df_out["risk_score"].apply(assign_status)
        return df_out

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
