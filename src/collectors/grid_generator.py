"""
Grid Generator untuk Daratan Kalimantan (Resolusi ~5km)
Menghasilkan poligon grid kotak, memotong hanya daratan Indonesia (membuang laut & Malaysia),
dan menghitung titik tengah (lat/lon) untuk setiap grid_id.
"""

import sys
from pathlib import Path
import json
import requests
import numpy as np

# Menambahkan parent dir ke sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    KALIMANTAN_BBOX,
    GRID_RESOLUTION_DEG,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

def download_kalimantan_boundary() -> Path:
    """
    Mengunduh GeoJSON batas wilayah Indonesia tingkat provinsi (ADM1) dari geoBoundaries,
    kemudian memfilter 5 provinsi Kalimantan (Barat, Tengah, Selatan, Timur, Utara).
    """
    output_path = RAW_DATA_DIR / "kalimantan_boundary.geojson"
    if output_path.exists():
        print(f"[OK] Batas Kalimantan sudah ada di: {output_path}")
        return output_path

    print("[INFO] Mengunduh batas wilayah administratif Kalimantan...")
    url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/IDN/ADM1/geoBoundaries-IDN-ADM1_simplified.geojson"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        geojson_data = resp.json()

        filtered_features = []
        for feat in geojson_data.get("features", []):
            shape_name = feat.get("properties", {}).get("shapeName", "").lower()
            if "kalimantan" in shape_name:
                filtered_features.append(feat)

        if not filtered_features:
            print("[WARN] Tidak menemukan provinsi dengan nama shapeName standar, menyimpan semua fitur...")
            filtered_features = geojson_data.get("features", [])

        kalimantan_geojson = {
            "type": "FeatureCollection",
            "features": filtered_features,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(kalimantan_geojson, f)

        print(f"[OK] Batas Kalimantan berhasil disimpan ke: {output_path}")
        return output_path

    except Exception as e:
        print(f"[ERROR] Gagal mengunduh batas wilayah: {e}")
        print("[INFO] Menggunakan bounding box kasar daratan Kalimantan...")
        return None


def generate_kalimantan_grid(boundary_path: Path = None):
    """
    Membuat grid kotak 5 km x 5 km di seluruh daratan Kalimantan.
    """
    import shapely
    from shapely.geometry import box, Polygon, MultiPolygon
    import geopandas as gpd

    print(f"[INFO] Membuat grid beresolusi ~{GRID_RESOLUTION_DEG * 111:.1f} km...")

    min_lon = KALIMANTAN_BBOX["min_lon"]
    max_lon = KALIMANTAN_BBOX["max_lon"]
    min_lat = KALIMANTAN_BBOX["min_lat"]
    max_lat = KALIMANTAN_BBOX["max_lat"]
    step = GRID_RESOLUTION_DEG

    lons = np.arange(min_lon, max_lon, step)
    lats = np.arange(min_lat, max_lat, step)

    grid_boxes = []
    grid_meta = []
    idx = 1

    for lon in lons:
        for lat in lats:
            cell = box(lon, lat, lon + step, lat + step)
            grid_boxes.append(cell)
            grid_meta.append({
                "grid_id": f"KAL_{idx:05d}",
                "lon": round(lon + (step / 2.0), 4),
                "lat": round(lat + (step / 2.0), 4),
                "min_lon": round(lon, 4),
                "min_lat": round(lat, 4),
                "max_lon": round(lon + step, 4),
                "max_lat": round(lat + step, 4),
            })
            idx += 1

    gdf_grid = gpd.GeoDataFrame(grid_meta, geometry=grid_boxes, crs="EPSG:4324")
    print(f"[INFO] Total grid awal sebelum dipotong: {len(gdf_grid):,} kotak.")

    # Potong dengan polygon daratan Kalimantan jika file boundary tersedia
    if boundary_path and boundary_path.exists():
        print("[INFO] Memotong grid dengan batas daratan resmi Kalimantan...")
        try:
            gdf_boundary = gpd.read_file(boundary_path)
            # Pastikan CRS cocok
            gdf_boundary = gdf_boundary.to_crs("EPSG:4324")
            
            # Gabungkan poligon provinsi jadi 1 kesatuan daratan
            kalimantan_union = gdf_boundary.unary_union
            
            # Saring grid yang menyentuh/berada di dalam daratan
            gdf_grid = gdf_grid[gdf_grid.intersects(kalimantan_union)].copy()
            # Reset ID agar urut
            gdf_grid["grid_id"] = [f"KAL_{i+1:05d}" for i in range(len(gdf_grid))]
            print(f"[OK] Total grid daratan setelah dipotong: {len(gdf_grid):,} kotak.")
        except Exception as e:
            print(f"[WARN] Pemotongan gagal ({e}), menggunakan grid bounding box.")

    # Simpan dalam format GeoJSON dan Parquet
    geojson_out = PROCESSED_DATA_DIR / "kalimantan_grid_5km.geojson"
    parquet_out = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"

    gdf_grid.to_file(geojson_out, driver="GeoJSON")
    gdf_grid.to_parquet(parquet_out)

    print(f"[BERHASIL] File grid disimpan di:")
    print(f"  - GeoJSON: {geojson_out}")
    print(f"  - Parquet: {parquet_out}")
    return gdf_grid


if __name__ == "__main__":
    boundary_file = download_kalimantan_boundary()
    generate_kalimantan_grid(boundary_file)
