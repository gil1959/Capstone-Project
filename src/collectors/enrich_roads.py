"""
Collector & Enriched Data Jalan Raya OpenStreetMap (OSM) Kalimantan.
Sistem Peringatan Dini Kebakaran Hutan Kalimantan (Borneo Siaga).

Fungsi:
1. Mengunduh data jalan raya pulau Kalimantan resmi dari Geofabrik OpenStreetMap.
2. Mengekstrak layer jalan (gis_osm_roads_free_1.shp) yang mencakup jalan nasional, provinsi,
   kabupaten, jalan pedesaan, serta jalan perkebunan sawit/HTI.
3. Menghitung jarak terdekat (dalam km) dari pusat masing-masing 22.276 kotak grid 5km ke jalan terdekat.
4. Menambahkan kolom baru 'distance_to_road_km' ke data/processed/kalimantan_grid_enriched.parquet.
5. Bersifat statis (cukup dijalankan 1 kali saja).
"""

import sys
import os
import zipfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

GEOFABRIK_URL = "https://download.geofabrik.de/asia/indonesia/kalimantan-latest-free.shp.zip"
OSM_DIR = RAW_DATA_DIR / "osm_roads"


def download_file_with_progress(url: str, dest_path: Path):
    """Mengunduh file dengan indikator persentase progres."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[INFO] File arsip {dest_path.name} sudah ada ({dest_path.stat().st_size / 1024 / 1024:.1f} MB). Melewati unduhan.")
        return

    print(f"[UNDUH] Mengunduh data jalan Kalimantan dari Geofabrik OpenStreetMap...")
    print(f"        URL: {url}")
    
    headers = {"User-Agent": "BorneoSiagaResearch/2.0 (student research)"}
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB per chunk
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"\r        Progress: {pct:.1f}% ({mb_down:.1f} / {mb_tot:.1f} MB)", end="", flush=True)

    print(f"\n[OK] Unduhan selesai: {dest_path.name}")


def extract_road_shapefile(zip_path: Path, extract_dir: Path) -> Path:
    """Hanya mengekstrak file gis_osm_roads_free_1.* dari zip untuk menghemat ruang disk."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    shp_file = extract_dir / "gis_osm_roads_free_1.shp"
    if shp_file.exists():
        print(f"[INFO] Shapefile jalan {shp_file.name} sudah diekstrak.")
        return shp_file

    print(f"[EKSTRAK] Mengekstrak layer jalan raya Kalimantan...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for item in z.namelist():
            if "gis_osm_roads_free_1" in item:
                z.extract(item, extract_dir)
                print(f"        -> Ekstrak: {item}")

    return shp_file


def compute_road_distances(gdf_grid: gpd.GeoDataFrame, shp_roads_path: Path) -> gpd.GeoDataFrame:
    """Menghitung jarak terdekat dari masing-masing kotak grid ke jaringan jalan OpenStreetMap."""
    print(f"[SPASIAL] Membaca data garis jalan raya dari {shp_roads_path.name}...")
    gdf_roads = gpd.read_file(shp_roads_path)
    print(f"[OK] Total ruas jalan terbaca: {len(gdf_roads):,} segmen.")

    # Filter tipe jalan penting (jalan utama, sekunder, pemukiman, hingga jalan perkebunan/track)
    road_classes = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "track", "service"
    ]
    if "fclass" in gdf_roads.columns:
        gdf_roads = gdf_roads[gdf_roads["fclass"].isin(road_classes)].copy()
        print(f"     Ruas jalan terpilih (arteri s/d kebun/track): {len(gdf_roads):,} segmen.")

    # Ambil sampel koordinat dari garis jalan untuk pencarian cepat via KDTree
    print("[SPASIAL] Mengekstrak titik-titik koordinat jalan untuk KDTree...")
    road_pts = []
    for geom in gdf_roads.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            # Sampel setiap 3 titik agar tidak terlalu berat namun tetap sangat presisi
            road_pts.extend(coords[::2])
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                road_pts.extend(coords[::2])

    road_pts_arr = np.array(road_pts, dtype=np.float32)  # format: [lon, lat]
    print(f"[OK] Terkumpul {len(road_pts_arr):,} titik koordinat jaringan jalan.")

    # Bangun KDTree spasial
    print("[SPASIAL] Membangun indeks pencarian cepat KDTree...")
    # Urutan: [lat, lon]
    road_lat_lon = np.column_stack([road_pts_arr[:, 1], road_pts_arr[:, 0]])
    tree = cKDTree(road_lat_lon)

    # Titik tengah 22.276 kotak grid
    grid_lat_lon = np.column_stack([gdf_grid["lat"].values, gdf_grid["lon"].values])

    print("[SPASIAL] Menghitung jarak terdekat ke jalan untuk 22.276 kotak...")
    # Hitung jarak derajat, lalu konversi ke kilometer (1 derajat ~ 111.32 km di ekuator)
    dist_deg, _ = tree.query(grid_lat_lon, k=1)
    dist_km = dist_deg * 111.32

    gdf_grid["distance_to_road_km"] = np.round(dist_km, 2)

    avg_dist = gdf_grid["distance_to_road_km"].mean()
    min_dist = gdf_grid["distance_to_road_km"].min()
    max_dist = gdf_grid["distance_to_road_km"].max()
    print("\n" + "=" * 65)
    print("   RINGKASAN JARAK JALAN RAYA (DISTANCE TO ROAD) KALIMANTAN")
    print("=" * 65)
    print(f"  - Rata-rata jarak ke jalan : {avg_dist:.2f} km")
    print(f"  - Jarak terdekat           : {min_dist:.2f} km (kotak tepat di atas jalan)")
    print(f"  - Jarak terjauh (pedalaman): {max_dist:.2f} km")
    print("=" * 65)

    return gdf_grid


def run_enrichment():
    print("\n" + "=" * 75)
    print("   [BORNEO SIAGA] PENGAYAAN DATA JARAK JALAN RAYA (OPENSTREETMAP)")
    print("=" * 75)

    # 1. Pastikan grid master ada
    grid_enriched_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    if not grid_enriched_path.exists():
        grid_enriched_path = PROCESSED_DATA_DIR / "kalimantan_grid_5km.parquet"
        if not grid_enriched_path.exists():
            raise FileNotFoundError("Master grid Kalimantan belum ada!")

    gdf_grid = gpd.read_parquet(grid_enriched_path)
    print(f"[OK] Master grid dimuat ({len(gdf_grid):,} kotak).")

    # 2. Unduh data Geofabrik OSM
    zip_path = OSM_DIR / "kalimantan-latest-free.shp.zip"
    download_file_with_progress(GEOFABRIK_URL, zip_path)

    # 3. Ekstrak shapefile jalan
    shp_roads = extract_road_shapefile(zip_path, OSM_DIR)

    # 4. Hitung jarak
    gdf_updated = compute_road_distances(gdf_grid, shp_roads)

    # 5. Simpan pembaruan ke kalimantan_grid_enriched.parquet
    out_path = PROCESSED_DATA_DIR / "kalimantan_grid_enriched.parquet"
    gdf_updated.to_parquet(out_path)
    print(f"\n[SUKSES] Data grid telah diperbarui dengan kolom 'distance_to_road_km':")
    print(f"         Tersimpan di: {out_path}\n")


if __name__ == "__main__":
    run_enrichment()
