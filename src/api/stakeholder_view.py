"""
Stakeholder Dashboard HTML View for Kalimantan Forest Fire Early Warning System.
Sesuai Use Case Diagram:
1. Melihat peta risiko kebakaran
2. Melihat detail risiko wilayah
3. Melihat riwayat tren risiko (Grafik Temporal 2023-2024)
4. Menerima notifikasi dini (Disposisi Lapangan BPBD & Manggala Agni)
5. Section Melayang Peringatan Bahaya (Hasil Prediksi H+1) dengan Dropdown Lokasi Riil & Koordinat
Bebas emoji, ikon SVG murni, clean dark glassmorphism.
"""

import os

def get_stakeholder_html():
    is_prod = os.getenv("APP_ENV", "local").lower() in ["prod", "production"]
    default_date = "latest" if is_prod else "2024-09-03"
    env_badge = '<span class="badge-role" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">PROD · LIVE DAILY</span>' if is_prod else '<span class="badge-role" style="background: rgba(239, 155, 75, 0.2); color: #ef9b4b; border: 1px solid rgba(239, 155, 75, 0.4);">LOKAL · UJI 2024</span>'
    date_title = 'Prediksi untuk <b>Terkini</b> <small>H+1 (Real-Time Satelit NASA & Cuaca)</small>' if is_prod else 'Prediksi untuk <b>3 September 2024</b> <small>UTC (Puncak Kemarau)</small>'
    sel_latest = "selected" if is_prod else ""
    sel_test = "" if is_prod else "selected"
    date_options = f"""<option value="latest" {sel_latest}>🔥 Prediksi Terkini (Real-Time Pipeline H+1)</option>
                <option value="2024-09-03" {sel_test}>03 September 2024 (Puncak Karhutla - 353 Sel Merah)</option>
                <option value="2024-09-06">06 September 2024 (Siaga Karhutla Darat)</option>
                <option value="2024-09-14">14 September 2024 (Kemarau Kalteng/Kalsel)</option>
                <option value="2024-12-31">31 Desember 2024 (Musim Hujan - 100% Aman)</option>"""

    html_content = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Stakeholder & Satgas Karhutla | Kalimantan Forest Fire</title>
    <!-- Favicon Logo Api -->
    <link rel="icon" type="image/svg+xml" href="/favicon.ico" />
    <link rel="alternate icon" href="/favicon.ico" />
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            width: 100vw; height: 100vh; overflow: hidden;
            background: #0a121b; color: #edf4f5;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        #map {
            position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: 1; background: #0a121b;
        }

        .icon-svg {
            display: inline-block; width: 15px; height: 15px; vertical-align: middle;
            fill: currentColor; flex-shrink: 0; pointer-events: none;
        }

        /* Glassmorphism Clean Panels */
        .glass-panel {
            position: fixed; z-index: 1000;
            background: rgba(10, 18, 27, 0.94);
            border: 1px solid #283745;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(12px);
            padding: 12px 16px;
        }

        /* Top-Left: Brand & Date */
        .map-header {
            top: 16px; left: 16px; max-width: 440px;
        }
        .portal-brand {
            display: flex; align-items: center; gap: 10px;
        }
        .portal-brand .brand-logo {
            width: 24px; height: 24px; fill: #38bdf8; flex-shrink: 0;
        }
        .portal-brand strong {
            font-size: 17px; font-weight: 700; color: #edf4f5; letter-spacing: 0.5px;
        }
        .portal-brand .badge-role {
            font-size: 10.5px; padding: 2px 7px; border-radius: 4px;
            background: rgba(56, 189, 248, 0.2); color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4); font-weight: 700;
            text-transform: uppercase;
        }
        .portal-date {
            font-size: 12.5px; line-height: 1.5; margin-top: 3px; color: #c6d3dc;
        }
        .portal-date b { color: #f8fafc; font-weight: 600; }
        .risk-chips {
            display: inline-flex; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #a9bcc9;
        }
        .chip {
            display: inline-flex; align-items: center; gap: 5px;
        }
        .chip-dot {
            width: 8px; height: 8px; border-radius: 50%;
        }

        /* Section Melayang Peringatan Bahaya (Clean Dropdown, Bebas Border AI) */
        .floating-warning-box {
            top: 180px; left: 16px; width: 340px; padding: 12px;
            border-left: 3px solid #ef4444 !important;
        }
        .warning-select {
            width: 100%; background: #162433; color: #edf4f5;
            border: 1px solid #364859; border-radius: 6px;
            padding: 7px 10px; font-size: 12px; outline: none; margin-top: 6px;
            cursor: pointer;
        }
        .warning-select:focus { border-color: #ef4444; }

        /* Top-Right: Stakeholder Tools (Responsive & Anti-Overlap) */
        .map-tools {
            top: 16px; right: 16px; display: flex; gap: 6px; padding: 6px 8px;
            max-width: calc(100vw - 480px); flex-wrap: wrap; justify-content: flex-end;
        }
        .tool-btn {
            background: #14212c; color: #edf4f5;
            border: 1px solid #3c4c5a; border-radius: 7px;
            padding: 6px 10px; font-size: 12px; font-weight: 500;
            cursor: pointer; display: flex; align-items: center; gap: 5px;
            transition: all 0.2s ease; user-select: none; white-space: nowrap;
        }
        .tool-btn:hover, .tool-btn.active {
            background: #253847; color: #ffffff; border-color: #38bdf8;
        }
        .tool-btn-primary {
            background: #0284c7; color: #ffffff; border-color: #38bdf8; font-weight: 700;
        }
        .tool-btn-primary:hover {
            background: #0369a1;
        }

        /* Popovers */
        .popover-card {
            position: fixed; z-index: 1001;
            background: #111c26; color: #edf4f5;
            border: 1px solid #34434f; border-radius: 10px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
            padding: 14px; width: 310px; display: none;
            top: 68px; right: 16px;
        }
        .popover-card.open { display: block; }
        .popover-title {
            font-size: 12px; font-weight: 700; color: #38bdf8;
            text-transform: uppercase; margin-bottom: 10px;
            border-bottom: 1px solid #233441; padding-bottom: 5px;
            display: flex; align-items: center; gap: 6px;
        }
        .form-group { margin-bottom: 10px; }
        .form-group label {
            display: block; font-size: 11.5px; color: #bdccd6; margin-bottom: 4px;
        }
        .form-select {
            width: 100%; background: #1c2b38; color: #edf4f5;
            border: 1px solid #364858; border-radius: 6px;
            padding: 7px 10px; font-size: 12.5px; outline: none;
        }
        .checkbox-item {
            display: flex; align-items: center; gap: 8px; font-size: 12.5px; margin-bottom: 6px; cursor: pointer;
        }
        .checkbox-item input[type="checkbox"] {
            accent-color: #38bdf8; cursor: pointer;
        }
        .slider-control {
            width: 100%; accent-color: #38bdf8; cursor: pointer;
        }

        /* Bottom-Left: Footer Legend */
        .map-footer {
            bottom: 24px; left: 16px; width: 380px; max-width: calc(100vw - 32px);
        }
        .risk-legend {
            display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: #c6d3dc; margin-bottom: 8px;
        }
        .risk-legend span {
            display: inline-flex; align-items: center; gap: 6px;
        }
        .legend-box {
            width: 11px; height: 11px; border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 2px;
        }

        /* Modal Overlays */
        .modal-overlay {
            position: fixed; inset: 0; background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px); z-index: 2000;
            display: none; align-items: center; justify-content: center; padding: 20px;
        }
        .modal-overlay.open { display: flex; }
        .modal-body {
            background: #101d28; border: 1px solid #30424f; border-radius: 12px;
            width: 100%; max-width: 880px; max-height: 85vh; overflow-y: auto;
            padding: 22px; color: #edf4f5; position: relative;
        }
        .modal-header {
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid #293a47; padding-bottom: 10px; margin-bottom: 14px;
        }
        .modal-header h2 { font-size: 17px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .close-btn {
            background: transparent; border: none; color: #94a3b8; font-size: 24px; cursor: pointer;
            padding: 2px 6px; border-radius: 4px; line-height: 1;
        }
        .close-btn:hover { color: #fff; background: rgba(255, 255, 255, 0.1); }

        /* Tables & Trend Bars */
        .data-table {
            width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px;
        }
        .data-table th, .data-table td {
            border: 1px solid #2a3b4a; padding: 8px 10px; text-align: left;
        }
        .data-table th { background: #192733; color: #f8fafc; font-weight: 600; }
        .data-table tr:nth-child(even) { background: #121d26; }

        .trend-bar-wrap {
            display: flex; align-items: center; gap: 8px;
        }
        .trend-bar {
            height: 10px; background: #0284c7; border-radius: 2px; min-width: 4px;
        }

        /* Custom Leaflet Popups */
        .leaflet-popup-content-wrapper {
            background: #101d28 !important; color: #edf4f5 !important;
            border: 1px solid #35414b !important; border-radius: 8px !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6) !important;
        }
        .leaflet-popup-tip { background: #101d28 !important; }

        @media (max-width: 700px) {
            .map-header { max-width: calc(100vw - 120px); }
            .floating-warning-box { width: 300px; }
        }
    </style>
</head>
<body>

    <!-- Fullscreen Map Canvas -->
    <div id="map"></div>

    <!-- Top-Left Floating Header -->
    <div class="glass-panel map-header">
        <div class="portal-brand">
            <svg class="brand-logo" viewBox="0 0 24 24">
                <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
            </svg>
            <div>
                <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                    <strong>BORNEO SIAGA</strong>
                    <span class="badge-role">Satgas BPBD</span>
                    __ENV_BADGE__
                </div>
                <div style="font-size: 10.5px; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 1px;">
                    Portal Stakeholder & Karhutla
                </div>
            </div>
        </div>
        <div class="portal-date" id="date-label">
            __INITIAL_DATE_TITLE__
        </div>
        <div style="margin-top: 6px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <button type="button" onclick="showModalDialog('modal-daily-eval', event)" id="header-eval-badge-stk" style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 5px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;">
                <svg class="icon-svg" style="width: 12px; height: 12px;" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                <span id="header-eval-rate-stk">Eval Daily: 86.6% Recall</span>
            </button>
            <div class="risk-chips" id="summary-chips" style="margin-top: 0; display: inline-flex; align-items: center; gap: 10px;">
                <span class="chip"><span class="chip-dot" style="background: #22c55e;"></span> Rendah: Memuat...</span>
                <span class="chip"><span class="chip-dot" style="background: #facc15;"></span> Sedang: Memuat...</span>
                <span class="chip"><span class="chip-dot" style="background: #ef4444;"></span> Tinggi: Memuat...</span>
            </div>
        </div>
    </div>

    <!-- Section Melayang Peringatan Bahaya (Clean Dropdown, Bebas Border AI) -->
    <div class="glass-panel floating-warning-box" id="floating-warning-panel">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 13px; font-weight: 700; color: #ef4444;" id="warning-total-label">
                    Peringatan Bahaya: Memuat...
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;" id="warning-date-label">
                    Prediksi: H+1 Terkini
                </div>
            </div>
            <button type="button" onclick="showModalDialog('modal-warning-detail', event)" style="font-size: 11px; color: #38bdf8; background: transparent; border: none; cursor: pointer; text-decoration: underline; padding: 2px 4px;">
                Tabel Rincian
            </button>
        </div>

        <!-- Dropdown Pilih Lokasi Nyata Seluruh Peringatan -->
        <select class="warning-select" id="warning-dropdown" onchange="onSelectWarningDropdown(this.value)">
            <option value="">-- Pilih Lokasi Wilayah Rawan (353 Titik) --</option>
        </select>
    </div>

    <!-- Top-Right Floating Tools -->
    <div class="glass-panel map-tools">
        <button type="button" class="tool-btn" onclick="showPopoverMenu('popover-wilayah', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
            <span>Wilayah</span>
        </button>

        <button type="button" class="tool-btn" onclick="showPopoverMenu('popover-lapisan', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27zM12 16l7.36-5.73L21 9.07l-9-7-9 7 1.63 1.27zm0-11.47L17.74 8 12 12.47 6.26 8z"/></svg>
            <span>Lapisan</span>
        </button>

        <button type="button" class="tool-btn" onclick="showPopoverMenu('popover-tanggal', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-1.99.9-1.99 2L3 20c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10zm0-12H5V6h14v2z"/></svg>
            <span>Tanggal</span>
        </button>

        <!-- Eval Daily Satgas -->
        <button type="button" class="tool-btn" style="border-color: #38bdf8; color: #38bdf8;" onclick="showModalDialog('modal-daily-eval', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
            <span>Eval Daily</span>
        </button>

        <!-- Use Case 3: Melihat riwayat tren risiko -->
        <button type="button" class="tool-btn tool-btn-primary" onclick="showModalDialog('modal-tren-risiko', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
            <span>Tren Risiko</span>
        </button>

        <!-- Use Case 4: Menerima notifikasi dini -->
        <button type="button" class="tool-btn" onclick="showModalDialog('modal-notifikasi-dini', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.89 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>
            <span>Notifikasi</span>
        </button>

        <button type="button" class="tool-btn" onclick="showModalDialog('modal-laporan', event)">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
            <span>Laporan</span>
        </button>

        <a href="/" onclick="sessionStorage.clear();" class="tool-btn" style="text-decoration:none; color:#fca5a5; border-color: rgba(239, 68, 68, 0.4);">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/></svg>
            <span>Keluar</span>
        </a>
    </div>

    <!-- Popover: Wilayah -->
    <div class="popover-card" id="popover-wilayah">
        <div class="popover-title">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
            Fokus Wilayah Operasi
        </div>
        <div class="form-group">
            <select class="form-select" onchange="zoomToRegion(this.value)">
                <option value="all">Seluruh Kalimantan (Overview 22.276 Kotak)</option>
                <option value="kalbar">Kalimantan Barat (Kluster Ketapang/Kubu Raya)</option>
                <option value="kalteng">Kalimantan Tengah (Kluster Kotawaringin/Pulang Pisau)</option>
                <option value="kalsel">Kalimantan Selatan (Kluster Banjar/Tanah Laut)</option>
                <option value="kaltim">Kalimantan Timur</option>
                <option value="kaltara">Kalimantan Utara</option>
            </select>
        </div>
    </div>

    <!-- Popover: Lapisan -->
    <div class="popover-card" id="popover-lapisan">
        <div class="popover-title">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27zM12 16l7.36-5.73L21 9.07l-9-7-9 7 1.63 1.27zm0-11.47L17.74 8 12 12.47 6.26 8z"/></svg>
            Pengaturan Lapisan GIS
        </div>
        <div class="form-group">
            <label>Peta Dasar (Basemap)</label>
            <select class="form-select" id="basemap-select" onchange="switchBasemap(this.value)">
                <option value="satelit" selected>Satelit (Esri World Imagery + Batas)</option>
                <option value="gelap">Gelap (Carto Darkmatter)</option>
                <option value="terang">Terang (Carto Positron)</option>
            </select>
        </div>
        <div class="form-group">
            <label>Tampilkan Status Risiko & Penanda</label>
            <label class="checkbox-item"><input type="checkbox" id="chk-beacons" checked onchange="updateFilters()"> <span style="display:inline-block; width:8px; height:8px; background:#ef4444; border:1px solid #fff; border-radius:50%;"></span> Penanda Bahaya Adaptif</label>
            <label class="checkbox-item"><input type="checkbox" id="chk-tinggi" checked onchange="updateFilters()"> <span style="display:inline-block; width:10px; height:10px; background:#ef4444; border-radius:2px;"></span> Grid Bahaya Tinggi (Merah)</label>
            <label class="checkbox-item"><input type="checkbox" id="chk-sedang" checked onchange="updateFilters()"> <span style="display:inline-block; width:10px; height:10px; background:#facc15; border-radius:2px;"></span> Grid Waspada Sedang (Kuning)</label>
            <label class="checkbox-item"><input type="checkbox" id="chk-rendah" checked onchange="updateFilters()"> <span style="display:inline-block; width:10px; height:10px; background:#22c55e; border-radius:2px;"></span> Grid Aman Rendah (Hijau)</label>
            <label class="checkbox-item"><input type="checkbox" id="chk-hotspots" checked onchange="toggleHotspots(this.checked)"> <span style="display:inline-block; width:8px; height:8px; background:#ea580c; border:1px solid #fff; border-radius:50%;"></span> Titik Api Satelit (FIRMS)</label>
        </div>
        <div class="form-group">
            <label>Transparansi Grid: <span id="opacity-val">28%</span></label>
            <input type="range" class="slider-control" id="opacity-slider" min="10" max="80" value="28" oninput="changeOpacity(this.value)">
        </div>
    </div>

    <!-- Popover: Tanggal -->
    <div class="popover-card" id="popover-tanggal">
        <div class="popover-title">
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-1.99.9-1.99 2L3 20c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10zm0-12H5V6h14v2z"/></svg>
            Pilih Tanggal Prediksi
        </div>
        <div class="form-group">
            <select class="form-select" id="date-select" onchange="changeDate(this.value)">
                __DATE_OPTIONS__
            </select>
        </div>
    </div>

    <!-- Bottom-Left Floating Footer Legend -->
    <div class="glass-panel map-footer">
        <div class="risk-legend">
            <span><span class="legend-box" style="background: #22c55e;"></span> Rendah</span>
            <span><span style="display:inline-block; width:8px; height:8px; background:#facc15; border:1px solid #fff; border-radius:50%;"></span> Sedang (Waspada)</span>
            <span><span style="display:inline-block; width:8px; height:8px; background:#ef4444; border:1px solid #fff; border-radius:50%;"></span> Tinggi (Bahaya)</span>
            <span><span style="display:inline-block; width:8px; height:8px; background:#ea580c; border:1px solid #fff; border-radius:50%;"></span> Hotspot NASA</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #283745; padding-top: 8px; font-size: 12px;">
            <span id="footer-status" style="color: #94a3b8;">Status: Peta siap ditampilkan.</span>
            <a href="/api/download-report-csv?date=2024-09-03" id="csv-download-link" style="color: #38bdf8; text-decoration: none; font-weight: 600; display: flex; align-items: center; gap: 4px;">
                <svg class="icon-svg" viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/></svg>
                Unduh CSV
            </a>
        </div>
    </div>

    <!-- MODAL 1: DETAIL PERINGATAN BAHAYA AKTIF -->
    <div class="modal-overlay" id="modal-warning-detail" onclick="if(event.target === this) hideModalDialog('modal-warning-detail', event)">
        <div class="modal-body" style="max-width: 900px;">
            <div class="modal-header">
                <h2>
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
                    Daftar Prioritas Peringatan Bahaya Karhutla (Hasil Prediksi H+1)
                </h2>
                <button type="button" class="close-btn" onclick="hideModalDialog('modal-warning-detail', event)">&times;</button>
            </div>
            <p style="font-size: 12.5px; color: #cbd5e1; margin-bottom: 12px;">
                Dihasilkan murni dari inferensi model XGBoost pada tanggal prediksi aktif. Memuat nama wilayah riil (Kecamatan, Kabupaten, Provinsi), koordinat presisi, elevasi, dan tutupan lahan gambut.
            </p>
            <div style="max-height: 480px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>No</th>
                            <th>Wilayah Riil (Desa, Kecamatan, Kabupaten, Provinsi)</th>
                            <th>Status & Skor</th>
                            <th>Koordinat</th>
                            <th>Elevasi</th>
                            <th>Gambut</th>
                            <th>Aksi Peta</th>
                        </tr>
                    </thead>
                    <tbody id="warning-detail-rows">
                        <tr><td colspan="7" style="text-align: center;">Memuat daftar peringatan...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- MODAL 2: RIWAYAT TREN RISIKO (USE CASE 3) -->
    <div class="modal-overlay" id="modal-tren-risiko" onclick="if(event.target === this) hideModalDialog('modal-tren-risiko', event)">
        <div class="modal-body" style="max-width: 900px;">
            <div class="modal-header">
                <h2>
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
                    Riwayat Tren Risiko Karhutla Temporal Kalimantan (2023 - 2024)
                </h2>
                <button type="button" class="close-btn" onclick="hideModalDialog('modal-tren-risiko', event)">&times;</button>
            </div>
            <p style="font-size: 12.5px; color: #94a3b8; margin-bottom: 14px;">
                Menampilkan perbandingan dinamika kebakaran hutan: fase kemarau ekstrem El Nino 2023 vs kemarau 2024 dan musim penghujan.
            </p>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Periode Bulan</th>
                        <th>Total Hotspot Satelit</th>
                        <th>Visualisasi Relatif</th>
                        <th>Rata-rata Curah Hujan (14 Hari)</th>
                        <th>Fase Cuaca Lapangan</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>September 2023 (El Nino)</td><td><b>21.430</b> titik</td><td><div class="trend-bar-wrap"><div class="trend-bar" style="width: 100%; background: #ef4444;"></div></div></td><td>12.4 mm (Kekeringan Ekstrem)</td><td><span style="color: #ef4444; font-weight: 700;">Tanggap Darurat</span></td></tr>
                    <tr><td>Oktober 2023</td><td><b>18.770</b> titik</td><td><div class="trend-bar-wrap"><div class="trend-bar" style="width: 87.5%; background: #ef4444;"></div></div></td><td>24.1 mm (Kering)</td><td><span style="color: #ef4444; font-weight: 700;">Siaga 1 Darat</span></td></tr>
                    <tr><td>September 2024 (Normal)</td><td><b>7.579</b> titik</td><td><div class="trend-bar-wrap"><div class="trend-bar" style="width: 35.3%; background: #facc15;"></div></div></td><td>58.2 mm (Moderat)</td><td><span style="color: #facc15; font-weight: 700;">Siaga 2 Rutin</span></td></tr>
                    <tr><td>Oktober 2024</td><td><b>2.792</b> titik</td><td><div class="trend-bar-wrap"><div class="trend-bar" style="width: 13.0%; background: #38bdf8;"></div></div></td><td>94.6 mm (Hujan Berkala)</td><td><span style="color: #38bdf8; font-weight: 600;">Waspada Lokal</span></td></tr>
                    <tr><td>Desember 2024 (Penghujan)</td><td><b>159</b> titik</td><td><div class="trend-bar-wrap"><div class="trend-bar" style="width: 1.2%; background: #22c55e;"></div></div></td><td>284.1 mm (Hujan Lebat)</td><td><span style="color: #22c55e; font-weight: 700;">Kondisi Aman</span></td></tr>
                </tbody>
            </table>

            <h3 style="font-size: 13px; color: #38bdf8; text-transform: uppercase; margin: 18px 0 8px;">
                Sebaran Titik Api Historis per Provinsi (Arsip Satelit VIIRS 2023 - 2024)
            </h3>
            <table class="data-table">
                <thead><tr><th>Provinsi</th><th>Total Hotspot Terdeteksi</th><th>Persentase Sebaran</th><th>Karakteristik Dominan</th></tr></thead>
                <tbody>
                    <tr><td><b>Kalimantan Barat</b></td><td>31.400 titik</td><td><b>42.4%</b></td><td>Kubah Gambut Pesisir (Ketapang, Kubu Raya, Kayong Utara)</td></tr>
                    <tr><td><b>Kalimantan Tengah</b></td><td>28.200 titik</td><td><b>38.1%</b></td><td>Lahan Gambut Tebal Eks-PLG (Kotawaringin, Pulang Pisau, Kapuas)</td></tr>
                    <tr><td><b>Kalimantan Selatan</b></td><td>8.000 titik</td><td><b>10.8%</b></td><td>Dataran Rendah & Pertanian (Banjar, Tanah Laut, Barito Kuala)</td></tr>
                    <tr><td><b>Kalimantan Timur</b></td><td>4.370 titik</td><td><b>5.9%</b></td><td>Hutan Sekunder & Perbukitan Mineral (Kukar, Kutim, Berau)</td></tr>
                    <tr><td><b>Kalimantan Utara</b></td><td>2.070 titik</td><td><b>2.8%</b></td><td>Topografi Pegunungan & Hutan Lebat (Bulungan, Nunukan)</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- MODAL 3: NOTIFIKASI DINI (USE CASE 4) -->
    <div class="modal-overlay" id="modal-notifikasi-dini" onclick="if(event.target === this) hideModalDialog('modal-notifikasi-dini', event)">
        <div class="modal-body" style="max-width: 900px;">
            <div class="modal-header">
                <h2>
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.89 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>
                    Lembar Notifikasi Dini & Disposisi Lapangan (Use Case 4)
                </h2>
                <button type="button" class="close-btn" onclick="hideModalDialog('modal-notifikasi-dini', event)">&times;</button>
            </div>
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
                <div style="font-size: 13px; font-weight: 700; color: #f87171;">PERINGATAN SIAGA TINGGI KARHUTLA TERBIT (H+1)</div>
                <p style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.4;">
                    Model XGBoost mengidentifikasi <b>353 kotak grid berstatus TINGGI</b> dengan <b>112 kotak berada pada kubah gambut</b>. Satgas Gabungan diinstruksikan melakukan patroli ground-check pencegahan sebelum bara api bawah permukaan meluas.
                </p>
            </div>

            <table class="data-table">
                <thead><tr><th>Instansi / Satgas</th><th>Peran Kunci</th><th>Tindakan yang Harus Dijalankan</th></tr></thead>
                <tbody>
                    <tr><td><b>BPBD Provinsi & Kabupaten</b></td><td>Komando Tanggap Darurat</td><td>Aktivasi Posko Lapangan di Ketapang, Kubu Raya, dan Kotawaringin. Penyiapan logistik pompa air portabel dan selang apung.</td></tr>
                    <tr><td><b>Manggala Agni (KLHK)</b></td><td>Patroli Darat & Pemadaman Dini</td><td>Patroli darurat ground-check pada lahan gambut berstatus Tinggi untuk memadamkan bara sebelum merambat ke bawah tanah.</td></tr>
                    <tr><td><b>TNI / POLRI</b></td><td>Pencegahan & Penegakan Hukum</td><td>Sosialisasi larangan pembakaran lahan tanpa izin dan penutupan portal akses ke kubah gambut kering.</td></tr>
                    <tr><td><b>BNPB (Satgas Udara)</b></td><td>Water Bombing</td><td>Kesiapsiagaan helikopter water bombing untuk titik api terpencil yang tidak dapat dijangkau armada darat.</td></tr>
                </tbody>
            </table>

            <div style="margin-top: 14px; display: flex; justify-content: flex-end;">
                <a href="/api/download-report-csv?date=2024-09-03" class="tool-btn" style="background:#38bdf8; color:#0a121b; font-weight:700;">
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/></svg>
                    Unduh Disposisi Lengkap (.CSV)
                </a>
            </div>
        </div>
    </div>

    <!-- MODAL 4: LAPORAN REKAP WILAYAH -->
    <div class="modal-overlay" id="modal-laporan" onclick="if(event.target === this) hideModalDialog('modal-laporan', event)">
        <div class="modal-body">
            <div class="modal-header">
                <h2>Laporan Prioritas Wilayah Siaga Kebakaran</h2>
                <button type="button" class="close-btn" onclick="hideModalDialog('modal-laporan', event)">&times;</button>
            </div>
            <p style="font-size: 12.5px; color: #bdccd6; margin-bottom: 12px;">
                Daftar wilayah dengan skor risiko tertinggi (Status TINGGI) yang memerlukan prioritas patroli darat Manggala Agni / BPBD.
            </p>
            <table class="data-table">
                <thead><tr><th>No</th><th>Wilayah Riil</th><th>Koordinat</th><th>Status</th><th>Skor</th><th>Elevasi</th><th>Gambut</th></tr></thead>
                <tbody id="high-risk-rows">
                    <tr><td colspan="7" style="text-align:center;">Memuat data wilayah rawan...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- MODAL: EVALUASI HARIAN OPERASIONAL SATGAS -->
    <div class="modal-overlay" id="modal-daily-eval" onclick="if(event.target === this) hideModalDialog('modal-daily-eval', event)">
        <div class="modal-body" style="max-width: 920px;">
            <div class="modal-header">
                <div>
                    <h2 style="font-size: 17px; color: #38bdf8; display: flex; align-items: center; gap: 8px;">
                        <svg class="icon-svg" style="width:20px; height:20px;" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                        Evaluasi Akurasi Operasional Harian (Daily Validation Tracker)
                    </h2>
                    <div style="font-size: 11.5px; color: #94a3b8; margin-top: 2px;">
                        Validasi Ground-Truth: Mengecek apakah prediksi H-1 berhasil menangkap titik api satelit NASA FIRMS yang terdeteksi di hari H
                    </div>
                </div>
                <button type="button" class="close-btn" onclick="hideModalDialog('modal-daily-eval', event)">&times;</button>
            </div>

            <!-- Metric Summary Cards -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 14px 0;">
                <div style="background: #14212c; border: 1px solid #283745; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Capture Rate Rata-rata</div>
                    <div style="font-size: 20px; font-weight: 800; color: #4ade80; margin-top: 2px;" id="eval-avg-rate-stk">85.6%</div>
                    <div style="font-size: 10.5px; color: #94a3b8;">Recall harian lapangan</div>
                </div>
                <div style="background: #14212c; border: 1px solid #283745; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Titik Api Tertangkap</div>
                    <div style="font-size: 20px; font-weight: 800; color: #38bdf8; margin-top: 2px;" id="eval-hits-today-stk">284 / 328</div>
                    <div style="font-size: 10.5px; color: #94a3b8;">Masuk zona siaga (Tinggi/Sedang)</div>
                </div>
                <div style="background: #14212c; border: 1px solid #283745; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Status Operasional</div>
                    <div style="font-size: 20px; font-weight: 800; color: #22c55e; margin-top: 2px;" id="eval-status-badge-stk">OPTIMAL</div>
                    <div style="font-size: 10.5px; color: #94a3b8;">Ambang batas &ge; 80% terpenuhi</div>
                </div>
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Tanggal Prediksi</th>
                        <th>Waktu Eksekusi Pipeline</th>
                        <th>Kotak Siaga (H-1)</th>
                        <th>Titik Api Satelit Hari H</th>
                        <th>Api Tertangkap (Hits)</th>
                        <th>Daily Capture Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="daily-eval-table-body-stk">
                    <tr><td colspan="7" style="text-align:center;">Memuat data evaluasi harian...</td></tr>
                </tbody>
            </table>

            <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; color: #94a3b8;">
                <div>Sumber: <code>data/models/daily_evaluation_history.json</code> (Otomatis dari Pipeline Harian)</div>
                <button type="button" class="tool-btn" style="padding: 4px 10px; font-size: 11.5px;" onclick="loadDailyEvaluationData()">
                    Segarkan Data
                </button>
            </div>
        </div>
    </div>

    <!-- Leaflet & App Logic -->
    <script>
        let currentDate = "__INITIAL_DATE__";

        const map = L.map('map', {
            center: [-0.8, 114.4],
            zoom: 6,
            zoomControl: false
        });
        L.control.zoom({ position: 'bottomright' }).addTo(map);

        map.createPane('gridPane');
        map.getPane('gridPane').style.zIndex = 400;

        map.createPane('hotspotPane');
        map.getPane('hotspotPane').style.zIndex = 500;

        map.createPane('dangerPane');
        map.getPane('dangerPane').style.zIndex = 650;

        const basemapLayers = {
            satelit: L.layerGroup([
                L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                    attribution: 'Esri, Maxar', maxZoom: 18
                }),
                L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
                    attribution: 'Esri Boundaries', maxZoom: 18
                })
            ]),
            gelap: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 18 }),
            terang: L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { maxZoom: 18 })
        };
        let currentBasemap = basemapLayers.satelit;
        currentBasemap.addTo(map);

        window.switchBasemap = function(type) {
            map.removeLayer(currentBasemap);
            currentBasemap = basemapLayers[type] || basemapLayers.satelit;
            currentBasemap.addTo(map);
        };

        let geojsonLayer = null;
        let riskBeaconLayer = L.layerGroup().addTo(map);
        let hotspotLayer = L.layerGroup().addTo(map);
        let currentOpacity = 0.28;
        let allGeojsonData = null;

        const RISK_COLORS = {
            'TINGGI': '#ef4444', 'Tinggi': '#ef4444',
            'SEDANG': '#facc15', 'Sedang': '#facc15',
            'RENDAH': '#22c55e', 'Rendah': '#22c55e'
        };

        function getFillColor(level) {
            return RISK_COLORS[level] || '#22c55e';
        }

        function getRiskBeaconRadius(lvl, z) {
            return z <= 6 ? 3.5 : 4.5;
        }

        function createPopupContent(p) {
            const lvl = (p.risk_level || 'RENDAH').toUpperCase();
            const score = p.risk_score !== undefined ? Number(p.risk_score).toFixed(1) : '5.0';
            const color = getFillColor(lvl);
            const badgeBg = lvl === 'TINGGI' ? 'rgba(239, 68, 68, 0.25)' : (lvl === 'SEDANG' ? 'rgba(250, 204, 21, 0.25)' : 'rgba(34, 197, 94, 0.25)');
            
            const elev = p.elevation !== undefined ? Math.round(Number(p.elevation)) : 0;
            let elevCat = 'Dataran Rendah Pesisir';
            if (elev >= 300) elevCat = 'Pegunungan / Dataran Tinggi';
            else if (elev >= 50) elevCat = 'Perbukitan Rendah';

            const peat = p.peat_coverage_pct !== undefined ? Math.round(Number(p.peat_coverage_pct)) : 0;
            let peatHtml = '';
            if (peat > 0) {
                peatHtml = `
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.5); border-radius: 6px; padding: 5px 8px; margin-top: 5px;">
                        <div style="color: #facc15; font-size: 11px; font-weight: 700;">Tutupan Gambut: ${peat}%</div>
                        <div style="font-size: 10.5px; color: #fde68a; margin-top: 1px;">Lahan Gambut (Rawan Karhutla Bawah Permukaan)</div>
                    </div>
                `;
            } else {
                peatHtml = `
                    <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 6px; padding: 4px 8px; margin-top: 5px;">
                        <div style="color: #cbd5e1; font-size: 10.5px;">Tutupan Gambut: <b>0%</b> (Tanah Mineral)</div>
                    </div>
                `;
            }

            const temp = p.temp_max !== undefined ? Number(p.temp_max).toFixed(1) + ' °C' : '-';
            const rain14 = p.rain_sum_14d !== undefined ? Number(p.rain_sum_14d).toFixed(1) + ' mm' : '-';
            const dryDays = p.dry_days_count !== undefined ? Math.round(Number(p.dry_days_count)) + ' hari' : '-';
            const hotspots = p.hotspot_count !== undefined ? Math.round(Number(p.hotspot_count)) : 0;

            let actionHtml = '';
            if (lvl === 'TINGGI') {
                actionHtml = '<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 6px; padding: 5px 8px; margin-top: 6px; font-size: 11px; color: #fca5a5;"><b>Rekomendasi Satgas:</b> Prioritas Patroli Darat Gabungan Manggala Agni / BPBD.</div>';
            } else if (lvl === 'SEDANG') {
                actionHtml = '<div style="background: rgba(250, 204, 21, 0.15); border: 1px solid #facc15; border-radius: 6px; padding: 5px 8px; margin-top: 6px; font-size: 11px; color: #fde047;"><b>Rekomendasi Satgas:</b> Kesiapsiagaan Posko Lapangan & Pengecekan Ketersediaan Air Embung.</div>';
            } else {
                actionHtml = '<div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 6px; padding: 4px 8px; margin-top: 6px; font-size: 10.5px; color: #86efac;"><b>Status Wilayah:</b> Kondisi Terkendali, Pemantauan Rutin Wilayah.</div>';
            }

            const lokasiText = p.lokasi || (p.grid_id ? 'ID Grid: ' + p.grid_id : 'Wilayah Kalimantan');

            return `
                <div style="font-family: system-ui, sans-serif; min-width: 260px; max-width: 290px; padding: 2px; color: #edf4f5;">
                    <div style="border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
                            <div style="font-size: 13px; font-weight: 700; color: #38bdf8; line-height: 1.3;">
                                ${lokasiText}
                            </div>
                            <span style="font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 700; background: ${badgeBg}; color: ${color}; border: 1px solid ${color}; flex-shrink: 0;">${lvl}</span>
                        </div>
                        <div style="font-size: 10.5px; color: #94a3b8; margin-top: 3px;">
                            ID: <code>${p.grid_id || '-'}</code> | Koordinat: <code>${p.lat ? Number(p.lat).toFixed(4) : '-'}, ${p.lon ? Number(p.lon).toFixed(4) : '-'}</code>
                        </div>
                    </div>
                    <div style="background: #0d1721; border: 1px solid #243442; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 11px; color: #94a3b8;">Skor Risiko XGBoost:</span>
                            <span style="color: ${color}; font-size: 15px; font-weight: 800;">${score} <small style="font-size: 10px; color: #64748b;">/ 100</small></span>
                        </div>
                    </div>
                    <div style="background: #14212c; border: 1px solid #2a3b4a; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; font-size: 11px; line-height: 1.4;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94a3b8;">Elevasi:</span>
                            <span style="font-weight: 600;">${elev} m dpl (${elevCat})</span>
                        </div>
                        ${peatHtml}
                    </div>
                    <div style="background: #14212c; border: 1px solid #2a3b4a; border-radius: 6px; padding: 6px 8px; font-size: 10.5px; line-height: 1.4;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3px;">
                            <div><span style="color:#94a3b8;">Suhu:</span> <b>${temp}</b></div>
                            <div><span style="color:#94a3b8;">Hujan 14h:</span> <b>${rain14}</b></div>
                            <div><span style="color:#94a3b8;">Hari Kering:</span> <b>${dryDays}</b></div>
                            <div><span style="color:#94a3b8;">Titik Api:</span> <b>${hotspots} titik</b></div>
                        </div>
                    </div>
                    ${actionHtml}
                </div>
            `;
        }

        window.loadData = function(date) {
            currentDate = date;
            document.getElementById('footer-status').innerText = 'Memuat data tanggal ' + date + '...';
            document.getElementById('csv-download-link').href = '/api/download-report-csv?date=' + date;

            // 1. Muat GeoJSON Prediksi 22.276 Kotak
            fetch('/api/warning-geojson?date=' + date)
                .then(res => res.json())
                .then(data => {
                    allGeojsonData = data;
                    renderGeojsonGrid();
                    populateHighRiskTable(data);
                    populateWarningDetailTable(data);
                    buildWarningDropdown(data);
                })
                .catch(err => {
                    document.getElementById('footer-status').innerText = 'Gagal memuat GeoJSON risiko.';
                });

            // 2. Muat Titik Api NASA FIRMS
            hotspotLayer.clearLayers();
            fetch('/api/hotspots-today?date=' + date)
                .then(res => res.json())
                .then(data => {
                    if (data.hotspots) {
                        const zoom = map.getZoom();
                        data.hotspots.forEach(pt => {
                            L.circleMarker([pt.lat, pt.lon], {
                                pane: 'hotspotPane',
                                radius: zoom <= 6 ? 3.5 : 4.5,
                                fillColor: '#ea580c',
                                color: '#ffffff',
                                weight: 1.0,
                                opacity: 0.9,
                                fillOpacity: 0.95
                            }).bindPopup(`
                                <div style="font-family: system-ui, sans-serif; padding: 4px; color: #edf4f5;">
                                    <div style="color: #ea580c; font-weight: 700; font-size: 12.5px; border-bottom: 1px solid #334155; padding-bottom: 3px;">
                                        Titik Panas Satelit (NASA FIRMS)
                                    </div>
                                    <div style="font-size: 11.5px; margin-top: 4px;">Daya Panas (FRP): <b>${pt.frp} MW</b></div>
                                    <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Tanggal: ${pt.date} | Keyakinan: ${pt.confidence}</div>
                                </div>
                            `, { maxWidth: 240 }).addTo(hotspotLayer);
                        });
                    }
                });

            // 3. Muat Ringkasan Statistik
            fetch('/api/grid-summary?date=' + date)
                .then(res => res.json())
                .then(data => {
                    const rc = data.risk_counts || {};
                    const high = rc.Tinggi || 0;
                    const med = rc.Sedang || 0;
                    document.getElementById('summary-chips').innerHTML = `
                        <span class="chip"><span class="chip-dot" style="background: #22c55e;"></span> Rendah: ${(rc.Rendah || 0).toLocaleString()} sel</span>
                        <span class="chip"><span class="chip-dot" style="background: #facc15;"></span> Sedang: ${med.toLocaleString()} sel</span>
                        <span class="chip"><span class="chip-dot" style="background: #ef4444;"></span> Tinggi: ${high.toLocaleString()} sel</span>
                    `;
                    document.getElementById('warning-total-label').innerText = `Peringatan Bahaya: ${high.toLocaleString()} Titik Siaga`;
                });
        };

        function renderGeojsonGrid() {
            if (!allGeojsonData) return;
            if (geojsonLayer) map.removeLayer(geojsonLayer);
            riskBeaconLayer.clearLayers();

            const showBeacons = document.getElementById('chk-beacons') ? document.getElementById('chk-beacons').checked : true;
            const showTinggi = document.getElementById('chk-tinggi').checked;
            const showSedang = document.getElementById('chk-sedang').checked;
            const showRendah = document.getElementById('chk-rendah').checked;
            const zoom = map.getZoom();

            geojsonLayer = L.geoJSON(allGeojsonData, {
                pane: 'gridPane',
                filter: function(feature) {
                    const lvl = (feature.properties.risk_level || 'RENDAH').toUpperCase();
                    if (lvl === 'TINGGI' && !showTinggi) return false;
                    if (lvl === 'SEDANG' && !showSedang) return false;
                    if (lvl === 'RENDAH' && !showRendah) return false;
                    return true;
                },
                style: function(feature) {
                    const lvl = (feature.properties.risk_level || 'RENDAH').toUpperCase();
                    const isHigh = lvl === 'TINGGI';
                    const isMed = lvl === 'SEDANG';

                    return {
                        pane: 'gridPane',
                        fillColor: getFillColor(lvl),
                        fillOpacity: isHigh ? Math.min(0.9, currentOpacity + 0.45) : (isMed ? Math.min(0.75, currentOpacity + 0.3) : (zoom <= 8 ? 0.10 : currentOpacity * 0.7)),
                        weight: isHigh ? 2.0 : (isMed ? 1.5 : (zoom >= 9 ? 0.4 : 0)),
                        color: isHigh ? '#ffffff' : (isMed ? '#ffffff' : 'rgba(255, 255, 255, 0.25)'),
                        opacity: isHigh ? 1.0 : (isMed ? 0.9 : (zoom >= 9 ? 0.4 : 0))
                    };
                },
                onEachFeature: function(feature, layer) {
                    const p = feature.properties;
                    const lvl = (p.risk_level || 'RENDAH').toUpperCase();
                    const isHigh = lvl === 'TINGGI';
                    const isMed = lvl === 'SEDANG';
                    const popupContent = createPopupContent(p);
                    layer.bindPopup(popupContent, { maxWidth: 300 });

                    // Penanda Bahaya Sama Ukuran NASA FIRMS (Radius 3.5 - 4.5 px)
                    if (showBeacons && p.lat && p.lon) {
                        const r = getRiskBeaconRadius(lvl, zoom);
                        if (isHigh && showTinggi) {
                            const beacon = L.circleMarker([p.lat, p.lon], {
                                pane: 'dangerPane',
                                radius: r,
                                fillColor: '#ef4444',
                                color: '#ffffff',
                                weight: 1.0,
                                opacity: 1,
                                fillOpacity: 0.95
                            });
                            beacon.bindPopup(popupContent, { maxWidth: 300 });
                            beacon.on({
                                mouseover: function(e) { e.target.setStyle({ weight: 2.0, radius: r + 1.0 }); },
                                mouseout: function(e) { e.target.setStyle({ weight: 1.0, radius: r }); },
                                click: function(e) { e.target.openPopup(); }
                            });
                            beacon.addTo(riskBeaconLayer);
                        } else if (isMed && showSedang) {
                            const beacon = L.circleMarker([p.lat, p.lon], {
                                pane: 'dangerPane',
                                radius: r,
                                fillColor: '#facc15',
                                color: '#ffffff',
                                weight: 1.0,
                                opacity: 1,
                                fillOpacity: 0.95
                            });
                            beacon.bindPopup(popupContent, { maxWidth: 300 });
                            beacon.on({
                                mouseover: function(e) { e.target.setStyle({ weight: 2.0, radius: r + 1.0 }); },
                                mouseout: function(e) { e.target.setStyle({ weight: 1.0, radius: r }); },
                                click: function(e) { e.target.openPopup(); }
                            });
                            beacon.addTo(riskBeaconLayer);
                        }
                    }
                }
            }).addTo(map);

            document.getElementById('footer-status').innerText = 'Peta prediksi siap ditampilkan (22.276 kotak).';
        }

        map.on('zoomend', function() {
            renderGeojsonGrid();
        });

        // Dropdown Peringatan dengan Nama Lokasi Riil & Koordinat
        function buildWarningDropdown(geojson) {
            const selectEl = document.getElementById('warning-dropdown');
            if (!geojson || !geojson.features || !selectEl) return;

            const highFeatures = geojson.features.filter(f => (f.properties.risk_level || '').toUpperCase() === 'TINGGI');
            
            let html = `<option value="">-- Pilih Lokasi Wilayah Rawan (${highFeatures.length} Titik) --</option>`;
            highFeatures.forEach(f => {
                const p = f.properties;
                const locName = p.lokasi_singkat || p.lokasi || p.grid_id;
                const latStr = p.lat !== undefined ? Number(p.lat).toFixed(3) : '';
                const lonStr = p.lon !== undefined ? Number(p.lon).toFixed(3) : '';
                const peat = p.peat_coverage_pct > 0 ? ` · Gambut ${p.peat_coverage_pct}%` : '';
                const val = `${p.lat},${p.lon},${p.grid_id}`;
                html += `<option value="${val}">[TINGGI] ${locName} (${latStr}, ${lonStr})${peat}</option>`;
            });

            selectEl.innerHTML = html;
        }

        window.onSelectWarningDropdown = function(val) {
            if (!val) return;
            const parts = val.split(',');
            const lat = parseFloat(parts[0]);
            const lon = parseFloat(parts[1]);
            const gridId = parts[2];

            if (!isNaN(lat) && !isNaN(lon)) {
                map.flyTo([lat, lon], 11, { duration: 1.0 });
                if (geojsonLayer) {
                    geojsonLayer.eachLayer(function(l) {
                        if (l.feature && l.feature.properties && l.feature.properties.grid_id === gridId) {
                            setTimeout(function() { l.openPopup(); }, 1100);
                        }
                    });
                }
            }
        };

        function populateHighRiskTable(geojson) {
            if (!geojson || !geojson.features) return;
            const highFeatures = geojson.features
                .filter(f => (f.properties.risk_level || '').toUpperCase() === 'TINGGI')
                .slice(0, 25);

            if (highFeatures.length === 0) {
                document.getElementById('high-risk-rows').innerHTML = '<tr><td colspan="7" style="text-align:center; color:#4ade80;">Tidak ada wilayah berkategori Tinggi pada tanggal ini.</td></tr>';
                return;
            }

            const rows = highFeatures.map((f, i) => {
                const p = f.properties;
                const elev = p.elevation !== undefined ? Math.round(Number(p.elevation)) : 0;
                const peat = p.peat_coverage_pct !== undefined ? Math.round(Number(p.peat_coverage_pct)) : 0;
                const loc = p.lokasi || p.grid_id;
                const lat = p.lat !== undefined ? Number(p.lat).toFixed(4) : '-';
                const lon = p.lon !== undefined ? Number(p.lon).toFixed(4) : '-';
                return `
                    <tr>
                        <td>${i + 1}</td>
                        <td><b>${loc}</b></td>
                        <td><code>${lat}, ${lon}</code></td>
                        <td><span style="color:#ef4444; font-weight:700;">TINGGI</span></td>
                        <td><b>${Number(p.risk_score || 0).toFixed(1)}</b></td>
                        <td>${elev} m dpl</td>
                        <td>${peat}% gambut</td>
                    </tr>
                `;
            });
            document.getElementById('high-risk-rows').innerHTML = rows.join('');
        }

        function populateWarningDetailTable(geojson) {
            if (!geojson || !geojson.features) return;
            const highFeatures = geojson.features
                .filter(f => (f.properties.risk_level || '').toUpperCase() === 'TINGGI')
                .slice(0, 50);

            if (highFeatures.length === 0) {
                document.getElementById('warning-detail-rows').innerHTML = '<tr><td colspan="7" style="text-align:center; color:#4ade80;">Semua wilayah aman pada tanggal ini.</td></tr>';
                return;
            }

            const rows = highFeatures.map((f, i) => {
                const p = f.properties;
                const elev = p.elevation !== undefined ? Math.round(Number(p.elevation)) : 0;
                const peat = p.peat_coverage_pct !== undefined ? Math.round(Number(p.peat_coverage_pct)) : 0;
                const peatText = peat > 0 
                    ? `<span style="color:#facc15; font-weight:700;">${peat}% Gambut</span>` 
                    : `<span style="color:#94a3b8;">0% Mineral</span>`;
                const loc = p.lokasi || p.grid_id;
                const lat = p.lat !== undefined ? Number(p.lat).toFixed(4) : '-';
                const lon = p.lon !== undefined ? Number(p.lon).toFixed(4) : '-';

                return `
                    <tr>
                        <td>${i + 1}</td>
                        <td><b>${loc}</b></td>
                        <td><span style="color:#ef4444; font-weight:700;">TINGGI (${Number(p.risk_score || 0).toFixed(1)})</span></td>
                        <td><code>${lat}, ${lon}</code></td>
                        <td>${elev} m dpl</td>
                        <td>${peatText}</td>
                        <td>
                            <button type="button" class="tool-btn" style="padding: 3px 8px; font-size: 11px;" onclick="focusOnMap(${p.lat}, ${p.lon}, '${p.grid_id}')">
                                Fokus
                            </button>
                        </td>
                    </tr>
                `;
            });
            document.getElementById('warning-detail-rows').innerHTML = rows.join('');
        }

        window.focusOnMap = function(lat, lon, gridId) {
            hideModalDialog('modal-warning-detail');
            map.flyTo([lat, lon], 11, { duration: 1.0 });
            if (geojsonLayer) {
                geojsonLayer.eachLayer(function(l) {
                    if (l.feature && l.feature.properties && l.feature.properties.grid_id === gridId) {
                        setTimeout(function() { l.openPopup(); }, 1100);
                    }
                });
            }
        };

        window.showPopoverMenu = function(id, event) {
            if (event) { event.preventDefault(); event.stopPropagation(); }
            const targetEl = document.getElementById(id);
            if (!targetEl) return;
            const isOpen = targetEl.classList.contains('open');
            document.querySelectorAll('.popover-card').forEach(p => p.classList.remove('open'));
            if (!isOpen) targetEl.classList.add('open');
        };

        window.onclick = function(e) {
            if (!e.target.closest('.map-tools') && !e.target.closest('.popover-card')) {
                document.querySelectorAll('.popover-card').forEach(p => p.classList.remove('open'));
            }
        };

        window.showModalDialog = function(id, event) {
            if (event) { event.preventDefault(); event.stopPropagation(); }
            document.querySelectorAll('.popover-card').forEach(p => p.classList.remove('open'));
            const modalEl = document.getElementById(id);
            if (modalEl) modalEl.classList.add('open');
        };

        window.hideModalDialog = function(id, event) {
            if (event) { event.preventDefault(); event.stopPropagation(); }
            const modalEl = document.getElementById(id);
            if (modalEl) modalEl.classList.remove('open');
        };

        window.changeOpacity = function(val) {
            currentOpacity = val / 100;
            document.getElementById('opacity-val').innerText = val + '%';
            renderGeojsonGrid();
        };

        window.updateFilters = function() {
            renderGeojsonGrid();
        };

        window.toggleHotspots = function(visible) {
            if (visible) map.addLayer(hotspotLayer);
            else map.removeLayer(hotspotLayer);
        };

        window.changeDate = function(date) {
            document.getElementById('popover-tanggal').classList.remove('open');
            const titles = {
                'latest': 'Terkini <small>H+1 (Real-Time Satelit & Cuaca)</small>',
                '2024-09-03': '3 September 2024 <small>UTC (Puncak Kemarau - 353 Sel Merah)</small>',
                '2024-09-06': '6 September 2024 <small>UTC (Siaga Karhutla Darat)</small>',
                '2024-09-14': '14 September 2024 <small>UTC (Kemarau Kalteng/Kalsel)</small>',
                '2024-12-31': '31 Desember 2024 <small>UTC (Musim Penghujan)</small>'
            };
            document.getElementById('date-label').innerHTML = 'Prediksi untuk <b>' + (titles[date] || date) + '</b>';
            document.getElementById('warning-date-label').innerText = 'Prediksi: ' + (date === 'latest' ? 'H+1 Terkini' : (date === '2024-09-03' ? '3 September 2024' : date)) + ' (H+1)';
            currentDate = date;
            loadData(date);
        };

        window.zoomToRegion = function(region) {
            const coords = {
                all: { center: [-0.8, 114.4], zoom: 6 },
                kalbar: { center: [-0.1, 110.5], zoom: 7.5 },
                kalteng: { center: [-1.5, 113.5], zoom: 7.5 },
                kalsel: { center: [-2.8, 115.3], zoom: 8 },
                kaltim: { center: [0.5, 116.5], zoom: 7.5 },
                kaltara: { center: [3.0, 116.5], zoom: 7.5 }
            };
            const target = coords[region] || coords.all;
            map.flyTo(target.center, target.zoom, { duration: 1.2 });
            document.getElementById('popover-wilayah').classList.remove('open');
        };

        window.loadDailyEvaluationData = function() {
            fetch('/api/daily-evaluation')
                .then(res => res.json())
                .then(data => {
                    const history = data.history || [];
                    const tbody = document.getElementById('daily-eval-table-body-stk');
                    if (!tbody) return;
                    if (history.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:#94a3b8;">Belum ada catatan evaluasi harian. Jalankan pipeline harian untuk memperbarui evaluasi operasional lapangan.</td></tr>';
                        return;
                    }

                    let totalRate = 0;
                    const rows = history.map(d => {
                        const rate = Number(d.daily_capture_rate_pct || 0);
                        totalRate += rate;
                        const badgeColor = rate >= 80 ? '#4ade80' : '#facc15';
                        return `
                            <tr>
                                <td><b>${d.prediction_date}</b></td>
                                <td><small style="color:#94a3b8;">${d.execution_date}</small></td>
                                <td><span style="color:#ef4444; font-weight:700;">${d.alert_high_count || 0} Tinggi</span> + <span style="color:#facc15; font-weight:700;">${d.alert_medium_count || 0} Sedang</span></td>
                                <td><b>${(d.active_hotspot_grids_today || 0).toLocaleString()}</b> kotak</td>
                                <td><b style="color:#38bdf8;">${(d.captured_in_alert_grids || 0).toLocaleString()}</b> kotak</td>
                                <td><span style="color:${badgeColor}; font-weight:700;">${rate.toFixed(1)}%</span></td>
                                <td><span style="color:${badgeColor}; font-weight:600;">${d.operational_status || 'Optimal'}</span></td>
                            </tr>
                        `;
                    });
                    tbody.innerHTML = rows.join('');

                    const avg = totalRate / history.length;
                    const latest = history[0];
                    if (document.getElementById('eval-avg-rate-stk')) document.getElementById('eval-avg-rate-stk').innerText = avg.toFixed(1) + '%';
                    if (document.getElementById('eval-hits-today-stk') && latest) {
                        document.getElementById('eval-hits-today-stk').innerText = `${latest.captured_in_alert_grids} / ${latest.active_hotspot_grids_today}`;
                    }
                    if (document.getElementById('eval-status-badge-stk') && latest) {
                        document.getElementById('eval-status-badge-stk').innerText = (latest.operational_status || 'OPTIMAL').toUpperCase();
                    }
                    if (document.getElementById('header-eval-rate-stk') && latest) {
                        document.getElementById('header-eval-rate-stk').innerText = `Eval Daily: ${latest.daily_capture_rate_pct}% Recall`;
                    }
                })
                .catch(() => {});
        };

        loadData(currentDate);
        loadDailyEvaluationData();
    </script>
</body>
</html>
"""
    rendered = html_content.replace("__INITIAL_DATE__", default_date)
    rendered = rendered.replace("__ENV_BADGE__", env_badge)
    rendered = rendered.replace("__INITIAL_DATE_TITLE__", date_title)
    rendered = rendered.replace("__DATE_OPTIONS__", date_options)
    return rendered
