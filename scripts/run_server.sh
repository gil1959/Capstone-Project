#!/bin/bash
# ==============================================================================
# BORNEO SIAGA - Production Web Server Runner
# Menjalankan FastAPI Portal dalam mode Produksi (APP_ENV=production)
# Memuat prediksi terkini (H+1 Live Daily) dan Evaluasi Harian Lapangan.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Tentukan Python / Uvicorn dari Virtualenv jika ada
if [ -d "$PROJECT_ROOT/venv" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# Konfigurasi Environment Server
export APP_ENV=production
export PYTHONUNBUFFERED=1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"

echo "======================================================================"
echo "  [BORNEO SIAGA] MENJALANKAN WEB SERVER PRODUKSI (PROD)"
echo "  Host: $HOST | Port: $PORT | Workers: $WORKERS"
echo "  Mode: $APP_ENV (Otomatis menampilkan Prediksi H+1 Terkini & Eval Daily)"
echo "======================================================================"

exec $PYTHON_EXEC -m uvicorn src.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --proxy-headers \
    --forwarded-allow-ips='*'
