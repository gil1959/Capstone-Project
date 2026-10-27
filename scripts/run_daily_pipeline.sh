#!/bin/bash
# ==============================================================================
# BORNEO SIAGA - Production Daily Pipeline Runner
# Menjalankan penarikan satelit FIRMS NRT, cuaca Open-Meteo, rekayasa 47 fitur,
# inferensi XGBoost untuk 22.276 kotak Kalimantan, dan Daily Validation Tracker.
# ==============================================================================

set -e

# Direktori proyek (relatif terhadap letak script ini)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Tentukan Python dari Virtualenv jika ada
if [ -d "$PROJECT_ROOT/venv" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# Buat folder logs jika belum ada
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/daily_pipeline_$(date +'%Y-%m-%d').log"

echo "======================================================================" | tee -a "$LOG_FILE"
echo "  [BORNEO SIAGA] MEMULAI CRON PIPELINE HARIAN" | tee -a "$LOG_FILE"
echo "  Waktu: $(date +'%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "  Python: $PYTHON_EXEC" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Jalankan script runner harian server
$PYTHON_EXEC src/prod_daily_runner.py 2>&1 | tee -a "$LOG_FILE"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[$LOG_FILE] Eksekusi pipeline harian berhasil diselesaikan." | tee -a "$LOG_FILE"
else
    echo "[ERROR] Pipeline harian mengalami kegagalan dengan kode keluar $STATUS." | tee -a "$LOG_FILE"
    exit $STATUS
fi
