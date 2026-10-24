#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [ -x "/opt/miniconda3/envs/fastapi-medical-demo/bin/uvicorn" ]; then
    UVICORN_BIN="/opt/miniconda3/envs/fastapi-medical-demo/bin/uvicorn"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/uvicorn" ]; then
    UVICORN_BIN="$VIRTUAL_ENV/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
    UVICORN_BIN="$(command -v uvicorn)"
elif python3 -m uvicorn --version >/dev/null 2>&1; then
    UVICORN_BIN="python3 -m uvicorn"
else
    echo "Error: uvicorn not found. Activate your virtual environment or install requirements.txt." >&2
    exit 1
fi

echo "Starting MedVisionAI server on http://${HOST}:${PORT}"
exec $UVICORN_BIN backend.main:app --host "$HOST" --port "$PORT"
