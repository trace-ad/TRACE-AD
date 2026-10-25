#!/bin/bash
# Run TRACE-AD on one UCR domain:
#   DATA=/path/to/data DOM=ECG bash scripts/run_ucr.sh
# Domains (must match $DATA/segments/UCR/<DOM>):
#   ECG ABP RESP Gait NASA_Data Power_Demand EPG Air_Temperature Acceleration_Sensor_Data
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate your Python environment first if needed, e.g.:
#   source /path/to/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # optional: use a warm HF cache for offline runs
DATA="${DATA:?set DATA to your data root (containing raw/ and segments/)}"
DOM="${DOM:?set DOM=ECG (etc.)}"
echo "host=$(hostname) DOM=$DOM"
python run.py --dataset UCR \
    --data-dir "$DATA/raw/ucr" \
    --seg-dir  "$DATA/segments/UCR/$DOM" \
    --out-dir  "out/ucr_$DOM" \
    --eval
echo "===== UCR $DOM done ====="
