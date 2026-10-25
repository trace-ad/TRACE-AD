#!/bin/bash
# Run TRACE-AD on one TSB-AD-U subset:
#   DATA=/path/to/data SUB=TSB_MITDB bash scripts/run_tsb.sh
# Subsets (must match $DATA/segments/TSB/<SUB>): TSB_MITDB TSB_SVDB TSB_SED
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate your Python environment first if needed, e.g.:
#   source /path/to/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # optional: use a warm HF cache for offline runs
DATA="${DATA:?set DATA to your data root (containing raw/ and segments/)}"
SUB="${SUB:?set SUB=TSB_MITDB (etc.)}"
# Optional: restrict to a space-separated series list, e.g. SERIES="219 222 223 224"
EXTRA=""; OUTSUF=""
if [ -n "${SERIES:-}" ]; then EXTRA="--series ${SERIES}"; OUTSUF="_sel"; fi
echo "host=$(hostname) SUB=$SUB SERIES=${SERIES:-all}"
python run.py --dataset TSB \
    --data-dir "$DATA/raw/tsb" \
    --seg-dir  "$DATA/segments/TSB/$SUB" \
    --out-dir  "out/${SUB}${OUTSUF}" \
    --eval $EXTRA
echo "===== TSB $SUB done ====="
