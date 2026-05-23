#!/usr/bin/env bash
# Preprocess every Amazon dataset under data/ into train/valid/test splits.
# Runs sequentially to keep peak memory and disk usage bounded.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p log

CONFIGS=(
  configs/all_beauty_ctmid.yaml
  configs/toys_and_games_ctmid.yaml
  configs/home_and_kitchen_ctmid.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  echo "=== preprocessing ${name} ==="
  python3 scripts/preprocess_amazon.py --config "$cfg" "$@" 2>&1 | tee "log/preprocess_${name}.log"
  df -h . | tail -1
done

echo "=== all datasets preprocessed ==="
