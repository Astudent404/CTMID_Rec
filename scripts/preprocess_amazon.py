"""Generic Amazon Reviews preprocessing CLI for CTMID.

Examples
--------
    python scripts/preprocess_amazon.py --config configs/all_beauty_ctmid.yaml
    python scripts/preprocess_amazon.py --config configs/toys_and_games_ctmid.yaml
    python scripts/preprocess_amazon.py --config configs/home_and_kitchen_ctmid.yaml --num-workers 32
    python scripts/preprocess_amazon.py --config configs/all_beauty_ctmid.yaml --limit 20000 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.data.amazon_preprocess import preprocess_amazon
from ctmid.utils.config import load_merged_config, resolve_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and slice an Amazon Reviews dataset for CTMID.")
    parser.add_argument("--config", action="append", default=[], required=True,
                        help="YAML config path. Can be specified multiple times (later files override earlier).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Read only the first N lines from each raw file (smoke testing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse, clean and report stats without writing processed files.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Worker processes for JSON parsing. Default: min(32, cpu_count-2). 0/1 = serial.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_project_paths(load_merged_config(args.config), PROJECT_ROOT)
    stats = preprocess_amazon(config, limit=args.limit, dry_run=args.dry_run, num_workers=args.num_workers)
    print("=== stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
