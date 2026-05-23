from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.data.amazon_preprocess import preprocess_all_beauty
from ctmid.utils.config import load_config, resolve_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Amazon All Beauty jsonl.gz for CTMID.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "all_beauty_ctmid.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Read only the first N rows from each raw file.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report stats without writing processed files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_project_paths(load_config(args.config), PROJECT_ROOT)
    stats = preprocess_all_beauty(config, limit=args.limit, dry_run=args.dry_run)
    print(stats)


if __name__ == "__main__":
    main()
