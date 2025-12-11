from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List


def list_selfplay_files(data_dir: str, pattern: str) -> List[str]:
    # Collect only files that match the pattern and exist as files
    paths = [p for p in glob.glob(os.path.join(data_dir, pattern)) if os.path.isfile(p) and '.tmp.' not in os.path.basename(p)]
    # Sort by modified time ascending (oldest first)
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths


def prune_data_dir(data_dir: str, pattern: str, max_files: int) -> None:
    if max_files <= 0:
        print(f"[prune-data] disabled (max_files={max_files})", flush=True)
        return

    files = list_selfplay_files(data_dir, pattern)
    total = len(files)
    if total <= max_files:
        print(f"[prune-data] nothing to do total={total} max_files={max_files}", flush=True)
        return

    to_remove = total - max_files
    removed = 0
    removed_bytes = 0

    for i in range(to_remove):
        f = files[i]
        try:
            size = os.path.getsize(f)
        except OSError:
            size = 0
        try:
            os.remove(f)
            removed += 1
            removed_bytes += size
            print(f"[prune-data] removed {os.path.basename(f)} size={size}", flush=True)
        except Exception as e:
            print(f"[prune-data] failed to remove {f}: {e}", file=sys.stderr, flush=True)

    kept = total - removed
    print(
        f"[prune-data] done total={total} kept={kept} removed={removed} removed_bytes={removed_bytes}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Prune old self-play files by FIFO (mtime)")
    ap.add_argument("--data-dir", required=True, help="data directory path")
    ap.add_argument("--max-files", type=int, default=50, help="maximum files to keep (FIFO)")
    ap.add_argument(
        "--pattern",
        default="selfplay_ep*.joblib",
        help="glob pattern to match self-play files (default: selfplay_ep*.joblib)",
    )
    args = ap.parse_args()

    try:
        prune_data_dir(args.data_dir, args.pattern, int(args.max_files))
    except Exception as e:
        print(f"[prune-data] error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
