"""Fast combined loop for non-parallel self-play + training.

目的:
  - バッチファイル経由で Python プロセスを 2 回起動するオーバーヘッドを削減し、
    自己対戦完了直後に学習を即開始できるようにする。
  - 既存の `non_parallel_self_play.py` / `non_parallel_trainer.py` のロジックを再利用。

挙動:
  1. self-play を実行し生成ファイルパスを取得 (run_self_play の戻り値)
  2. 直後に trainer.train_loop を呼び出し、 files_override でそのファイルのみ取り込み
  3. これを指定回数反復 (iterations)

利点:
  - Python import / JIT 初期化を 1 度に集約
  - self-play 出力ファイルの検索 (glob) を省略し即 ingestion
  - 圧縮レベル調整 (config.selfplay_joblib_compress) により I/O 時間短縮

使い方例:
  python -m non_parallel_trainer.fast_loop --episodes 20 --updates 500 --iterations 999 \
      --workers 15 --data-dir data --log-dir logs --checkpoint-dir checkpoints \
      --version-interval 0

注意:
  - pos_rate 等の追加メトリックは元スクリプトに依存 (ここでは改変しない)
  - 複数ファイル取り込みを希望する場合は --no-single-file を指定して既存挙動へフォールバック
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from non_parallel_trainer.non_parallel_self_play import run_self_play
from non_parallel_trainer.non_parallel_trainer import train_loop, _load_config, _resolve_device


def main():
    p = argparse.ArgumentParser(description="Fast combined non-parallel loop")
    p.add_argument('--episodes', type=int, default=20, help='自己対戦エピソード数 / 1ループ')
    p.add_argument('--updates', type=int, default=500, help='train_step 回数 / 1ループ')
    p.add_argument('--iterations', type=int, default=0, help='総反復回数 (0=無限)')
    p.add_argument('--workers', type=int, default=15, help='self-play ワーカー数')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--log-dir', type=str, default='logs')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--config-json', type=str, default=None)
    p.add_argument('--version-interval', type=int, default=0, help='更新間隔タグ付き ckpt (0=無効)')
    p.add_argument('--device', type=str, default=None, help='auto/cpu/cuda 指定')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--no-single-file', action='store_true', help='直近ファイルのみ取り込み最適化を無効化')
    p.add_argument('--selfplay-compress', type=int, default=None, help='自己対戦 joblib 圧縮レベル override')
    args = p.parse_args()

    # 設定ロード & 上書き
    cfg: Dict[str, Any] = _load_config(ALPHA_ZERO_CONFIG, args.config_json)
    if args.seed is not None:
        cfg['seed'] = int(args.seed)
    if args.device is not None:
        cfg['device'] = args.device
    if args.batch_size is not None:
        cfg['batch_size'] = int(args.batch_size)
    if args.selfplay_compress is not None:
        cfg['selfplay_joblib_compress'] = int(args.selfplay_compress)
    cfg['checkpoint_dir'] = args.checkpoint_dir
    cfg['checkpoint_path'] = os.path.join(args.checkpoint_dir, 'policy_value_latest.pt')
    cfg['log_dir'] = args.log_dir
    cfg['clear_logs_on_start'] = False
    cfg['device'] = _resolve_device(cfg.get('device'))
    cfg.setdefault('val_eval_every_updates', 200)
    # shared replay は trainer 内部でエフェメラル生成

    it = 0
    print(f"[FAST-LOOP] start episodes={args.episodes} updates={args.updates} workers={args.workers} iterations={'infinite' if args.iterations==0 else args.iterations}")
    try:
        while True:
            it += 1
            if args.iterations and it > args.iterations:
                break
            print(f"\n=== fast-iteration {it} ===")
            # self-play 実行 (出力ファイルパス取得)
            latest_path = run_self_play(
                cfg,
                episodes=int(args.episodes),
                workers=int(args.workers),
                model_path=cfg['checkpoint_path'],
                data_dir=args.data_dir,
                log_dir=args.log_dir,
                checkpoint_dir=args.checkpoint_dir,
            )
            # 直近ファイルのみで学習 (no-single-file 指定が無ければ最適化適用)
            files_override = None if args.no-single-file else [latest_path]
            train_loop(
                cfg,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
                max_files=None,  # files_override なので未使用
                max_samples_per_file=None,
                updates=int(args.updates),
                version_interval=int(args.version_interval),
                files_override=files_override,
            )
    except KeyboardInterrupt:
        print("[FAST-LOOP] interrupted by user")

    print(f"[FAST-LOOP] finished iterations={it- (0 if args.iterations==0 else 0)}")


if __name__ == '__main__':
    main()