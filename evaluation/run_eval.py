"""Elo 評価 CLI

実行例:
  python -m evaluation.run_eval --episodes 20 --checkpoint checkpoints/policy_value_latest.pt --num-sim 32
"""
from __future__ import annotations

import argparse

from evaluation.evaluator import Evaluator
from evaluation.plot_eval import plot_eval
import os


def main():
    parser = argparse.ArgumentParser(description="Daifugo AlphaZero 評価 (Elo)")
    parser.add_argument("--episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy_value_latest.pt", help="評価するモデルのチェックポイントパス")
    parser.add_argument("--device", type=str, default=None, help="使用デバイス (cpu/cuda). 省略時は自動")
    parser.add_argument("--num-sim", type=int, default=None, help="MCTS シミュレーション数 (デフォルトは config 既定値)")
    parser.add_argument("--elo-dir", type=str, default="logs/elo", help="Elo レーティング保存ディレクトリ")
    parser.add_argument("--mix", type=str, default="rule,random,random", help="ベースライン3枠の種類をカンマ区切り (rule / random)")
    parser.add_argument("--no-tb", action="store_true", help="TensorBoard 出力を無効化")
    parser.add_argument("--metrics-csv", type=str, default=None, help="評価メトリクスCSV(指定で上書き)")
    parser.add_argument("--no-auto-plot", action="store_true", help="評価後に自動でグラフ更新を行わない")
    parser.add_argument("--seed", type=int, default=123, help="乱数シード")
    args = parser.parse_args()

    baseline_mix = [m.strip() for m in args.mix.split(',') if m.strip()]

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_simulations=args.num_sim,
        elo_dir=args.elo_dir,
        seed=args.seed,
        baseline_mix=baseline_mix,
        metrics_csv=args.metrics_csv,
        use_tensorboard=not args.no_tb,
    )
    evaluator.run(num_episodes=args.episodes)
    if not args.no_auto_plot:
        # 自動プロット更新 (figs ディレクトリを明示)
        out_dir = os.path.join(args.elo_dir, "figs")
        plot_eval(args.elo_dir, out_dir, show=False)
        print(f"[EVAL] plots updated -> {out_dir}")


if __name__ == "__main__":
    main()
