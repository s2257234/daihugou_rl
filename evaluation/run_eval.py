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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy_value_latest.pt", help="評価するモデルのチェックポイント（ベア名なら checkpoints/ を自動付与）")
    parser.add_argument("--device", type=str, default=None, help="使用デバイス (cpu/cuda). 省略時は自動")
    parser.add_argument("--num-sim", type=int, default=None, help="MCTS シミュレーション数 (デフォルトは config 既定値)")
    parser.add_argument("--elo-dir", type=str, default="logs/elo", help="Elo レーティング保存ディレクトリ")
    parser.add_argument("--mix", type=str, default="rule,random,random", help="ベースライン3枠の種類をカンマ区切り (rule / random)")
    parser.add_argument("--no-tb", action="store_true", help="TensorBoard 出力を無効化")
    parser.add_argument("--metrics-csv", type=str, default=None, help="評価メトリクスCSV(指定で上書き)")
    parser.add_argument("--no-auto-plot", action="store_true", help="評価後に自動でグラフ更新を行わない")
    parser.add_argument("--seed", type=int, default=123, help="乱数シード")
    parser.add_argument("--workers", type=int, default=1, help="並列評価ワーカー数 (1 で逐次)")
    parser.add_argument("--det-mode-eval", type=str, default=None, choices=["fixed_once","stochastic","none"], help="評価時 determinization モードを上書き")
    parser.add_argument("--no-rotate-seats", action="store_true", help="座席回転を無効化して固定座席で評価")
    parser.add_argument("--past-checkpoints", type=str, default=None, help="過去モデルのチェックポイントをカンマ区切りで指定（ベア名なら checkpoints/ を自動付与） 例: ep100.pt,ep200.pt")
    # 旧重複オプションを削除
    args = parser.parse_args()

    baseline_mix = [m.strip() for m in args.mix.split(',') if m.strip()]

    # パス正規化: ベア名には "checkpoints/" を付与（絶対パスや既にディレクトリ含みはそのまま）
    import os
    def _norm_ckpt(p: str) -> str:
        if not p:
            return p
        try:
            if os.path.isabs(p):
                return p
            # 既にスラッシュやバックスラッシュを含む -> そのまま
            if ("/" in p) or ("\\" in p):
                return p
            # ベア名: checkpoints/ を付与
            return os.path.join("checkpoints", p)
        except Exception:
            return p

    # 過去モデルパスの解析
    past_ckpts = None
    if args.past_checkpoints:
        past_ckpts = [_norm_ckpt(p.strip()) for p in args.past_checkpoints.split(',') if p.strip()]

    # checkpoint の正規化
    ckpt_main = _norm_ckpt(args.checkpoint)

    evaluator = Evaluator(
    checkpoint_path=ckpt_main,
        device=args.device,
        num_simulations=args.num_sim,
        elo_dir=args.elo_dir,
        seed=args.seed,
        baseline_mix=baseline_mix,
        metrics_csv=args.metrics_csv,
        use_tensorboard=not args.no_tb,
        seat_rotation=(not args.no_rotate_seats),
        determinization_mode_override=args.det_mode_eval,
        past_checkpoints=past_ckpts,
    )
    evaluator.run(num_episodes=args.episodes, workers=args.workers)
    # 参加者ラベルを統一形式で表示（席名のマッピング表示は廃止）
    labels = evaluator.get_participant_labels()
    print("[EVAL] Participants:")
    for lb in labels:
        print(f"  - {lb}")
    if not args.no_auto_plot:
        # 自動プロット更新 (figs ディレクトリを明示)
        out_dir = os.path.join(args.elo_dir, "figs")
        plot_eval(args.elo_dir, out_dir, show=False)
        print(f"[EVAL] plots updated -> {out_dir}")


if __name__ == "__main__":
    main()
