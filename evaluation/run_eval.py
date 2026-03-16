"""Elo 評価 CLI

実行例:
  python -m evaluation.run_eval --episodes 20 --checkpoint checkpoints/policy_value_latest.pt --num-sim 32
"""
from __future__ import annotations

import argparse

from evaluation.evaluator import Evaluator
from evaluation.plot_eval import plot_eval
import os

_RUN_EVAL_FALLBACK_LOGGED = set()


def _log_run_eval_fallback_once(key: str, msg: str, exc: Exception | None = None) -> None:
    if key in _RUN_EVAL_FALLBACK_LOGGED:
        return
    _RUN_EVAL_FALLBACK_LOGGED.add(key)
    try:
        if exc is not None:
            print(f"{msg} ({type(exc).__name__}: {exc})")
        else:
            print(msg)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Daifugo AlphaZero 評価 (Elo)")
    parser.add_argument("--episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy_value_latest.pt", help="評価するモデルのチェックポイント（ベア名なら checkpoints/ を自動付与）")
    parser.add_argument("--device", type=str, default=None, help="使用デバイス (cpu/cuda). 省略時は自動")
    parser.add_argument("--num-sim", type=int, default=None, help="MCTS シミュレーション数 (デフォルトは config 既定値)")
    parser.add_argument("--elo-dir", type=str, default="logs/elo", help="Elo レーティング保存ディレクトリ")
    parser.add_argument("--mix", type=str, default="ucb_mcts,random,rule", help="ベースライン3枠の種類をカンマ区切り (ucb_mcts / rule / random)")
    parser.add_argument("--no-tb", action="store_true", help="TensorBoard 出力を無効化")
    parser.add_argument("--metrics-csv", type=str, default=None, help="評価メトリクスCSV(指定で上書き)")
    parser.add_argument("--no-auto-plot", action="store_true", help="評価後に自動でグラフ更新を行わない")
    parser.add_argument("--seed", type=int, default=123, help="乱数シード")
    parser.add_argument("--workers", type=int, default=1, help="並列評価ワーカー数 (1 で逐次)")
    parser.add_argument("--det-mode-eval", type=str, default=None, choices=["fixed_once","stochastic","none"], help="評価時 determinization モードを上書き")
    parser.add_argument("--no-rotate-seats", action="store_true", help="座席回転を無効化して固定座席で評価")
    parser.add_argument("--past-checkpoints", type=str, default=None, help="過去モデルのチェックポイントをカンマ区切りで指定（ベア名なら checkpoints/ を自動付与） 例: ep100.pt,ep200.pt")
    parser.add_argument("--eval-vs-random", action="store_true", help="追加評価: 学習済みモデル vs ランダム3人（別ディレクトリに出力）")
    parser.add_argument("--eval-vs-rule", action="store_true", help="追加評価: 学習済みモデル vs ルールベース3人（別ディレクトリに出力）")
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
        except Exception as e:
            _log_run_eval_fallback_once(
                "norm_ckpt_exception",
                "[eval-fallback] checkpoint path normalization failed; using original input",
                e,
            )
            return p

    # 過去モデルパスの解析
    past_ckpts = None
    if args.past_checkpoints:
        past_ckpts = [_norm_ckpt(p.strip()) for p in args.past_checkpoints.split(',') if p.strip()]

    # checkpoint の正規化
    ckpt_main = _norm_ckpt(args.checkpoint)

    def _run_eval_once(elo_dir: str, mix: list[str], metrics_csv: str | None, label: str):
        evaluator = Evaluator(
            checkpoint_path=ckpt_main,
            device=args.device,
            num_simulations=args.num_sim,
            elo_dir=elo_dir,
            seed=args.seed,
            baseline_mix=mix,
            metrics_csv=metrics_csv,
            use_tensorboard=not args.no_tb,
            seat_rotation=(not args.no_rotate_seats),
            determinization_mode_override=args.det_mode_eval,
            past_checkpoints=past_ckpts,
        )
        participants = evaluator.run(num_episodes=args.episodes, workers=args.workers)
        print(f"[EVAL] Participants ({label}):")
        for lb in participants:
            print(f"  - {lb}")
        if not args.no_auto_plot:
            out_dir = os.path.join(elo_dir, "figs")
            plot_eval(elo_dir, out_dir, show=False)
            print(f"[EVAL] plots updated -> {out_dir}")

    # 通常評価 (追加評価フラグが無い場合のみ)
    if (not args.eval_vs_random) and (not args.eval_vs_rule):
        _run_eval_once(args.elo_dir, baseline_mix, args.metrics_csv, "default")

    # 追加評価: vs random
    if args.eval_vs_random:
        elo_dir_random = os.path.join(args.elo_dir, "vs_random")
        _run_eval_once(elo_dir_random, ["random", "random", "random"], None, "vs_random")

    # 追加評価: vs rule
    if args.eval_vs_rule:
        elo_dir_rule = os.path.join(args.elo_dir, "vs_rule")
        _run_eval_once(elo_dir_rule, ["rule", "rule", "rule"], None, "vs_rule")


if __name__ == "__main__":
    main()
