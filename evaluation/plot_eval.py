"""評価結果可視化スクリプト

eval_metrics.csv (episode,win_rate,avg_rank,rating_p0,raw_rank,steps) と
ratings.csv (timestamp,game_index,player,rating) を読み取り PNG グラフを出力。

使用例:
  python -m evaluation.plot_eval --elo-dir logs/elo --out-dir logs/elo/figs
"""
from __future__ import annotations

import argparse
import os
import csv
from typing import List, Dict

import matplotlib.pyplot as plt


def load_eval_metrics(path: str):
    episodes = []
    win_rates = []
    avg_ranks = []
    ratings = []
    if not os.path.exists(path):
        return episodes, win_rates, avg_ranks, ratings
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(row["episode"]))
                win_rates.append(float(row["win_rate"]))
                avg_ranks.append(float(row["avg_rank"]))
                ratings.append(float(row["rating_p0"]))
            except Exception:
                continue
    return episodes, win_rates, avg_ranks, ratings


def load_ratings_history(path: str):
    # returns history[player] = [(game_index, rating_float), ...] sorted
    history: Dict[str, List[tuple]] = {}
    if not os.path.exists(path):
        return history
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gi = int(row["game_index"])  # 1-based 累積
                player = row["player"]
                rating = float(row["rating"])
                history.setdefault(player, []).append((gi, rating))
            except Exception:
                pass
    for p in history:
        history[p].sort(key=lambda x: x[0])
    return history


def plot_eval(elo_dir: str, out_dir: str, show: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(elo_dir, "eval_metrics.csv")
    ratings_csv = os.path.join(elo_dir, "ratings.csv")
    episodes, win_rates, avg_ranks, ratings = load_eval_metrics(metrics_csv)
    ratings_hist = load_ratings_history(ratings_csv)

    if episodes:
        # 勝率と平均順位
        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(episodes, win_rates, label="win_rate", color="tab:blue")
        ax1.set_ylabel("Win Rate", color="tab:blue")
        ax1.set_xlabel("Episode")
        ax1.set_ylim(0, 1.0)
        ax2 = ax1.twinx()
        ax2.plot(episodes, avg_ranks, label="avg_rank", color="tab:orange")
        ax2.set_ylabel("Avg Rank", color="tab:orange")
        ax2.set_ylim(1, 4)
        ax1.set_title("Evaluation Progress (P0)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "eval_progress.png"))
        if show:
            plt.show()
        plt.close(fig)

        # レーティング(P0)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(episodes, ratings, color="tab:green")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Elo Rating (P0)")
        ax.set_title("P0 Rating Over Episodes")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "p0_rating.png"))
        if show:
            plt.show()
        plt.close(fig)

    if ratings_hist:
        fig, ax = plt.subplots(figsize=(7,4))
        for player, seq in sorted(ratings_hist.items()):
            xs = [g for g, _ in seq]
            ys = [r for _, r in seq]
            ax.plot(xs, ys, label=player)
        ax.set_xlabel("Game Index")
        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo Ratings History")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "elo_history.png"))
        if show:
            plt.show()
        plt.close(fig)

    print(f"[PLOT] generated figures in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="評価結果可視化")
    parser.add_argument("--elo-dir", type=str, default="logs/elo", help="Elo 保存ディレクトリ")
    parser.add_argument("--out-dir", type=str, default=None, help="出力先 (省略時 elo-dir/figs)")
    parser.add_argument("--show", action="store_true", help="matplotlib で表示も行う")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.elo_dir, "figs")
    plot_eval(args.elo_dir, out_dir, show=args.show)


if __name__ == "__main__":
    main()
