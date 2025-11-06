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
from matplotlib import rcParams

# 日本語フォント対応（Windows想定のフォントを優先）
rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "Noto Sans CJK JP",
    "IPAGothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "DejaVu Sans",
]


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
    agent_csv = os.path.join(elo_dir, "agent_winrates.csv")
    episodes, win_rates, avg_ranks, ratings = load_eval_metrics(metrics_csv)
    ratings_hist = load_ratings_history(ratings_csv)
    # 各エージェントの最終順位分布を読み込み（各エージェントの最新行を採用）
    agent_final_dist = {}  # agent -> {rank1_rate,rank2_rate,rank3_rate,rank4_rate, win_rate}
    if os.path.exists(agent_csv):
        import csv as _csv
        last_row_by_agent = {}
        with open(agent_csv, "r", encoding="utf-8") as f:
            rdr = _csv.DictReader(f)
            for row in rdr:
                agent = row.get("agent")
                if not agent:
                    continue
                try:
                    ep = int(row.get("episode", "0") or 0)
                except Exception:
                    ep = 0
                # 常に最新 episode で置換
                prev = last_row_by_agent.get(agent)
                if (prev is None) or (ep >= prev[0]):
                    last_row_by_agent[agent] = (ep, row)
        for agent, (_ep, row) in last_row_by_agent.items():
            try:
                r1 = float(row.get("rank1_rate", row.get("win_rate", 0.0)))
                r2 = float(row.get("rank2_rate", 0.0))
                r3 = float(row.get("rank3_rate", 0.0))
                r4 = float(row.get("rank4_rate", 0.0))
                wr = float(row.get("win_rate", r1))
            except Exception:
                r1=r2=r3=r4=wr=0.0
            agent_final_dist[agent] = {
                "rank1_rate": r1,
                "rank2_rate": r2,
                "rank3_rate": r3,
                "rank4_rate": r4,
                "win_rate": wr,
            }

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

    if agent_final_dist:
        # 最終モデルの強さ（順位分布）を積み上げ棒で可視化
        agents = sorted(agent_final_dist.keys())
        r1 = [agent_final_dist[a]["rank1_rate"] for a in agents]
        r2 = [agent_final_dist[a]["rank2_rate"] for a in agents]
        r3 = [agent_final_dist[a]["rank3_rate"] for a in agents]
        r4 = [agent_final_dist[a]["rank4_rate"] for a in agents]
        import numpy as _np
        x = _np.arange(len(agents))
        fig, ax = plt.subplots(figsize=(max(7, len(agents)*0.9), 5))
        b1 = ax.bar(x, r1, label="1位率", color="#2ca02c")
        b2 = ax.bar(x, r2, bottom=r1, label="2位率", color="#1f77b4")
        b3 = ax.bar(x, r3, bottom=[a+b for a,b in zip(r1,r2)], label="3位率", color="#ff7f0e")
        b4 = ax.bar(x, r4, bottom=[a+b+c for a,b,c in zip(r1,r2,r3)], label="4位率", color="#d62728")
        ax.set_xticks(x, agents, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("割合")
        ax.set_title("最終モデルの強さ: 順位分布（積み上げ）")
        ax.legend(loc="upper right", ncol=4)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "agent_rank_distribution.png"))
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
