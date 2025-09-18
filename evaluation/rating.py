"""Elo レーティング計算ユーティリティ

多人数(4人)ゲーム用の簡易 Elo 実装。
アプローチ:
 1. 1ゲーム終了後の最終順位リスト (rankings; index=順位-1, 値=player_id) を入力とする。
 2. 全ての (i,j) ペアについて、i の順位 < j の順位 なら i 勝利とみなし 1.0, 逆は 0.0 (同順位は無視: 本ゲームでは発生しない想定)。
 3. 各ペアで通常の Elo 期待値 / 更新式を適用し両者のレーティングを更新。
 4. 複数ペアを同一ゲーム内で順序依存を減らすため、各プレイヤー毎に Δ を蓄積しゲーム終了後まとめて加算。

式:
  E_A = 1 / (1 + 10^{(R_B - R_A)/400})
  ΔA += K * (S_A - E_A)

K-factor は初期デフォルト 32。必要に応じて調整可能。

永続化:
  RatingManager は ratings.json (player_id -> rating) と ratings.csv (履歴追記: timestamp, game_index, player_id, rating) を扱う。

注意:
  - 簡易実装のためパフォーマンス最適化は行っていない。
  - 同時に複数プロセスから書き込む用途は想定していない (排他なし)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import json
import os
import csv
import time


@dataclass
class EloConfig:
    initial_rating: float = 1000.0
    k_factor: float = 32.0
    min_rating: float = 0.0  # クリッピング下限 (負値増幅防止)


@dataclass
class RatingManager:
    save_dir: str = "logs/elo"
    config: EloConfig = field(default_factory=EloConfig)
    ratings: Dict[str, float] = field(default_factory=dict)  # key: player_name
    game_index: int = 0  # 永続化された最後のゲーム番号 (CSV の行数ではない)

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._ratings_path = os.path.join(self.save_dir, "ratings.json")
        self._history_path = os.path.join(self.save_dir, "ratings.csv")
        self._load_if_exists()

    # ----------------------------
    # 永続化
    # ----------------------------
    def _load_if_exists(self):
        if os.path.exists(self._ratings_path):
            try:
                with open(self._ratings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # game_index を含む形式 {"ratings": {...}, "game_index": N} を期待
                if isinstance(data, dict) and "ratings" in data:
                    self.ratings = {k: float(v) for k, v in data.get("ratings", {}).items()}
                    self.game_index = int(data.get("game_index", 0))
                else:
                    # 後方互換: 旧形式 (player_id -> rating)
                    self.ratings = {k: float(v) for k, v in data.items()}
            except Exception:
                pass
        # 履歴CSVがなければヘッダを書く
        if not os.path.exists(self._history_path):
            with open(self._history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "game_index", "player", "rating"])

    def _save(self):
        data = {
            "ratings": self.ratings,
            "game_index": self.game_index,
        }
        with open(self._ratings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _append_history(self):
        ts = int(time.time())
        with open(self._history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for player, rating in self.ratings.items():
                writer.writerow([ts, self.game_index, player, f"{rating:.2f}"])

    # ----------------------------
    # 公開 API
    # ----------------------------
    def ensure_player(self, player: str):
        if player not in self.ratings:
            self.ratings[player] = self.config.initial_rating

    def ensure_players(self, players: Iterable[str]):
        for p in players:
            self.ensure_player(p)

    def get_rating(self, player: str) -> float:
        return self.ratings.get(player, self.config.initial_rating)

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_from_rankings(self, rankings: List[str]):
        """最終順位 (先頭=1位) のプレイヤー名リストからレーティングを更新。
        重複しない前提。本ゲームで同着は起こらない前提。
        """
        # 未登録プレイヤー初期化
        self.ensure_players(rankings)
        # プレイヤーごとの Δ 累積
        delta: Dict[str, float] = {p: 0.0 for p in rankings}
        k = self.config.k_factor
        n = len(rankings)
        # ペアワイズ比較 O(n^2)
        for i in range(n):
            for j in range(i + 1, n):
                winner = rankings[i]
                loser = rankings[j]
                ra = self.ratings[winner]
                rb = self.ratings[loser]
                ea = self.expected_score(ra, rb)
                eb = self.expected_score(rb, ra)
                # 勝者視点 S=1, 敗者視点 S=0
                delta[winner] += k * (1.0 - ea)
                delta[loser] += k * (0.0 - eb)
        # 一括適用
        for p, d in delta.items():
            new_r = self.ratings[p] + d
            if new_r < self.config.min_rating:
                new_r = self.config.min_rating
            self.ratings[p] = new_r
        # ゲームインデックス更新 & 保存
        self.game_index += 1
        self._save()
        self._append_history()

    def batch_update(self, list_of_rankings: List[List[str]]):
        for r in list_of_rankings:
            self.update_from_rankings(r)

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.ratings)

__all__ = ["EloConfig", "RatingManager"]
