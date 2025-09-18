"""評価用対戦実行モジュール

目的:
  - 学習済み PolicyValueNet + AlphaZeroAgent を指定 checkpoint から読み込んで評価
  - ベースライン (RandomAgent / RuleBasedAgent) と 4人対戦を複数エピソード実行
  - 各エピソードの最終順位を Elo RatingManager に渡してレーティング更新
  - 評価過程の簡易統計 (勝率, 平均順位) を表示

想定利用:
  from evaluation.evaluator import Evaluator
  ev = Evaluator(checkpoint_path="checkpoints/policy_value_latest.pt")
  ev.run(num_episodes=20)

設計メモ:
  - 学習コード(trainer)に依存しないよう最小限の import のみ。
  - AlphaZeroAgent の config は agents.config.ALPHA_ZERO_CONFIG をロードし
    必要最小限 (デバイス, num_simulations) を override 可。
  - 評価では探索コストを抑えるため num_simulations を CLI から小さめ指定できるようにする想定。
"""
from __future__ import annotations

import os
import random
from typing import List, Dict, Any

from agents.models import PolicyValueNet
from agents.drl_agent import AlphaZeroAgent
from agents.config import ALPHA_ZERO_CONFIG
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from game.environment import DaifugoSimpleEnv
from evaluation.rating import RatingManager, EloConfig

class Evaluator:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/policy_value_latest.pt",
        device: str | None = None,
        num_simulations: int | None = None,
        elo_dir: str = "logs/elo",
        seed: int | None = 123,
        baseline_mix: List[str] | None = None,
        metrics_csv: str | None = None,
        use_tensorboard: bool = True,
    ):
        if seed is not None:
            random.seed(seed)
        self.checkpoint_path = checkpoint_path
        self.device = device or self._auto_device()
        self.num_simulations = num_simulations or ALPHA_ZERO_CONFIG.get("num_simulations", 32)
        self.elo = RatingManager(save_dir=elo_dir, config=EloConfig())
        # baseline_mix 例: ["rule", "random", "random"] -> 学習エージェント + 3 baseline
        self.baseline_mix = baseline_mix or ["rule", "random", "random"]
        self.config = dict(ALPHA_ZERO_CONFIG)
        self.config["num_simulations"] = self.num_simulations
        self.config["device"] = self.device
        # メトリクス CSV 設定
        self.metrics_csv = metrics_csv or os.path.join(elo_dir, "eval_metrics.csv")
        os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
        if not os.path.exists(self.metrics_csv):
            try:
                with open(self.metrics_csv, "w", encoding="utf-8") as f:
                    f.write("episode,win_rate,avg_rank,rating_p0,raw_rank,steps\n")
            except Exception:
                pass
        # TensorBoard (任意)
        self.tb_writer = None
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self.tb_writer = SummaryWriter(log_dir=os.path.join(elo_dir, "tb_eval"))
            except Exception as e:
                print(f"[WARN] TensorBoard writer init failed: {e}")
        # モデル読み込み
        self.model = PolicyValueNet(
            max_policy_size=self.config["max_policy_size"],
            hidden_size=self.config["hidden_size"],
            num_players=self.config["num_players"],
            device=self.device,
        )
        if os.path.exists(self.checkpoint_path):
            try:
                self.model.load(self.checkpoint_path)
            except Exception as e:
                print(f"[WARN] checkpoint load failed: {e}")
        else:
            print(f"[WARN] checkpoint not found: {self.checkpoint_path}. Using random initialized model.")
        # 評価用 AlphaZeroAgent (player_id=0 固定)
        self.eval_agent = AlphaZeroAgent(player_id=0, model=self.model, config=self.config)

    def _auto_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _make_baseline_agent(self, spec: str, player_id: int):
        if spec == "rule":
            return RuleBasedAgent(player_id=player_id)
        return RandomAgent(player_id=player_id)

    def _build_agents(self) -> List[Any]:
        agents = [self.eval_agent]
        # baseline_mix の長さに合わせて 3 名を生成 (足りなければ random で埋める)
        mix = list(self.baseline_mix)
        while len(mix) < 3:
            mix.append("random")
        for i, spec in enumerate(mix[:3], start=1):
            agents.append(self._make_baseline_agent(spec, player_id=i))
        return agents

    def play_one_game(self) -> List[int]:
        env = DaifugoSimpleEnv(num_players=4, agent_classes=None)
        # 環境の agents を上書き
        env.agents = self._build_agents()
        # 学習エージェントに環境参照 (MCTS 内で利用する可能性)
        if hasattr(self.eval_agent, 'set_env_ref'):
            self.eval_agent.set_env_ref(env)
        # リセット
        env.reset()
        # 進行
        step_limit = 1000
        steps = 0
        prev_rankings: List[int] = list(getattr(env.game, 'rankings', []))
        while not getattr(env.game, 'done', False):
            if steps >= step_limit:
                print(f"[WARN] step limit reached ({step_limit}) forcing termination")
                break
            current_player_id = env.game.turn
            agent = env.agents[current_player_id]
            if isinstance(agent, AlphaZeroAgent):
                action = agent.select_action(env, training=False)
            else:
                # baseline: シンプル観測
                current_player = env.game.players[current_player_id]
                hand = current_player.hand
                field = env.game.current_field[:]
                legal_actions = env._generate_legal_actions(hand, field)
                obs_simple = {'hand': hand, 'field': field}
                action = agent.select_action(obs_simple, legal_actions=legal_actions)
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            steps += 1
        rankings: List[int] = list(getattr(env.game, 'rankings', []))
        if len(rankings) != 4:
            # 強制終了時など順位未確定は残りをランダム末尾扱い
            remaining = [i for i in range(4) if i not in rankings]
            rankings += remaining
        return rankings

    def run(self, num_episodes: int = 10):
        win_counts = {0: 0}
        rank_sum = {0: 0}
        for ep in range(num_episodes):
            rankings = self.play_one_game()
            steps = None  # 現在 step カウントは play_one_game 内部で未返却なので拡張余地
            # Elo 更新 (player 名は pid の文字列化で区別) 順位リストを文字列へ変換
            ranking_names = [f"P{pid}" for pid in rankings]
            self.elo.update_from_rankings(ranking_names)
            # 集計 (評価対象=player0)
            if rankings[0] == 0:
                win_counts[0] += 1
            rank_sum[0] += rankings.index(0) + 1
            avg_rank = rank_sum[0] / (ep + 1)
            win_rate = win_counts[0] / (ep + 1)
            r = self.elo.get_rating("P0")
            # CSV 追記
            try:
                with open(self.metrics_csv, "a", encoding="utf-8") as f:
                    f.write(f"{ep+1},{win_rate:.6f},{avg_rank:.6f},{r:.2f},{'|'.join(map(str, rankings))},{'' if steps is None else steps}\n")
            except Exception:
                pass
            # TensorBoard 出力
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("eval/win_rate", win_rate, ep + 1)
                self.tb_writer.add_scalar("eval/avg_rank", avg_rank, ep + 1)
                self.tb_writer.add_scalar("eval/rating_p0", r, ep + 1)
            if (ep + 1) % 5 == 0 or ep == num_episodes - 1:
                print(f"[EVAL] ep={ep+1}/{num_episodes} win_rate={win_rate:.2%} avg_rank={avg_rank:.2f} rating(P0)={r:.1f}")
        # 最終リーダーボード
        board = self.elo.get_leaderboard()
        print("[EVAL] Leaderboard (top):")
        for name, rating in board[:10]:
            print(f"  {name}: {rating:.1f}")
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

__all__ = ["Evaluator"]
