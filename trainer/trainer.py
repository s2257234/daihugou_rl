"""AlphaZero 大富豪 学習トレーナ (初期版)

目的:
  - 自己対局によるデータ収集
  - リプレイバッファへの保存
  - モデルの定期学習 (train_step 呼び出し)
  - チェックポイント保存

前提:
  - 環境クラス: DaifugoSimpleEnv ( game 属性に進行状態を保持 )
  - エージェント: AlphaZeroAgent (agents.drl_agent.AlphaZeroAgent)
  - モデル: PolicyValueNet (agents.models.PolicyValueNet)

制約:
  - 現在 後続で損失計算 / 逆伝播を実装する。
  - 複数エージェント(4人)分のデータ管理は簡易。各 AlphaZeroAgent が自分のバッファを内部保持する。

使い方(例):
  from agents.drl_agent import AlphaZeroAgent
  from agents.models import PolicyValueNet
  from agents.config import ALPHA_ZERO_CONFIG
  from game.environment import DaifugoSimpleEnv

  trainer = Trainer(config=ALPHA_ZERO_CONFIG)
  trainer.setup()
  trainer.self_play(num_episodes=10)
  trainer.train_updates(num_updates=5)

TODO:
  - value ターゲット: 『次に上がるプレイヤー』フェーズ分類 (最終順位報酬なし)
  - 複数プレイヤー視点の value -> root 視点整合
  - 温度スケジューリング (中盤以降 temperature -> 0)
  - ログ収集 / TensorBoard 対応
"""
from __future__ import annotations

import os
import random
from typing import Dict, Any, List
import argparse

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.drl_agent import AlphaZeroAgent
from agents.replay_buffer import ReplayBuffer
from agents.models import PolicyValueNet
from agents.config import ALPHA_ZERO_CONFIG
from utils.logger import TrainingLogger

try:
    from game.environment import DaifugoSimpleEnv
except ImportError as e:  # pragma: no cover
    raise ImportError("DaifugoSimpleEnv が見つかりません。game.environment を確認してください") from e


class Trainer:
    def __init__(self, config: Dict[str, Any] | None = None):
        # 設定読み込み
        self.config = dict(ALPHA_ZERO_CONFIG)
        if config:
            self.config.update(config)
        # メンバ初期化
        self.agents: List[AlphaZeroAgent] = []
        self.env = None
        self.model = None
        random.seed(self.config.get("seed", 42))
        # ウォームアップ設定: 最初の n エピソードは他プレイヤーをランダム/ルールベースにして多様な盤面を収集
        self.warmup_episodes = self.config.get("warmup_episodes", 0)
        self.warmup_mix = self.config.get("warmup_mix", ["random", "rule", "random"])  # 学習エージェント以外の順番
        # 学習対象プレイヤーID (単純化: 0固定) 今後拡張可
        self.learning_player_id = self.config.get("learning_player_id", 0)
    # ロガー (setup で初期化)
        self.logger = None  # type: ignore  # TrainingLogger インスタンスは setup で設定

    # -----------------------------------------------------
    # 準備
    # -----------------------------------------------------
    def setup(self):
        # デバイス決定とモデル生成
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            try:
                import torch  # lazy import to check availability
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        else:
            resolved_device = device_cfg
        self.model = PolicyValueNet(
            max_policy_size=self.config["max_policy_size"],
            hidden_size=self.config["hidden_size"],
            num_players=self.config["num_players"],
            device=resolved_device,
        )
        # ロガー生成
        self.logger = TrainingLogger(
            log_dir=self.config.get("log_dir", "logs"),
            use_tensorboard=self.config.get("enable_tensorboard", True),
            clear_existing=self.config.get("clear_logs_on_start", False)
        )
        # 共有リプレイバッファ (use_shared_replay=true の場合)
        self.shared_replay = None
        if self.config.get("use_shared_replay", False):
            self.shared_replay = ReplayBuffer(
                maxlen=self.config.get("buffer_size", 50000),
                path=self.config.get("replay_path", "replay_buffer.joblib")
            )
        # 初期は全員学習エージェント (ウォームアップで差し替える)
        self.agents = []
        for i in range(self.config["num_players"]):
            ag = AlphaZeroAgent(player_id=i, model=self.model, config=self.config)
            # 共有モードならリプレイ参照注入
            if self.shared_replay is not None:
                ag.replay_buffer = self.shared_replay
            self.agents.append(ag)
        # 環境生成 (内部の env.agents も後で差し替える可能性があるため agent_classes=None)
        self.env = DaifugoSimpleEnv(num_players=self.config["num_players"], agent_classes=None)
        # 直ちに環境の agents を学習エージェント群で上書き (ウォームアップ0の場合の不整合防止)
        self.env.agents = self.agents
        # 各エージェントへ環境参照を渡す (obs だけ渡される呼び出し互換のため)
        for ag in self.agents:
            if hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
            # 後方互換: エージェントが logger を受け取れるなら設定
            if hasattr(ag, 'logger'):
                ag.logger = self.logger
            # 念のため共有リプレイ再注入 (ウォームアップ戻し時等)
            if self.shared_replay is not None:
                ag.replay_buffer = self.shared_replay
        # 既存リプレイのロード (継続学習対応)
        if self.shared_replay is not None:
            replay_path = self.config.get("replay_path", "replay_buffer.joblib")
            if os.path.exists(replay_path):
                try:
                    loaded = ReplayBuffer.load(replay_path)
                    # load は新インスタンスを返すので差し替え
                    self.shared_replay = loaded
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent):
                            ag.replay_buffer = self.shared_replay
                    if self.logger:
                        self.logger.log_text(f"[replay] loaded existing shared buffer size={len(self.shared_replay)}")
                    else:
                        print(f"[INFO] loaded replay size={len(self.shared_replay)}")
                except Exception as e:
                    print(f"[WARN] replay load failed: {e}")

    # ウォームアップ用: 他プレイヤーを簡易エージェントに差し替え
    def _apply_warmup_opponents(self):
        from agents.random_agent import RandomAgent
        from agents.rule_based_agent import RuleBasedAgent
        new_list = []
        for i in range(self.config["num_players"]):
            if i == self.learning_player_id:
                # 学習対象プレイヤーは AlphaZero のまま
                new_list.append(self.agents[i])
            else:
                spec = None
                if self.warmup_mix:
                    spec = self.warmup_mix[(i - (1 if i > self.learning_player_id else 0)) % len(self.warmup_mix)]
                if spec == "rule":
                    new_list.append(RuleBasedAgent(player_id=i))
                else:
                    new_list.append(RandomAgent(player_id=i))
        # トレーナ保持リスト置換
        self.agents = new_list
        # 環境内の agents も同期
        self.env.agents = self.agents
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
                if self.shared_replay is not None:
                    ag.replay_buffer = self.shared_replay

    def _restore_learning_agents(self):
        # すべて AlphaZeroAgent に戻す (学習対象以外を新規インスタンスにしてもよいが、簡単のため再生成)
        if not any(isinstance(a, AlphaZeroAgent) for a in self.agents if a.player_id == self.learning_player_id):
            # 念のため学習プレイヤーが消えていたら復元
            self.agents[self.learning_player_id] = AlphaZeroAgent(player_id=self.learning_player_id, model=self.model, config=self.config)
        for i in range(self.config["num_players"]):
            if i == self.learning_player_id:
                continue
            if not isinstance(self.agents[i], AlphaZeroAgent):
                self.agents[i] = AlphaZeroAgent(player_id=i, model=self.model, config=self.config)
        self.env.agents = self.agents
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
                if self.shared_replay is not None:
                    ag.replay_buffer = self.shared_replay

    # -----------------------------------------------------
    # 自己対局 (データ収集)
    # -----------------------------------------------------
    def self_play(self, num_episodes: int = 1):
        for ep in range(num_episodes):
            # ゲーム開始時にエポック（エピソード）進行を表示
            print(f"[EPOCH] {ep+1}/{num_episodes}")
            # ウォームアップ期間中は対戦相手をランダム/ルールベースに
            if ep < self.warmup_episodes:
                self._apply_warmup_opponents()
            elif ep == self.warmup_episodes:
                # ウォームアップ直後に全員学習エージェントへ戻す
                self._restore_learning_agents()
            self._play_one_episode(ep)

    def _play_one_episode(self, episode_index: int):
        # 環境を初期化 (DaifugoSimpleEnv が reset を持つ想定)
        if hasattr(self.env, "reset"):
            self.env.reset()
        # エピソード内フェーズ勝利カウント (学習プレイヤー視点)
        phase_wins = 0
        phase_attempts = 0
        # 各学習エージェントの move カウンタをリセット
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        # ゲーム進行ループ (env.game.done を監視)
        step_count = 0
        prev_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
        max_steps = self.config.get("max_episode_steps", 1000)
        while not getattr(self.env.game, "done", False):
            if step_count >= max_steps:
                print(f"[WARN] episode_index={episode_index} step_limit_reached={max_steps} -> force_terminate")
                break
            current_player_id = self.env.game.turn
            agent = self.agents[current_player_id]
            # 学習エージェントと簡易エージェントで呼び出し方法を分岐
            if isinstance(agent, AlphaZeroAgent):
                action = agent.select_action(self.env, training=True)
            else:
                # 観測と合法手を取得しシンプルエージェントへ渡す
                current_player = self.env.game.players[self.env.game.turn]
                hand = current_player.hand
                field = self.env.game.current_field[:]
                legal_actions = self.env._generate_legal_actions(hand, field)
                obs_simple = {'hand': hand, 'field': field}
                action = agent.select_action(obs_simple, legal_actions=legal_actions)
            # 行動適用
            # パス表現は None に統一 (環境側 step で None を直接パス処理できるようになった)
            ext_act = action  # action が None ならそのままパス
            try:
                self.env.step(external_action=ext_act)
            except TypeError:
                self.env.step(ext_act)
            step_count += 1
            # フェーズ(誰かが新たに上がった)検知
            current_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                # 新規に上がったプレイヤー(複数同時も許容)
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent):
                            was_active = ag.player_id not in prev_rankings  # 以前まだ上がっていなかったか
                            # 学習プレイヤー視点のフェーズ統計更新
                            if ag.player_id == self.learning_player_id and was_active:
                                phase_attempts += 1
                                if winner_id == self.learning_player_id:
                                    phase_wins += 1
                            ag.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                prev_rankings = current_rankings
        # 念のため未確定フェーズを 0 でクリア
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent):
                ag.flush_unfinished_phase()
    # ステップ上限で打ち切られた場合も flush 済なのでそのまま終了
        # 最終順位ベース報酬は付与しない方針 (finalize_game は no-op)
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent):
                ag.finalize_game()
        # Episode メトリクス集計
        rankings = list(getattr(self.env.game, 'rankings', []))
        episode_len = step_count
        avg_rank = None
        first_rate = 0.0
        if rankings:
            # 学習プレイヤー順位 (1-based)
            if self.learning_player_id in rankings:
                avg_rank = rankings.index(self.learning_player_id) + 1
            first_rate = 1.0 if rankings and rankings[0] == self.learning_player_id else 0.0
        # 学習プレイヤーの累積フェーズ勝率 (Agent内のカウンタ) 取得
        learner_agent = self.agents[self.learning_player_id]
        cum_phase_rate = None
        if isinstance(learner_agent, AlphaZeroAgent) and learner_agent.total_value_samples > 0:
            cum_phase_rate = learner_agent.total_positive / learner_agent.total_value_samples

        phase_win_rate = (phase_wins / phase_attempts) if phase_attempts > 0 else None
        # フェーズ予測精度 (学習プレイヤーでのみ定義)
        phase_acc = None
        if isinstance(learner_agent, AlphaZeroAgent) and learner_agent.episode_phase_total > 0:
            phase_acc = learner_agent.episode_phase_correct / learner_agent.episode_phase_total
        ep_metrics = {
            "avg_rank": avg_rank,
            "first_rate": first_rate,
            "episode_len": episode_len,
            "phase_acc": phase_acc,
            "phase_win_rate": phase_win_rate,
            "phase_wins": phase_wins,
            "phase_attempts": phase_attempts,
            "cum_phase_win_rate": cum_phase_rate,
        }
        if self.logger:
            self.logger.log_episode(ep_metrics)

    # 旧最終順位一括報酬方式は廃止 (フェーズごとの finalize_phase を利用)
    def _finalize_episode_rewards(self):  # 互換: 何もしない
        return

    # -----------------------------------------------------
    # モデル学習 (ダミー)
    # -----------------------------------------------------
    def train_updates(self, num_updates: int = 1):
        for i in range(num_updates):
            loss_info = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
            # ログ用にエポック情報を付与（存在する辞書に無害に追加）
            if isinstance(loss_info, dict):
                loss_info.setdefault("total_epochs", num_updates)
            # サンプル不足/偏り警告
            if loss_info.get("reason") == "no_data":
                print("[WARN] train_step skipped: no_data (consider increasing episodes or buffer)")
            else:
                # 共有バッファ存在時に学習プレイヤーサンプルの割合を軽くチェック
                if self.config.get("use_shared_replay", False) and self.shared_replay is not None:
                    total = len(self.shared_replay)
                    if total > 0:
                        own = sum(1 for _ in self.shared_replay.iter_all(owner_pid=self.learning_player_id))
                        ratio = own / total
                        if ratio < 0.15:  # しきい値は暫定
                            print(f"[WARN] low data share for learner: {own}/{total} ({ratio:.2%})")
            if (i + 1) % self.config.get("log_interval", 50) == 0:
                print(f"[TRAIN] epoch={i+1}/{num_updates} loss={loss_info}")
            # ロガーへ (train_step 内で既に push されている場合は二重記録を避ける)
            if self.logger and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                self.logger.log_train(loss_info)
        self._save_checkpoint()

    # -----------------------------------------------------
    # チェックポイント
    # -----------------------------------------------------
    def _save_checkpoint(self):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        path = self.config["checkpoint_path"]
        if self.model is not None:
            self.model.save(path)
        # リプレイ保存: 共有モードなら共有バッファを保存、そうでなければ従来通り代表エージェント
        replay_path = self.config.get("replay_path", "replay_buffer.joblib")
        if self.shared_replay is not None:
            try:
                self.shared_replay.save(replay_path)
            except Exception as e:
                print(f"[WARN] shared replay save failed: {e}")
        else:
            self.agents[0].save_replay(replay_path)


__all__ = ["Trainer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero 大富豪 小規模学習")
    parser.add_argument("--episodes", type=int, default=3, help="自己対局エピソード数 (default: 3)")
    parser.add_argument("--updates", type=int, default=5, help="train_step 呼び出し回数 (default: 5)")
    parser.add_argument("--num-sim", type=int, default=None, help="MCTS シミュレーション数を一時的に上書き")
    parser.add_argument("--batch-size", type=int, default=None, help="train_step のバッチサイズ一時上書き")
    parser.add_argument("--log-dir", type=str, default=None, help="ログディレクトリ上書き")
    parser.add_argument("--no-tb", action="store_true", help="TensorBoard を無効化")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード上書き")
    parser.add_argument("--device", type=str, default=None, help="使用デバイスを指定 (cpu/cuda/cuda:0)。未指定はauto")
    args = parser.parse_args()

    cfg = dict(ALPHA_ZERO_CONFIG)
    if args.num_sim is not None:
        cfg["num_simulations"] = args.num_sim
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir
    if args.no_tb:
        cfg["enable_tensorboard"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    #print("[CLI] Config overrides:", {k: cfg[k] for k in ["num_simulations","batch_size","log_dir","enable_tensorboard","device"] if k in cfg})
    trainer = Trainer(config=cfg)
    trainer.setup()
    trainer.self_play(num_episodes=args.episodes)
    trainer.train_updates(num_updates=args.updates)
    print(f"[INFO] run finished episodes={args.episodes} updates={args.updates}")
    print("[INFO] checkpoints ->", cfg.get("checkpoint_path"))
    print("[INFO] replay ->", cfg.get("replay_path"))
