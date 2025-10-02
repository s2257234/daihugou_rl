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
import time
from typing import Dict, Any, List
from dataclasses import dataclass
import multiprocessing as mp
import argparse
import threading

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


# ======================================================
# 並列自己対局: ワーカープロセス用エントリ関数 (Windows spawn 対応)
# ======================================================
def _selfplay_worker_entry(worker_id: int, episodes: int, config: Dict[str, Any], model_path: str):
    """ワーカープロセスで自己対局を複数回実行し、確定サンプルを返す。

    戻り値: {"samples": List[Dict], "episodes": int, "worker": int}
    """
    # 乱数初期化（プロセス毎にズラす）
    try:
        seed = int(config.get("seed", 42)) + 10000 * int(worker_id)
    except Exception:
        seed = 42 + 10000 * int(worker_id)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    # モデル読込（CPU 推奨） + フル特徴量対応
    from agents.models import PolicyValueNet as _PVN
    device = config.get("selfplay_worker_device", "cpu")
    use_full = bool(config.get("use_full_features", False))
    try:
        model = _PVN.load(model_path, map_location=device)
        # 旧 ckpt で use_full_features が欠落しているが config が True の場合再構築
        if use_full and not getattr(model, 'use_full_features', False):
            # 後で環境初期化後に再構築
            pass
    except Exception:
        # フォールバック: 仮モデル (フル時は base_dim で後から再構築)
        try:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
                use_full_features=use_full,
                full_feature_dim=(2 + config.get("num_players", 4)) if use_full else None,
            )
        except Exception:
            # 最低限簡易
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
            )

    # エージェントと環境を構築（共有リプレイは使わずローカルに蓄積）
    num_players = int(config.get("num_players", 4))
    agents: List[AlphaZeroAgent] = []
    for pid in range(num_players):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=dict(config))
        # ワーカー内では共有リプレイを無効化し、ローカル list に保存させる
        try:
            ag._use_shared = False
            ag.replay_buffer = []
        except Exception:
            pass
        agents.append(ag)
    env = DaifugoSimpleEnv(num_players=num_players, agent_classes=None)
    env.agents = agents
    for ag in agents:
        if hasattr(ag, 'set_env_ref'):
            ag.set_env_ref(env)
        if hasattr(ag, 'logger'):
            ag.logger = None

    # フル特徴量なら最初の reset 後に次元確定して再構築
    if use_full:
        try:
            env.reset()
            probe_state = agents[0]._extract_state(env)
            full_dim = probe_state.get('full_input_dim')
            if full_dim and (not getattr(model, 'use_full_features', False) or getattr(model, 'backbone', None) and getattr(model.backbone[0], 'in_features', None) != full_dim):
                model = _PVN(
                    max_policy_size=config.get("max_policy_size", 128),
                    hidden_size=config.get("hidden_size", 128),
                    num_players=config.get("num_players", 4),
                    device=device,
                    use_full_features=True,
                    full_feature_dim=full_dim,
                )
                for ag in agents:
                    ag.set_model(model)
        except Exception:
            print("[WARN][worker] full feature rebuild failed, fallback simple")

    # エピソード実行ループ（Trainer._play_one_episode とほぼ同等の縮約版）
    total_episodes = int(episodes)
    for ep in range(total_episodes):
        if hasattr(env, 'reset'):
            env.reset()
        for ag in agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        step_count = 0
        prev_rankings: List[int] = list(getattr(env.game, "rankings", []))
        max_steps = int(config.get("max_episode_steps", 1000))
        while not getattr(env.game, "done", False):
            if step_count >= max_steps:
                break
            cur_pid = env.game.turn
            ag = agents[cur_pid]
            action = ag.select_action(env, training=True)
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            step_count += 1
            # フェーズ確定時処理
            current_rankings: List[int] = list(getattr(env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for az in agents:
                        was_active = az.player_id not in prev_rankings
                        az.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                prev_rankings = current_rankings
        # 未確定フェーズを 0 で確定し、ゲーム終端フック
        for az in agents:
            az.flush_unfinished_phase()
            az.finalize_game()

    # 確定サンプルを収集して返す
    out_samples: List[Dict[str, Any]] = []
    for az in agents:
        try:
            for s in az.replay_buffer:
                if isinstance(s, dict) and s.get("value") is not None:
                    out_samples.append(dict(s))
        except Exception:
            pass
    return {"samples": out_samples, "episodes": total_episodes, "worker": worker_id}


# ======================================================
# 常駐自己対局: 並行学習用ワーカープロセス (Producer)
# ======================================================
def _selfplay_daemon_worker(worker_id: int,
                            config: Dict[str, Any],
                            model_path: str,
                            sample_queue,  # mp.Queue
                            event_queue,   # mp.Queue ("ep_done" 通知など)
                            stop_event):   # mp.Event
    """常駐で自己対局を繰り返し、確定サンプルを逐次 sample_queue へ push する。

    event_queue へはエピソード完了ごとに ("ep_done", 1) を送る。
    モデル更新は model_path の mtime を監視して自動リロードする（数エピソード毎にチェック）。
    """
    # 乱数初期化（プロセス毎にズラす）
    try:
        seed = int(config.get("seed", 42)) + 10000 * int(worker_id)
    except Exception:
        seed = 42 + 10000 * int(worker_id)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    # モデル読込（CPU 推奨） + フル特徴量対応
    from agents.models import PolicyValueNet as _PVN
    device = config.get("selfplay_worker_device", "cpu")
    use_full = bool(config.get("use_full_features", False))
    try:
        model = _PVN.load(model_path, map_location=device)
        if use_full and not getattr(model, 'use_full_features', False):
            pass
    except Exception:
        try:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
                use_full_features=use_full,
                full_feature_dim=(2 + config.get("num_players", 4)) if use_full else None,
            )
        except Exception:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
            )

    # モデル更新監視
    def _get_mtime(p):
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0
    last_mtime = _get_mtime(model_path)
    # 過去モデルプール（親が保存したスナップショット）
    pool_dir = os.path.join(config.get("checkpoint_dir", "checkpoints"), "_pool")
    learning_pid = int(config.get("learning_player_id", 0) or 0)
    enable_mix = bool(config.get("keep_previous_model_opponent", False)) and int(config.get("previous_model_mix_players", 0) or 0) > 0
    mix_players = int(config.get("previous_model_mix_players", 0) or 0)
    mix_interval = int(config.get("opponent_mix_interval_episodes", 0) or 0)
    ep_since_mix = 0

    def _list_pool_files():
        try:
            if not os.path.isdir(pool_dir):
                return []
            files = [os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.endswith('.pt')]
            # 新しい順（将来のポリシーで優先したい場合）だがランダムに選ぶので順序は重要ではない
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files
        except Exception:
            return []

    def _assign_opponents_from_pool():
        """学習プレイヤー以外の一部に過去モデルを割当てる。プールが空なら全員最新で継続。"""
        try:
            # 学習プレイヤーは常に最新モデル
            try:
                ag_learner = agents[learning_pid]
                if hasattr(ag_learner, 'set_model'):
                    ag_learner.set_model(model)
            except Exception:
                pass

            if not enable_mix or mix_players <= 0:
                return
            pool_files = _list_pool_files()
            if not pool_files:
                return
            # 候補プレイヤー
            cand = [i for i in range(num_players) if i != learning_pid]
            if not cand:
                return
            random.shuffle(cand)
            selected = cand[:mix_players]
            # まず全員を最新に戻す（選ばれた者だけ後で過去モデルを当てる）
            for i in cand:
                try:
                    if hasattr(agents[i], 'set_model'):
                        agents[i].set_model(model)
                except Exception:
                    pass
            # 選ばれた相手にプールからランダム割当
            for idx in selected:
                try:
                    path = random.choice(pool_files)
                    opp_model = _PVN.load(path, map_location=device)
                    if hasattr(agents[idx], 'set_model'):
                        agents[idx].set_model(opp_model)
                except Exception:
                    # 読込失敗時は最新モデルのまま
                    pass
        except Exception:
            pass

    # エージェントと環境を構築（共有バッファは使用せず、サンプルは Queue へ）
    num_players = int(config.get("num_players", 4))
    agents: List[AlphaZeroAgent] = []
    for pid in range(num_players):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=dict(config))
        # ワーカー内では共有リプレイを使わない
        try:
            ag._use_shared = False
            ag.replay_buffer = []
            ag.logger = None
        except Exception:
            pass
        agents.append(ag)
    env = DaifugoSimpleEnv(num_players=num_players, agent_classes=None)
    env.agents = agents
    for ag in agents:
        if hasattr(ag, 'set_env_ref'):
            ag.set_env_ref(env)
        if hasattr(ag, 'logger'):
            ag.logger = None

    max_steps = int(config.get("max_episode_steps", 1000))
    check_interval_episodes = int(config.get("concurrent_model_check_every", 5) or 5)
    ep_since_check = 0

    # 初期の対戦相手割当（プールがあれば活用）
    if enable_mix and mix_players > 0:
        _assign_opponents_from_pool()

    # フル特徴量: 初期 reset で次元確定し必要なら再構築
    if use_full:
        try:
            env.reset()
            probe_state = agents[0]._extract_state(env)
            full_dim = probe_state.get('full_input_dim')
            if full_dim and (not getattr(model, 'use_full_features', False) or getattr(model, 'backbone', None) and getattr(model.backbone[0], 'in_features', None) != full_dim):
                model = _PVN(
                    max_policy_size=config.get("max_policy_size", 128),
                    hidden_size=config.get("hidden_size", 128),
                    num_players=config.get("num_players", 4),
                    device=device,
                    use_full_features=True,
                    full_feature_dim=full_dim,
                )
                for ag in agents:
                    ag.set_model(model)
        except Exception:
            print("[WARN][daemon_worker] full feature rebuild failed, fallback simple")

    while not (stop_event.is_set()):
        # 必要ならモデル更新
        if ep_since_check >= check_interval_episodes:
            ep_since_check = 0
            cur_mtime = _get_mtime(model_path)
            if cur_mtime > last_mtime:
                try:
                    model = _PVN.load(model_path, map_location=device)
                    # 学習プレイヤーのみ最新へ（対戦相手の過去モデル割当は維持）
                    try:
                        ag_learner = agents[learning_pid]
                        if hasattr(ag_learner, 'set_model'):
                            ag_learner.set_model(model)
                    except Exception:
                        pass
                except Exception:
                    pass
                last_mtime = cur_mtime
                # モデル更新後に、必要なら対戦相手の再割当を行う（次のミックスタイミングで行う）
                # 即時適用したい場合は以下を有効化
                # if enable_mix and mix_players > 0:
                #     _assign_opponents_from_pool()

        # 1 エピソード実行
        if hasattr(env, 'reset'):
            env.reset()
        for ag in agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        step_count = 0
        prev_rankings: List[int] = list(getattr(env.game, "rankings", []))
        dbg_samples_before = [len(getattr(ag, 'replay_buffer', [])) for ag in agents if hasattr(ag, 'replay_buffer')]
        while not getattr(env.game, "done", False):
            if step_count >= max_steps:
                break
            cur_pid = env.game.turn
            ag = agents[cur_pid]
            action = ag.select_action(env, training=True)
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            step_count += 1
            # フェーズ確定処理
            current_rankings: List[int] = list(getattr(env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for az in agents:
                        was_active = az.player_id not in prev_rankings
                        az.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                try:
                    if config.get('debug_concurrent_worker', False):
                        # 勝者発生時点で現在のフェーズ確定サンプル数をログ
                        labeled_counts = []
                        total_counts = []
                        for az in agents:
                            buf = getattr(az, 'replay_buffer', [])
                            total_counts.append(len(buf) if buf is not None else 0)
                            lc = 0
                            for s in buf:
                                if isinstance(s, dict) and s.get('value') is not None:
                                    lc += 1
                            labeled_counts.append(lc)
                        print(f"[WORKER{worker_id}][phase] winners={new_winners} totals={total_counts} labeled={labeled_counts}")
                except Exception:
                    pass
                prev_rankings = current_rankings
        # エピソード終端処理
        for az in agents:
            az.flush_unfinished_phase()
            az.finalize_game()
        if config.get('debug_concurrent_worker', False):
            try:
                after_counts = []
                labeled_after = []
                for az in agents:
                    buf = getattr(az, 'replay_buffer', [])
                    after_counts.append(len(buf) if buf is not None else 0)
                    la = 0
                    for s in buf:
                        if isinstance(s, dict) and s.get('value') is not None:
                            la += 1
                    labeled_after.append(la)
                print(f"[WORKER{worker_id}][episode_end] steps={step_count} buf_total={after_counts} buf_labeled={labeled_after}")
            except Exception:
                pass

        # このエピソードで確定したサンプルを Queue へ送信
        for az in agents:
            try:
                for s in az.replay_buffer:
                    if isinstance(s, dict) and s.get("value") is not None:
                        sample_queue.put(s, block=True)
                # ワーカー内ローカルバッファ消去（重複送信防止）
                az.replay_buffer.clear()
            except Exception:
                pass

        # 進捗イベント
        try:
            event_queue.put(("ep_done", 1), block=True)
        except Exception:
            pass
        ep_since_check += 1
        # 過去モデルミックスの周期適用
        if enable_mix and mix_interval > 0:
            ep_since_mix += 1
            if ep_since_mix >= mix_interval:
                _assign_opponents_from_pool()
                ep_since_mix = 0

    # 終了時フラッシュ（念のため）
    try:
        event_queue.put(("worker_exit", worker_id), block=False)
    except Exception:
        pass


class Trainer:
    def __init__(self, config: Dict[str, Any] | None = None):
        # 設定読み込み
        self.config = dict(ALPHA_ZERO_CONFIG)
        if config:
            self.config.update(config)
        # 進捗表示オプション (大量エピソード時の簡潔出力)
        self.minimal_progress = self.config.get("minimal_progress", True)
        self.progress_update_interval = self.config.get("progress_update_interval", 100)
        # 進捗バー設定
        self.use_progress_bar = self.config.get("progress_bar", True)
        self.progress_bar_width = self.config.get("progress_bar_width", 40)
        # 動的バー用内部ステート
        self._last_progress_len = 0
        # ETA 改善用パラメータ (エピソード時間の指数移動平均)
        self.eta_alpha = self.config.get("eta_smoothing_alpha", 0.25)  # 0.0(なし)〜1.0(最新のみ)
        self.monotonic_eta = self.config.get("monotonic_eta", True)    # True なら残り時間を単調減少にクランプ
        self._eta_smooth = None  # 型: float|None (EMAされた 1 エピソード所要時間秒)
        self._eta_prev_remaining = None  # 前回表示した残り秒 (単調減少用)
        # メンバ初期化
        self.agents = []  # type: ignore[list]
        self.env = None
        self.model = None
        random.seed(self.config.get("seed", 42))
        # ウォームアップ設定
        self.warmup_episodes = self.config.get("warmup_episodes", 0)
        self.warmup_mix = self.config.get("warmup_mix", ["random", "rule", "random"])  # 学習エージェント以外の順番
        # 学習対象プレイヤーID
        self.learning_player_id = self.config.get("learning_player_id", 0)
        # ロガー (setup で初期化)
        self.logger = None  # type: ignore
        # 追加: 周期チェックポイント & 直前モデルミックス用
        self.ckpt_interval = self.config.get("checkpoint_interval_episodes", 0) or 0
        self.keep_prev_model = self.config.get("keep_previous_model_opponent", False)
        self.prev_model_mix_players = int(self.config.get("previous_model_mix_players", 0) or 0)
        self._previous_model = None  # type: ignore
        self._episodes_total_run = 0  # 累積自己対局回数 (複数 self_play 呼び出し対応)
        # モデル世代管理 (リプレイに過去モデル由来サンプルを混在保持)
        self.model_version = 0
        # --- 追加: 複数過去モデルプール ---
        # 過去モデルを複数保持し、対戦相手へランダム割当して探索多様性を向上させる。
        # keep_previous_model_opponent / previous_model_mix_players が有効な場合にのみ使用。
        self.past_models = []  # List[PolicyValueNet]
        self.past_model_pool_size = int(self.config.get("past_model_pool_size", 5))  # 保存上限 (古い順に削除)
        # 何エピソードごとに opponent へ再割当するか (0/未指定なら checkpoint タイミングでのみ)
        self.opponent_mix_interval = int(self.config.get("opponent_mix_interval_episodes", 0) or 0)

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
        # フル特徴量使用時は一旦ダミー生成し、後で初回状態から次元を取得して再構築する方式を避けるため
        use_full = self.config.get("use_full_features", False)
        if use_full:
            # 一時インスタンス (full_feature_dim 後で差し替え。仮に base 次元で作る) -> 後続で lazy resize
            # ただし簡易: 最初の環境 reset 後に full_input_dim を取得して再初期化
            self.model = PolicyValueNet(
                max_policy_size=self.config["max_policy_size"],
                hidden_size=self.config["hidden_size"],
                num_players=self.config["num_players"],
                device=resolved_device,
                use_full_features=True,
                full_feature_dim= 2 + self.config["num_players"]  # 仮置き (後で再構築)
            )
        else:
            self.model = PolicyValueNet(
                max_policy_size=self.config["max_policy_size"],
                hidden_size=self.config["hidden_size"],
                num_players=self.config["num_players"],
                device=resolved_device,
                use_full_features=False,
            )
        # ロガー生成
        self.logger = TrainingLogger(
            log_dir=self.config.get("log_dir", "logs"),
            use_tensorboard=self.config.get("enable_tensorboard", True),
            clear_existing=self.config.get("clear_logs_on_start", False),
            log_mcts_samples=not self.config.get("disable_mcts_log", False),
            config=self.config
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
            # 初期世代を注入
            if hasattr(ag, 'model_version'):
                ag.model_version = self.model_version
            # 共有モードならリプレイ参照注入
            if self.shared_replay is not None:
                ag.replay_buffer = self.shared_replay
            self.agents.append(ag)
        # 環境生成 (内部の env.agents も後で差し替える可能性があるため agent_classes=None)
        self.env = DaifugoSimpleEnv(num_players=self.config["num_players"], agent_classes=None)
        # 直ちに環境の agents を学習エージェント群で上書き (ウォームアップ0の場合の不整合防止)
        self.env.agents = self.agents
        # フル特徴量モデル再構築 (初回状態から full_input_dim 取得)
        if use_full:
            try:
                _ = self.env.reset()
                sample_state = self.agents[0]._extract_state(self.env)
                full_dim = sample_state.get('full_input_dim')
                if full_dim is not None:
                    resolved_device2 = resolved_device
                    self.model = PolicyValueNet(
                        max_policy_size=self.config["max_policy_size"],
                        hidden_size=self.config["hidden_size"],
                        num_players=self.config["num_players"],
                        device=resolved_device2,
                        use_full_features=True,
                        full_feature_dim=full_dim,
                    )
                    # エージェントへ新モデルを再注入
                    for ag in self.agents:
                        ag.set_model(self.model)
                        ag.set_model_version(self.model_version)
                    if self.logger:
                        self.logger.log_text(f"[model] Rebuilt full-feature model input_dim={full_dim}")
            except Exception as e:
                print(f"[WARN] full feature model rebuild failed: {e}")
        # フルモード時: 旧フォーマットサンプル浄化 (初期残存している可能性に備える)
        if self.config.get('use_full_features'):
            for ag in self.agents:
                try:
                    if isinstance(ag.replay_buffer, list) and ag.replay_buffer:
                        ag.replay_buffer = [s for s in ag.replay_buffer if s.get('feature_version',1)==1]
                except Exception:
                    pass
        # 既存メタデータとの整合性チェック (存在すれば)
        try:
            meta_path = os.path.join(self.config['checkpoint_dir'], 'metadata.json')
            if os.path.isfile(meta_path):
                import json as _json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    old_meta = _json.load(f)
                if old_meta.get('use_full_features') != self.config.get('use_full_features'):
                    print('[WARN] metadata.use_full_features differs from current config')
        except Exception:
            pass
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
        # 初期チェックポイント (空のリプレイと初期モデル) を要求された場合に保存
        if self.config.get("initial_checkpoint_on_setup", False):
            try:
                self._save_checkpoint(version_tag=None)
                if self.logger:
                    self.logger.log_text("[init] initial checkpoint saved")
            except Exception as e:
                print(f"[WARN] initial checkpoint save failed: {e}")

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
        # 並列数で分岐
        workers = int(self.config.get("selfplay_workers", 0) or 0)
        if workers and workers > 1:
            return self._self_play_parallel(num_episodes=num_episodes, workers=workers)

        start_time = time.time()
        # 実行開始直後に 0% 進捗を表示して無音時間を減らす
        if self.minimal_progress and self.use_progress_bar and num_episodes > 0:
            bar, _pct = self._make_progress_bar(0, num_episodes)
            line = f"[SELFPLAY] {bar} 0/{num_episodes}"
            print(line, end='\r', flush=True)
            self._last_progress_len = len(line)
        for ep in range(num_episodes):
            ep_start = time.time()
            if not self.minimal_progress and not self.use_progress_bar:
                print(f"[EPOCH] {ep+1}/{num_episodes}")
            # ウォームアップ期間中は対戦相手をランダム/ルールベースに
            # 複数回の self_play 呼び出しでも通算エピソード数で判定
            if self._episodes_total_run < self.warmup_episodes:
                self._apply_warmup_opponents()
            elif self._episodes_total_run == self.warmup_episodes:
                # ウォームアップ直後に全員学習エージェントへ戻す
                self._restore_learning_agents()
            self._play_one_episode(ep)
            self._episodes_total_run += 1
            # 周期チェックポイント保存 (2000 エピソードごと等)
            if self.ckpt_interval > 0 and (self._episodes_total_run % self.ckpt_interval == 0):
                self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                # --- 過去モデルプールへ追加 (多世代保持) ---
                if self.keep_prev_model:
                    self._snapshot_current_model()
                # --- 対戦相手へ過去モデル再割当 (プールが空なら従来方式にフォールバック) ---
                if self.keep_prev_model and self.prev_model_mix_players > 0:
                    if self.past_models:
                        self._assign_past_models_to_opponents()
                    elif self._previous_model is not None:  # 後方互換: 旧単一方式
                        self._mix_previous_model_opponents()
                # チェックポイント間隔で世代を進め、エージェントへ新世代番号を反映
                self.model_version += 1
                for ag in self.agents:
                    if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                        ag.model_version = self.model_version
            # 任意のエピソード間隔で opponent 再割当 (checkpoint タイミング以外でも多様性確保)
            if self.opponent_mix_interval > 0 and self.keep_prev_model and self.prev_model_mix_players > 0:
                if (self._episodes_total_run % self.opponent_mix_interval == 0) and self.past_models:
                    self._assign_past_models_to_opponents()

            # 進捗表示 (エピソード終了後に確定時間で ETA 推定)
            if self.minimal_progress and self.use_progress_bar:
                done = ep + 1
                ep_dur = time.time() - ep_start
                # エピソード時間 EMA
                if self._eta_smooth is None:
                    self._eta_smooth = ep_dur
                else:
                    a = max(0.0, min(1.0, self.eta_alpha))
                    self._eta_smooth = a * ep_dur + (1 - a) * self._eta_smooth
                elapsed = time.time() - start_time
                estimated_total = (self._eta_smooth or 0.0) * num_episodes
                remaining = max(0.0, estimated_total - elapsed)
                if self.monotonic_eta and self._eta_prev_remaining is not None and remaining > self._eta_prev_remaining:
                    remaining = self._eta_prev_remaining
                self._eta_prev_remaining = remaining
                def _fmt(t: float):
                    m, s = divmod(int(t), 60)
                    h, m = divmod(m, 60)
                    return f"{h:d}:{m:02d}:{s:02d}"
                bar, _pct = self._make_progress_bar(done, num_episodes)
                line = f"[SELFPLAY] {bar} {done}/{num_episodes} eta={_fmt(remaining)}"
                pad = max(0, self._last_progress_len - len(line))
                print(line + ' ' * pad, end='\r' if done < num_episodes else '\n', flush=True)
                self._last_progress_len = len(line)
        # ループ終了後、バー表示時は改行が確定するので追加処理不要
        return

    # -----------------------------------------------------
    # 並行実行: 常駐自己対局(Producer) + 親プロセス学習(Consumer)
    # -----------------------------------------------------
    def train_concurrent(
        self,
        *,
        total_episodes: int,
        workers: int | None = None,
        updates_per_iter: int = 50,
        queue_maxsize: int = 15000,
        progress_print_every: int = 50,
    ):
        """自己対局をワーカープロセスで常時生成しつつ、親で学習を並行実行。

        仕様:
          - ワーカーはサンプルを sample_queue に push、親は随時消費して train_updates を小刻みに回す。
          - 既存のチェックポイント挙動を維持:
              * エピソード間隔 ckpt: ワーカーからの "ep_done" を集計し、ckpt_interval 到達時に保存
              * 学習後 ckpt: 親の学習イテレーションの各バッチグループ後に _save_checkpoint() を呼ぶ
          - 終了条件: total_episodes に達した時点でワーカー停止、残キューを消費して終了
        """
        assert total_episodes > 0
        workers = int(workers if workers is not None else (self.config.get("selfplay_workers", 0) or 0))
        workers = max(1, workers)

        # モデル配布ファイル（既存と同じ場所を使用）
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        model_blob_path = os.path.join(self.config["checkpoint_dir"], "_selfplay_worker_model.pt")
        if self.model is not None:
            self.model.save(model_blob_path)

        # 共有リプレイの存在に応じて取り込み先を決定
        dst_buffer = self.shared_replay if self.shared_replay is not None else None
        learner = self.agents[self.learning_player_id]
        if dst_buffer is None and isinstance(learner, AlphaZeroAgent):
            dst_buffer = learner.replay_buffer

        # 並行用 Queue / Event
        ctx = mp.get_context("spawn")
        sample_queue = ctx.Queue(maxsize=max(1000, queue_maxsize))
        event_queue = ctx.Queue(maxsize=10000)
        stop_event = ctx.Event()

        # ワーカー起動
        procs: List[mp.Process] = []
        for wid in range(workers):
            p = ctx.Process(
                target=_selfplay_daemon_worker,
                args=(wid, self.config, model_blob_path, sample_queue, event_queue, stop_event),
                daemon=True,
            )
            p.start()
            procs.append(p)

        # 進捗表示初期化
        if self.minimal_progress and self.use_progress_bar:
            bar, _pct = self._make_progress_bar(0, total_episodes)
            line = f"[SELFPLAY~] {bar} 0/{total_episodes} (workers={workers})"
            print(line, end='\r', flush=True)
            last_len_sp = len(line)
        else:
            last_len_sp = 0

        # ループ: イベントを処理しつつ、溜まったサンプルで学習を回す
        ep_done = 0
        train_it = 0
        ckpt_interval = int(self.ckpt_interval or 0)
        # サンプル数で学習を発火
        min_new_samples_before_train = int(self.config.get("concurrent_min_new_samples_before_train", 2000) or 2000)
        new_samples_since_train = 0
        # 保存周りを秒ベースでスロットリングして I/O 負荷を軽減
        latest_ckpt_interval_sec = float(self.config.get("concurrent_latest_save_every_sec", 30.0) or 0.0)
        blob_save_interval_sec = float(self.config.get("concurrent_blob_save_every_sec", 30.0) or 0.0)
        last_latest_ckpt_ts = 0.0
        last_blob_save_ts = 0.0
        # 親側で episodes_total_run を進めるため、event_queue からの "ep_done" をカウント
        status_log_sec = float(self.config.get("concurrent_status_log_sec", 0) or 0)
        status_log_include_mem = bool(self.config.get("status_log_include_memory", False))
        debug_flag = bool(self.config.get("concurrent_debug_logging", False))
        last_status_ts = time.time()

        def _drain_samples(max_items: int | None = None):
            nonlocal dst_buffer
            consumed = 0
            skipped = 0
            while True:
                if max_items is not None and consumed >= max_items:
                    break
                try:
                    s = sample_queue.get_nowait()
                except Exception:
                    break
                # サンプルバリデーション
                if not isinstance(s, dict):
                    skipped += 1
                    continue
                # 量子化 pi_q のみを保持する新形式にも対応
                if 'state' not in s or not ('pi' in s or 'pi_q' in s):
                    skipped += 1
                    continue
                # 取り込み
                if self.shared_replay is not None:
                    try:
                        self.shared_replay.append(s)
                    except Exception:
                        pass
                else:
                    try:
                        dst_buffer.append(s)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                consumed += 1
            if debug_flag and skipped > 0:
                print(f"[DEBUG] drain skipped={skipped} accepted={consumed} (reason: missing state or pi/pi_q)")
            return consumed

        try:
            while ep_done < total_episodes:
                # イベント処理（自己対局進捗）
                try:
                    evt, val = event_queue.get(timeout=0.1)
                except Exception:
                    evt = None
                if evt == "ep_done":
                    ep_done += int(val)
                    self._episodes_total_run += int(val)
                    # エピソード間隔 ckpt
                    if ckpt_interval > 0 and (self._episodes_total_run % ckpt_interval == 0):
                        self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                        if self.keep_prev_model:
                            self._snapshot_current_model()
                        if self.keep_prev_model and self.prev_model_mix_players > 0:
                            if self.past_models:
                                self._assign_past_models_to_opponents()
                            elif self._previous_model is not None:
                                self._mix_previous_model_opponents()
                        # モデル世代反映
                        self.model_version += 1
                        for ag in self.agents:
                            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                                ag.model_version = self.model_version

                    # 進捗表示
                    if self.minimal_progress and self.use_progress_bar:
                        bar, _pct = self._make_progress_bar(ep_done, total_episodes)
                        line = f"[SELFPLAY~] {bar} {ep_done}/{total_episodes} (workers={workers})"
                        pad = max(0, last_len_sp - len(line))
                        print(line + ' ' * pad, end='\r' if ep_done < total_episodes else '\n', flush=True)
                        last_len_sp = len(line)

                # サンプル取り込み（少しずつ）
                consumed_now = _drain_samples(max_items=500)
                new_samples_since_train += int(consumed_now)
                if debug_flag and consumed_now>0:
                    print(f"[DEBUG] drained={consumed_now} total_new={new_samples_since_train} replay_size={len(self.shared_replay) if self.shared_replay else 'n/a'}")
                # 追加デバッグ: 学習トリガ未達時に一定エピソードごとにラベル付サンプル比率を観測
                if debug_flag and (ep_done % max(1, int(self.config.get('debug_status_interval_eps', 25))) == 0):
                    try:
                        if self.shared_replay is not None:
                            total_rb = len(self.shared_replay)
                            labeled_rb = 0
                            try:
                                # shared_replay.iter_all で全要素にアクセスできる前提
                                for rec in self.shared_replay.iter_all():
                                    if isinstance(rec, dict) and rec.get('value') is not None:
                                        labeled_rb += 1
                            except Exception:
                                pass
                            print(f"[DEBUG][parent] ep_done={ep_done} shared_replay_total={total_rb} labeled={labeled_rb}")
                    except Exception:
                        pass

                # 新規サンプルがしきい値を超えたら学習を回す
                now = time.time()
                if new_samples_since_train >= min_new_samples_before_train:
                    # updates_per_iter ステップだけ学習
                    for _ in range(max(1, int(updates_per_iter))):
                        loss_info = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
                        # 学習回数カウント
                        train_it += 1
                        # 最小限の進捗表示
                        if self.minimal_progress and self.use_progress_bar:
                            if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                                loss_part = f"loss={loss_info['loss']:.4f}"
                            else:
                                loss_part = "loss=----"
                            line = f"[TRAIN~] it={train_it} {loss_part}"
                            print(line, end='\r', flush=True)
                        # ロガーへ
                        if self.logger and isinstance(loss_info, dict) and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                            self.logger.log_train(loss_info)
                    # 既存動作に合わせた保存は維持しつつ、高頻度I/Oを抑制
                    if latest_ckpt_interval_sec > 0.0:
                        if (now - last_latest_ckpt_ts) >= latest_ckpt_interval_sec:
                            self._save_checkpoint()
                            if self.keep_prev_model:
                                self._snapshot_current_model()
                            last_latest_ckpt_ts = now
                    else:
                        # 0 以下なら常に保存（従来挙動）
                        self._save_checkpoint()
                        if self.keep_prev_model:
                            self._snapshot_current_model()
                    # ワーカー配布用モデルもスロットリングして保存
                    if self.model is not None:
                        do_blob_save = True
                        if blob_save_interval_sec > 0.0:
                            do_blob_save = (now - last_blob_save_ts) >= blob_save_interval_sec
                        if do_blob_save:
                            try:
                                self.model.save(model_blob_path)
                            except Exception:
                                pass
                            else:
                                last_blob_save_ts = now
                    # 学習トリガをリセット
                    new_samples_since_train = 0

                # 定期ステータスログ
                now2 = time.time()
                if status_log_sec > 0 and (now2 - last_status_ts) >= status_log_sec:
                    last_status_ts = now2
                    try:
                        replay_size = len(self.shared_replay) if self.shared_replay is not None else (len(dst_buffer) if dst_buffer is not None and hasattr(dst_buffer, '__len__') else None)
                    except Exception:
                        replay_size = None
                    # 追加: 主要I/Oファイルサイズ (MB) を取得して status に添付
                    io_parts = []
                    try:
                        log_dir = self.config.get('log_dir', 'logs')
                        ev_path = os.path.join(log_dir, 'events.log')
                        mcts_path = os.path.join(log_dir, 'mcts_samples.jsonl')
                        ckpt_path = self.config.get('checkpoint_path', 'checkpoints/policy_value_latest.pt')
                        def _mb(p):
                            try:
                                return os.path.getsize(p) / (1024*1024)
                            except Exception:
                                return None
                        ev_mb = _mb(ev_path)
                        mcts_mb = _mb(mcts_path)
                        ckpt_mb = _mb(ckpt_path)
                        if ev_mb is not None:
                            io_parts.append(f"events:{ev_mb:.1f}MB")
                        if mcts_mb is not None:
                            io_parts.append(f"mcts:{mcts_mb:.1f}MB")
                        if ckpt_mb is not None:
                            io_parts.append(f"ckpt:{ckpt_mb:.1f}MB")
                    except Exception:
                        pass
                    io_sizes_str = (" io_sizes=" + ",".join(io_parts)) if io_parts else ""
                    msg = f"[status] ep_done={ep_done} train_it={train_it} new_since_train={new_samples_since_train} replay_size={replay_size} workers={workers}{io_sizes_str}"
                    if self.logger:
                        self.logger.log_text(msg)
                        if status_log_include_mem:
                            try:
                                sample_cnt = replay_size
                                self.logger.log_memory_snapshot(sample_count=sample_cnt, force=False)
                            except Exception:
                                pass
                    else:
                        print(msg)

            # 終了条件到達: ワーカー停止指示
            stop_event.set()
        finally:
            # 最後に少量の未学習サンプルが残っていれば 1 バーストだけ学習
            if new_samples_since_train > 0:
                for _ in range(max(1, int(updates_per_iter))):
                    _ = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
                    train_it += 1
                # 仕上げの保存（スロットリングに関わらず1回実施）
                self._save_checkpoint()
                if self.keep_prev_model and self.model is not None:
                    self._snapshot_current_model()
                    try:
                        self.model.save(model_blob_path)
                    except Exception:
                        pass
            # 残りサンプルを吸い上げ
            _ = _drain_samples(max_items=None)
            # ワーカー join
            for p in procs:
                try:
                    p.join(timeout=5)
                except Exception:
                    pass
            # 最終チェックポイント（念のため）
            self._save_checkpoint()
        # サマリーを返す（合計エピソード数と総学習ステップ数）
        return {"episodes": ep_done, "train_updates": train_it}

    # ---------------- 並列自己対局 ----------------
    def _self_play_parallel(self, num_episodes: int, workers: int):
        """マルチプロセスで自己対局を並行実行し、共有リプレイへ集約する。

        注意:
          - Windows は spawn 方式のため、ワーカー関数はトップレベルに定義。
          - モデルは CPU コピーをミニチェックポイント経由で各ワーカーへ配布。
          - ワーカーはローカルバッファに確定サンプルのみを溜め、終了時に親へ返す。
        """
        start_time = time.time()
        # 進捗用
        total_done = 0
        last_progress_len = 0

        # モデル配布: 最新を一時パスへ保存（各チャンクで共通利用）
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        model_blob_path = os.path.join(self.config["checkpoint_dir"], "_selfplay_worker_model.pt")
        if self.model is not None:
            self.model.save(model_blob_path)

        remaining = int(num_episodes)
        ckpt_interval = int(self.ckpt_interval or 0)

        # 実行開始直後に 0% 進捗を表示（長いエピソードでも即表示）
        if self.minimal_progress and self.use_progress_bar and num_episodes > 0:
            bar, _pct = self._make_progress_bar(0, num_episodes)
            line = f"[SELFPLAY*] {bar} 0/{num_episodes} (workers={workers})"
            print(line, end='\r', flush=True)
            last_progress_len = len(line)

        while remaining > 0:
            # 次のチェックポイント境界までの残り (0=無効なら全量)
            if ckpt_interval > 0:
                to_next = ckpt_interval - (self._episodes_total_run % ckpt_interval)
                if to_next <= 0:
                    to_next = ckpt_interval
                chunk = min(remaining, to_next)
            else:
                chunk = remaining

            # このチャンクを workers にほぼ均等割当
            base = chunk // workers
            rem = chunk % workers
            ep_splits = [base + (1 if i < rem else 0) for i in range(workers)]
            tasks = []
            for wid, n_ep in enumerate(ep_splits):
                if n_ep <= 0:
                    continue
                tasks.append((wid, n_ep, self.config, model_blob_path))

            # 実行（spawn プールをチャンク毎に生成して安全に実行）
            results = []
            if tasks:
                with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
                    for wid, n_ep, cfg, model_path in tasks:
                        results.append(pool.apply_async(_selfplay_worker_entry, (wid, n_ep, cfg, model_path)))
                    # 逐次回収しリプレイへ反映
                    for res in results:
                        worker_out = res.get()
                        samples = worker_out.get("samples", [])
                        if self.shared_replay is None:
                            learner = self.agents[self.learning_player_id]
                            if isinstance(learner, AlphaZeroAgent):
                                for s in samples:
                                    learner.replay_buffer.append(s)
                        else:
                            for s in samples:
                                self.shared_replay.append(s)
                        total_done += worker_out.get("episodes", 0)
                        if self.minimal_progress and self.use_progress_bar:
                            bar, _pct = self._make_progress_bar(total_done, num_episodes)
                            line = f"[SELFPLAY*] {bar} {total_done}/{num_episodes} (workers={workers})"
                            pad = max(0, last_progress_len - len(line))
                            print(line + ' ' * pad, end='\r' if total_done < num_episodes else '\n', flush=True)
                            last_progress_len = len(line)

            # エピソードカウンタを 1 ずつ進めて単一実行時と同じタイミングのフックを発火
            for _ in range(chunk):
                self._episodes_total_run += 1
                # 周期チェックポイント保存（単一実行時と同一判定）
                if self.ckpt_interval > 0 and (self._episodes_total_run % self.ckpt_interval == 0):
                    self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                    if self.keep_prev_model:
                        self._snapshot_current_model()
                    if self.keep_prev_model and self.prev_model_mix_players > 0:
                        if self.past_models:
                            self._assign_past_models_to_opponents()
                        elif self._previous_model is not None:
                            self._mix_previous_model_opponents()
                    # モデル世代を進め、エージェントへ反映
                    self.model_version += 1
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                            ag.model_version = self.model_version
                # 任意のエピソード間隔で opponent 再割当
                if self.opponent_mix_interval > 0 and self.keep_prev_model and self.prev_model_mix_players > 0:
                    if (self._episodes_total_run % self.opponent_mix_interval == 0) and self.past_models:
                        self._assign_past_models_to_opponents()

            remaining -= chunk

        # 終了ログ
        elapsed = time.time() - start_time
        if not self.minimal_progress:
            print(f"[SELFPLAY*] finished {num_episodes} episodes in {elapsed:.1f}s using {workers} workers")
        return

    # -----------------------------------------------------
    # 過去モデルスナップショット
    # -----------------------------------------------------
    def _snapshot_current_model(self):
        """現在モデルを CPU 上に複製し past_models に追加。容量上限を超えたら FIFO で削除。

        注意: self._previous_model (旧互換) も最新スナップショットとして更新し、
              プール未使用設定時のフォールバックを維持する。
        """
        if self.model is None:
            return
        try:
            # live モデル state_dict を CPU 上にコピー
            # フル特徴量モデルの場合は input_dim を揃える必要がある (そうでないと 246->6 などの shape mismatch が発生)
            use_full = getattr(self.model, 'use_full_features', False)
            full_dim = None
            if use_full:
                try:
                    # backbone 最初の Linear の in_features から復元
                    full_dim = int(getattr(self.model.backbone[0], 'in_features'))  # type: ignore[index]
                except Exception:
                    # 取得失敗時はフォールバック (ロードで再度失敗する可能性あり)
                    full_dim = None
            snap = PolicyValueNet(
                max_policy_size=self.config["max_policy_size"],
                hidden_size=self.config["hidden_size"],
                num_players=self.config["num_players"],
                device="cpu",
                use_full_features=use_full,
                full_feature_dim=full_dim if use_full else None,
            )
            # strict=True でロードし shape 不一致を早期検出 (問題あれば例外キャッチ側で警告)
            snap.load_state_dict(self.model.state_dict(), strict=True)  # type: ignore[arg-type]
            self.past_models.append(snap)
            self._previous_model = snap  # 互換
            # 上限超過なら古いものから削除
            if self.past_model_pool_size > 0 and len(self.past_models) > self.past_model_pool_size:
                overflow = len(self.past_models) - self.past_model_pool_size
                if overflow > 0:
                    del self.past_models[0:overflow]
            if self.logger:
                self.logger.log_text(f"[snapshot] pool_size={len(self.past_models)}")
        except Exception as e:
            if self.logger:
                self.logger.log_text(f"[WARN] snapshot failed: {e}")
            else:
                print(f"[WARN] snapshot failed: {e}")

    # -----------------------------------------------------
    # 過去モデルを対戦相手へランダム割当
    # -----------------------------------------------------
    def _assign_past_models_to_opponents(self):
        if not self.past_models:
            return
        candidate_indices = [i for i in range(self.config["num_players"]) if i != self.learning_player_id]
        if not candidate_indices:
            return
        random.shuffle(candidate_indices)
        selected = candidate_indices[: self.prev_model_mix_players]
        # 割当: 選択プレイヤーにランダムな過去モデルを割り当てる
        for idx in selected:
            ag = self.agents[idx]
            if isinstance(ag, AlphaZeroAgent):
                model_choice = random.choice(self.past_models)
                ag.set_model(model_choice)
        # 学習プレイヤーは常に最新モデル
        learner = self.agents[self.learning_player_id]
        if isinstance(learner, AlphaZeroAgent):
            learner.set_model(self.model)
        if self.logger:
            self.logger.log_text(f"[mix-multi] assigned past models to players={selected} (pool={len(self.past_models)})")
        else:
            print(f"[INFO] past models mixed into players={selected} (pool={len(self.past_models)})")
        if self.minimal_progress:
            # ループ終了で改行確定
            if not self.use_progress_bar:
                print()

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
        start_time = time.time()
        last_print = 0
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
            # 進捗出力 (動的バー)
            if self.minimal_progress and self.use_progress_bar:
                done = i + 1
                # シンプル表示: バー無し / ETA無し / 一行上書き
                if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                    loss_part = f"loss={loss_info['loss']:.4f}"
                else:
                    loss_part = "loss=----"
                lr_val = None
                try:
                    opt = getattr(self.agents[0], '_optimizer', None)
                    if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                        lr_val = opt.param_groups[0].get('lr', None)
                except Exception:
                    lr_val = None
                lr_part = f"lr={lr_val:.2e}" if lr_val is not None else "lr=----"
                line = f"[TRAIN] {done}/{num_updates} {loss_part} {lr_part}"
                pad = max(0, self._last_progress_len - len(line))
                print(line + ' ' * pad, end='\r' if done < num_updates else '\n', flush=True)
                self._last_progress_len = len(line)
            else:
                if (i + 1) % self.config.get("log_interval", 50) == 0:
                    print(f"[TRAIN] epoch={i+1}/{num_updates} loss={loss_info}")
            # ロガーへ (train_step 内で既に push されている場合は二重記録を避ける)
            if self.logger and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                self.logger.log_train(loss_info)
        self._save_checkpoint()
        # 学習直後の最新モデルもスナップショット (自己対局前に世代差が明確になる)
        if self.keep_prev_model:
            self._snapshot_current_model()
        # CSV サマリーモードならここで 1 行のみ書き出し
        if self.logger and getattr(self.logger, 'csv_summary_only', False):
            try:
                self.logger.write_csv_summaries()
            except Exception as e:
                print(f"[WARN] write_csv_summaries failed: {e}")

    # -----------------------------------------------------
    # 進捗バー生成ヘルパー
    # -----------------------------------------------------
    def _make_progress_bar(self, done: int, total: int, pct: float | None = None):
        width = max(10, int(self.progress_bar_width))
        ratio = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
        fill = int(ratio * width)
        bar = "#" * fill + "-" * (width - fill)
        pct_val = ratio * 100.0
        return f"[{bar}] {pct_val:6.2f}%", f"{pct_val:6.2f}%"

    # -----------------------------------------------------
    # チェックポイント
    # -----------------------------------------------------
    def _save_checkpoint(self):  # backward compatibility name
        self._save_checkpoint(version_tag=None)

    def _save_checkpoint(self, version_tag: str | None = None):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        # アトミック保存ヘルパ
        def _atomic_save(fn, save_callable):
            tmp = fn + ".tmp"
            try:
                save_callable(tmp)
                os.replace(tmp, fn)
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                print(f"[WARN] atomic save fallback ({fn}): {e}")
                try:
                    save_callable(fn)
                except Exception as ee:
                    print(f"[ERROR] save failed {fn}: {ee}")

        # 最新モデル保存 (atomic)
        latest_path = self.config["checkpoint_path"]
        if self.model is not None:
            _atomic_save(latest_path, lambda p: self.model.save(p))
        # バージョン付き保存
        if version_tag:
            ver_path = os.path.join(self.config["checkpoint_dir"], f"policy_value_{version_tag}.pt")
            if self.model is not None:
                _atomic_save(ver_path, lambda p: self.model.save(p))
        # リプレイ保存: 共有モードなら共有バッファを保存、そうでなければ従来通り代表エージェント
        replay_path = self.config.get("replay_path", "replay_buffer.joblib")
        if self.shared_replay is not None:
            try:
                # フル特徴量モードで legacy 混入していたらフィルタ
                if self.config.get('use_full_features'):
                    try:
                        before = len(self.shared_replay)
                        self.shared_replay.data = [s for s in self.shared_replay.data if s.get('feature_version',1)==1]
                        after = len(self.shared_replay)
                        if after < before:
                            print(f"[INFO] shared replay purge legacy {before-after} samples (full mode)")
                    except Exception:
                        pass
                self.shared_replay.save(replay_path)
            except Exception as e:
                print(f"[WARN] shared replay save failed: {e}")
        else:
            self.agents[0].save_replay(replay_path)
        # メタデータ保存
        try:
            # 簡易 config ハッシュ
            import json, hashlib, time as _time
            cfg_sorted = json.dumps(self.config, sort_keys=True, ensure_ascii=False).encode('utf-8')
            cfg_hash = hashlib.md5(cfg_sorted).hexdigest()
            meta = {
                "model_version": self.model_version,
                "use_full_features": bool(self.config.get('use_full_features')),\
                "num_players": self.config.get('num_players'),
                "hidden_size": self.config.get('hidden_size'),
                "max_policy_size": self.config.get('max_policy_size'),
                "seed": self.config.get('seed'),
                "config_md5": cfg_hash,
            }
            try:
                if self.model is not None and hasattr(self.model, 'backbone'):
                    meta['full_feature_dim'] = getattr(self.model.backbone[0], 'in_features', None)
            except Exception:
                pass
            meta['saved_at'] = _time.time()
            meta_path = os.path.join(self.config['checkpoint_dir'], 'metadata.json')
            def _write_meta(p):
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            # atomic 書き込み
            tmp_meta = meta_path + '.tmp'
            try:
                _write_meta(tmp_meta)
                os.replace(tmp_meta, meta_path)
            except Exception:
                try:
                    if os.path.exists(tmp_meta):
                        os.remove(tmp_meta)
                except Exception:
                    pass
                _write_meta(meta_path)
        except Exception as e:
            print(f"[WARN] metadata save failed: {e}")

    # 即時保存用ヘルパ (ユーザーが強制的に現在状態を吐き出したい場合)
    def force_save(self):
        try:
            self._save_checkpoint()
            if self.logger:
                self.logger.log_text("[force_save] checkpoint+replay saved")
        except Exception as e:
            print(f"[WARN] force_save failed: {e}")

    # -----------------------------------------------------
    # 直前モデルを対戦相手に混在させる
    # -----------------------------------------------------
    def _mix_previous_model_opponents(self):
        if self._previous_model is None:
            return
        # 学習プレイヤー以外から N 人を選び、そのエージェントの model を previous_model に差し替える
        candidate_indices = [i for i in range(self.config["num_players"]) if i != self.learning_player_id]
        if not candidate_indices:
            return
        random.shuffle(candidate_indices)
        selected = candidate_indices[: self.prev_model_mix_players]
        for idx in selected:
            ag = self.agents[idx]
            if isinstance(ag, AlphaZeroAgent):
                ag.set_model(self._previous_model)
        # 学習プレイヤーのモデルは必ず最新 (self.model) に戻しておく
        learner = self.agents[self.learning_player_id]
        if isinstance(learner, AlphaZeroAgent):
            learner.set_model(self.model)
        if self.logger:
            self.logger.log_text(f"[mix] applied previous model to players={selected}")
        else:
            print(f"[INFO] previous model mixed into players={selected}")


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
    parser.add_argument("--workers", type=int, default=None, help="自己対局の並列ワーカー数。未指定は設定値を使用")
    # 並行実行オプション
    parser.add_argument("--concurrent", action="store_true", help="自己対局と学習を並行実行する常駐モードを有効化")
    parser.add_argument("--concurrent-updates-per-iter", type=int, default=50, help="並行モードでの学習ステップ束ね数")
    parser.add_argument("--concurrent-queue-size", type=int, default=50000, help="並行モードでのサンプルQueueの最大長")
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="並行モード(--concurrent)での自己対局の総エピソード上限",
    )
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
    if args.workers is not None:
        cfg["selfplay_workers"] = max(0, int(args.workers))

    #print("[CLI] Config overrides:", {k: cfg[k] for k in ["num_simulations","batch_size","log_dir","enable_tensorboard","device"] if k in cfg})
    trainer = Trainer(config=cfg)
    trainer.setup()
    # 並行モードが指定された場合
    if args.concurrent:
        total_eps = args.total_episodes if args.total_episodes is not None else args.episodes
        summary = trainer.train_concurrent(
            total_episodes=int(total_eps),
            workers=args.workers if args.workers is not None else None,
            updates_per_iter=int(args.concurrent_updates_per_iter),
            queue_maxsize=int(args.concurrent_queue_size),
        )
        # 並行モードの終了メッセージ（実績値）
        try:
            print(f"[INFO] concurrent run finished total_episodes={summary.get('episodes')} train_updates={summary.get('train_updates')}")
        except Exception:
            pass
    else:
        # 従来の単発実行
        trainer.self_play(num_episodes=args.episodes)
        trainer.train_updates(num_updates=args.updates)
        print(f"[INFO] run finished episodes={args.episodes} updates={args.updates}")
    #print("[INFO] checkpoints ->", cfg.get("checkpoint_path"))
    #print("[INFO] replay ->", cfg.get("replay_path"))
