"""AlphaZero 大富豪 Trainer

このモジュールは学習管理を担当し、環境/モデル/エージェント生成は
agents.factory へ、長大な常駐自己対局ワーカー処理は trainer.workers へ委譲します。
"""

from __future__ import annotations

import argparse
import os
import random
import time
import queue as _queue
import multiprocessing as mp
import copy as _cp
from typing import Any, Dict, List

from agents.config import ALPHA_ZERO_CONFIG
from agents.drl_agent import AlphaZeroAgent
from agents.models import PolicyValueNet
from agents.factory import create_env_and_agents
from utils.logger import TrainingLogger
from agents.replay_buffer import ReplayBuffer
from trainer.workers import selfplay_daemon_worker_entry as _daemon_worker_entry, SelfplayDaemonWorker
from trainer import orchestrator as _orc


def _selfplay_worker_entry(worker_id: int, num_episodes: int, cfg: Dict[str, Any], model_path: str):
    """短期チャンク自己対局ワーカー (_self_play_parallel 用)。
    既存ロジック簡略版: 確定サンプルのみ収集して戻す。
    """
    bundle = create_env_and_agents(cfg, context="worker", model_path=model_path, worker_id=worker_id)
    agents = bundle.agents
    env = bundle.env
    learning_player_id = int(cfg.get("learning_player_id", 0) or 0)
    local_samples: List[Dict[str, Any]] = []
    zero_buf = bool(cfg.get("worker_zero_buffer", False))
    for _ep in range(num_episodes):
        if hasattr(env, "reset"):
            env.reset()
        for ag in agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, "reset_episode"):
                ag.reset_episode()
        step_count = 0
        max_steps = int(cfg.get("max_episode_steps", 1000) or 1000)
        while not getattr(env.game, "done", False):
            if step_count >= max_steps:
                break
            cur = env.game.turn
            agent = agents[cur]
            if isinstance(agent, AlphaZeroAgent):
                action = agent.select_action(env, training=False)
            else:
                current_player = env.game.players[cur]
                hand = current_player.hand
                field = env.game.current_field[:]
                legal_actions = env._generate_legal_actions(hand, field)
                obs_simple = {"hand": hand, "field": field}
                action = agent.select_action(obs_simple, legal_actions=legal_actions)
            ext_act = action
            try:
                env.step(external_action=ext_act)
            except TypeError:
                env.step(ext_act)
            step_count += 1
        for ag in agents:
            if isinstance(ag, AlphaZeroAgent):
                ag.flush_unfinished_phase()
                ag.finalize_game()
        # Collect labeled samples from all AlphaZeroAgent instances (remove filter by learning_player)
        for ag in agents:
            if isinstance(ag, AlphaZeroAgent):
                try:
                    labeled = ag.pop_labeled_samples()
                except Exception:
                    labeled = []
                if not zero_buf and labeled:
                    local_samples.extend(labeled)
    return {"samples": local_samples, "episodes": num_episodes}


def _selfplay_daemon_worker(worker_id: int, config: Dict[str, Any], model_path: str, sample_queue, event_queue, stop_event, control_queue, request_q, response_q):
    """常駐自己対局ワーカー委譲ラッパー。"""
    return _daemon_worker_entry(worker_id, config, model_path, sample_queue, event_queue, stop_event, control_queue, request_q, response_q)


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
        
        # --- 非同期ゲート評価用 ---
        self._gate_pool = None  # deprecated: ProcessPoolExecutor (not used)
        self._gate_future = None  # deprecated: Future (not used)
        self._gate_proc = None
        self._gate_queue = None
        self._gate_candidate_path = None
        self._gate_baseline_path = None
        # ゲート最新結果（毎更新ログ用に保持）
        try:
            self._last_gate_threshold = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
        except Exception:
            self._last_gate_threshold = 0.6
        self._last_gate_result = None  # 型: Optional[Dict[str, Any]]
        # 評価ゲートの起動制御（学習更新回数ベース）
        self._last_gate_start_it = -1  # 直近の評価開始時点の train_it（未開始は -1）

    def _should_start_gate(self, train_it: int) -> bool:
        """評価ゲートを開始してよいかを学習更新回数で判定する。
        - eval_gate_start_after_updates: この回数に到達するまで起動しない
        - eval_gate_every_updates: 直近の起動からこの回数に満たない間は起動しない
        """
        try:
            start_after = int(self.config.get("eval_gate_start_after_updates", 0) or 0)
        except Exception:
            start_after = 0
        try:
            every = int(self.config.get("eval_gate_every_updates", 0) or 0)
        except Exception:
            every = 0
        if train_it < start_after:
            return False
        if every > 0 and self._last_gate_start_it >= 0 and (train_it - self._last_gate_start_it) < every:
            return False
        return True

    # -----------------------------------------------------
    # 準備
    # -----------------------------------------------------
    def setup(self):
        # まずメインプロセスのスレッド環境変数を先に設定（ライブラリ初期化前に反映させる）
        try:
            tn_main_env = int(self.config.get("torch_num_threads_main", 0) or 0)
            if tn_main_env > 0:
                os.environ["OMP_NUM_THREADS"] = str(tn_main_env)
                os.environ["MKL_NUM_THREADS"] = str(tn_main_env)
                os.environ["OPENBLAS_NUM_THREADS"] = str(tn_main_env)
                os.environ["NUMEXPR_NUM_THREADS"] = str(tn_main_env)
        except Exception:
            pass

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
        # メインプロセスの CPU スレッド数を制御（必要時）
        try:
            tn_main = int(self.config.get("torch_num_threads_main", 0) or 0)
            itn_main = int(self.config.get("torch_num_interop_threads_main", 0) or 0)
            if tn_main > 0 or itn_main > 0:
                import torch as _t
                if tn_main > 0:
                    _t.set_num_threads(tn_main)
                if itn_main > 0 and hasattr(_t, "set_num_interop_threads"):
                    _t.set_num_interop_threads(itn_main)
        except Exception:
            pass
        # ロガー生成（先に作ってエージェントへ注入できるようにする）
        self.logger = TrainingLogger(
            log_dir=self.config.get("log_dir", "logs"),
            use_tensorboard=self.config.get("enable_tensorboard", True),
            clear_existing=self.config.get("clear_logs_on_start", False),
            log_mcts_samples=not self.config.get("disable_mcts_log", False),
            config=self.config
        )
        # 旧: 推論専用モデル分離 (GPU競合回避) を廃止。self.model のみ使用。
        # スレッド設定の実測値ログは削除（簡潔化）
        # 共有リプレイバッファ (use_shared_replay=true の場合)
        self.shared_replay = None
        if self.config.get("use_shared_replay", False):
            self.shared_replay = ReplayBuffer(
                maxlen=self.config.get("buffer_size", 50000),
                path=self.config.get("replay_path", "replay_buffer.joblib")
            )
            # Disable .bak backups if configured
            if hasattr(self.shared_replay, 'set_backup_enabled'):
                self.shared_replay.set_backup_enabled(bool(self.config.get('replay_backup_enable', True)))
            # cycle logger を接続（ロガーがあれば）
            if self.logger and hasattr(self.shared_replay, 'set_cycle_logger'):
                self.shared_replay.set_cycle_logger(lambda msg: self.logger.log_text(msg))
            # 非同期 append を有効化（設定時）
            try:
                if bool(self.config.get('replay_async_enabled', False)) and hasattr(self.shared_replay, 'enable_async'):
                    qcap = int(self.config.get('replay_async_queue_maxsize', 50000) or 50000)
                    self.shared_replay.enable_async(max_queue=qcap)
                    if self.logger:
                        self.logger.log_text(f"[replay] async_enabled cap={qcap}")
            except Exception:
                pass
            # 満杯トリガのサイズサイクル設定（設定時）
            try:
                if bool(self.config.get('replay_cycle_on_full_enabled', False)) and hasattr(self.shared_replay, 'configure_cycle_on_full'):
                    hi = int(self.config.get('replay_cycle_on_full_high_size', self.config.get('buffer_size', 250000)))
                    lo = int(self.config.get('replay_cycle_on_full_low_size', 100000))
                    mode = str(self.config.get('replay_cycle_shrink_mode', 'keep_newest') or 'keep_newest')
                    self.shared_replay.configure_cycle_on_full(high_size=hi, low_size=lo, mode=mode)
                    if self.logger:
                        self.logger.log_text(f"[replay-cycle] on_full enabled high={hi} low={lo} mode={mode}")
            except Exception:
                pass
        # モデル/エージェント/環境を共通ファクトリで生成
        latest_path = self.config.get("checkpoint_path", os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "policy_value_latest.pt"))
        bundle = create_env_and_agents(
            self.config,
            context="main",
            model_path=latest_path,
            resolved_device=resolved_device,
            shared_replay=self.shared_replay,
            logger=self.logger,
        )
        self.model = bundle.model
        self.agents = bundle.agents
        self.env = bundle.env
        self._loaded_from_checkpoint = bool(bundle.loaded_from_checkpoint)
        if self._loaded_from_checkpoint:
            # logger 初期化後なのでログも残す
            print(f"[resume] loaded model from {latest_path}")
            try:
                if self.logger:
                    self.logger.log_text(f"[resume] loaded model from {latest_path}")
            except Exception:
                pass
        # フルモード時: 旧フォーマットサンプル浄化 (初期残存している可能性に備える)
        if self.config.get('use_full_features'):
            for ag in self.agents:
                try:
                    if isinstance(ag.replay_buffer, list) and ag.replay_buffer:
                        ag.replay_buffer = [s for s in ag.replay_buffer if s.get('feature_version',1)==1]
                except Exception:
                    pass
        # 既存メタデータとの整合性チェック (存在すれば) + モデル世代の復元
        try:
            meta_path = os.path.join(self.config['checkpoint_dir'], 'metadata.json')
            if os.path.isfile(meta_path):
                import json as _json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    old_meta = _json.load(f)
                if old_meta.get('use_full_features') != self.config.get('use_full_features'):
                    print('[WARN] metadata.use_full_features differs from current config')
                # モデル世代の復元（存在すれば）
                try:
                    mv = int(old_meta.get('model_version'))
                    self.model_version = mv
                    # 既存エージェントへも反映
                    for ag in self.agents:
                        if hasattr(ag, 'model_version'):
                            ag.model_version = self.model_version
                        if hasattr(ag, 'set_model_version'):
                            ag.set_model_version(self.model_version)
                except Exception:
                    pass
        except Exception:
            pass
        # Optimizer の再開: 学習エージェントの optimizer 状態を復元（存在すれば）
        try:
            opt_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'optimizer_latest.pt')
            if os.path.isfile(opt_path):
                ag0 = self.agents[self.learning_player_id]
                # Optimizer を用意してから読み込み
                try:
                    if hasattr(ag0, 'ensure_optimizer'):
                        ag0.ensure_optimizer()
                    if hasattr(ag0, 'load_optimizer'):
                        ok = ag0.load_optimizer(opt_path, map_location=resolved_device)
                        if self.logger:
                            self.logger.log_text(f"[resume] optimizer load {'ok' if ok else 'failed'} from {opt_path}")
                except Exception as e:
                    print(f"[WARN] optimizer load failed: {e}")
        except Exception:
            pass
        # Scheduler の再開: 学習エージェントの学習率スケジューラを復元（存在すれば）
        try:
            sch_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'scheduler_latest.pt')
            if os.path.isfile(sch_path):
                ag0 = self.agents[self.learning_player_id]
                try:
                    if hasattr(ag0, 'ensure_optimizer'):
                        ag0.ensure_optimizer()
                    if hasattr(ag0, 'ensure_scheduler'):
                        ag0.ensure_scheduler()
                    if hasattr(ag0, 'load_scheduler'):
                        ok = ag0.load_scheduler(sch_path)
                        if self.logger:
                            self.logger.log_text(f"[resume] scheduler load {'ok' if ok else 'failed'} from {sch_path}")
                except Exception as e:
                    print(f"[WARN] scheduler load failed: {e}")
        except Exception:
            pass
        # 既存の train_updates.csv から update_step を復元して単調増加を保証
        try:
            log_dir = self.config.get("log_dir", "logs")
            csv_path = os.path.join(log_dir, "train_updates.csv")
            last_step = 0
            if os.path.isfile(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as rf:
                    lines = rf.readlines()
                # 末尾から走査し、ヘッダ以外の最終行の update_step を取得
                for line in reversed(lines):
                    s = line.strip()
                    if not s or s.startswith('update_step'):
                        continue
                    first = s.split(',')[0].strip()
                    try:
                        last_step = int(first)
                        break
                    except Exception:
                        continue
            # logger の内部カウンタへ反映（以降の log_train で +1 されて継続）
            if hasattr(self, 'logger') and self.logger is not None:
                try:
                    self.logger.update_step = int(last_step)
                    # 参考ログ
                    self.logger.log_text(f"[resume] logger.update_step restored to {last_step}")
                except Exception:
                    pass
        except Exception:
            pass
        # オプション: 今回の再開に限り Warmup をやり直す（ワンショット）
        try:
            if bool(self.config.get('resume_reset_warmup_once', False)):
                ag0 = self.agents[self.learning_player_id]
                if hasattr(ag0, 'reset_scheduler_warmup'):
                    done = ag0.reset_scheduler_warmup()
                    if self.logger:
                        self.logger.log_text(f"[resume] warmup reset {'ok' if done else 'failed'} (one-shot)")
                # プロセス内で一度だけ実行するためフラグを下ろす
                self.config['resume_reset_warmup_once'] = False
        except Exception:
            pass
        # 既存エージェントへ世代番号/参照を反映（ファクトリでも注入済みだが整合のため維持）
        for ag in self.agents:
            if hasattr(ag, 'model_version'):
                ag.model_version = self.model_version
        # 既存リプレイのロード (継続学習対応)
        if self.shared_replay is not None:
            replay_path = self.config.get("replay_path", "replay_buffer.joblib")
            if os.path.exists(replay_path):
                try:
                    loaded = ReplayBuffer.load(replay_path)
                    # 差し替え
                    self.shared_replay = loaded
                    # cycle logger 再接続
                    try:
                        if self.logger and hasattr(self.shared_replay, 'set_cycle_logger'):
                            self.shared_replay.set_cycle_logger(lambda msg: self.logger.log_text(msg))
                    except Exception:
                        pass
                    # ロード後も非同期を有効化
                    try:
                        if bool(self.config.get('replay_async_enabled', False)) and hasattr(self.shared_replay, 'enable_async'):
                            qcap = int(self.config.get('replay_async_queue_maxsize', 50000) or 50000)
                            self.shared_replay.enable_async(max_queue=qcap)
                    except Exception:
                        pass
                    # ロード後も on-full サイクル設定
                    try:
                        if bool(self.config.get('replay_cycle_on_full_enabled', False)) and hasattr(self.shared_replay, 'configure_cycle_on_full'):
                            hi = int(self.config.get('replay_cycle_on_full_high_size', self.config.get('buffer_size', 250000)))
                            lo = int(self.config.get('replay_cycle_on_full_low_size', 100000))
                            mode = str(self.config.get('replay_cycle_shrink_mode', 'keep_newest') or 'keep_newest')
                            self.shared_replay.configure_cycle_on_full(high_size=hi, low_size=lo, mode=mode)
                    except Exception:
                        pass
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent):
                            ag.replay_buffer = self.shared_replay
                    msg = f"[replay] loaded existing shared buffer size={len(self.shared_replay)} path={replay_path}"
                    if self.logger:
                        self.logger.log_text(msg)
                    else:
                        print(f"[INFO] {msg}")
                except Exception as e:
                    # 追加情報付き WARN （サイズ/mtime を表示し、破損なら自動隔離済みの可能性）
                    try:
                        st = os.stat(replay_path)
                        size = st.st_size
                        mtime = int(st.st_mtime)
                        print(f"[WARN] replay load failed path={replay_path} size={size} mtime={mtime} err={e}")
                    except Exception:
                        print(f"[WARN] replay load failed path={replay_path} err={e}")
                    # load が内部で空バッファ返却した場合 self.shared_replay は保持されたままなので再設定
                    if not isinstance(self.shared_replay, ReplayBuffer):
                        try:
                            # 破損隔離でファイルが移動された場合、新規空インスタンス生成
                            self.shared_replay = ReplayBuffer(maxlen=int(self.config.get('buffer_size', 50000)))
                            try:
                                if bool(self.config.get('replay_async_enabled', False)) and hasattr(self.shared_replay, 'enable_async'):
                                    qcap = int(self.config.get('replay_async_queue_maxsize', 50000) or 50000)
                                    self.shared_replay.enable_async(max_queue=qcap)
                            except Exception:
                                pass
                            try:
                                if bool(self.config.get('replay_cycle_on_full_enabled', False)) and hasattr(self.shared_replay, 'configure_cycle_on_full'):
                                    hi = int(self.config.get('replay_cycle_on_full_high_size', self.config.get('buffer_size', 250000)))
                                    lo = int(self.config.get('replay_cycle_on_full_low_size', 100000))
                                    mode = str(self.config.get('replay_cycle_shrink_mode', 'keep_newest') or 'keep_newest')
                                    self.shared_replay.configure_cycle_on_full(high_size=hi, low_size=lo, mode=mode)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        for ag in self.agents:
                            if isinstance(ag, AlphaZeroAgent):
                                ag.replay_buffer = self.shared_replay
                                # 旧推論専用モデル注入削除
                        if self.logger:
                            self.logger.log_text("[replay] started with empty buffer after load failure")
                        else:
                            print("[INFO] replay started empty after load failure")
        # 初期チェックポイント (空のリプレイと初期モデル) を要求された場合に保存
        # ただし、既存 ckpt からの再開時は二重保存を避けるためスキップ
        if self.config.get("initial_checkpoint_on_setup", False) and (not getattr(self, "_loaded_from_checkpoint", False)):
            try:
                self._save_checkpoint(version_tag=None)
                if self.logger:
                    self.logger.log_text("[init] initial checkpoint saved")
            except Exception as e:
                print(f"[WARN] initial checkpoint save failed: {e}")

    # 旧: 推論専用モデル同期は廃止（単一 self.model を使用）

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
            learner = AlphaZeroAgent(player_id=self.learning_player_id, model=self.model, config=self.config)
            if hasattr(learner, "logger"):
                learner.logger = self.logger
            if self.shared_replay is not None:
                learner.replay_buffer = self.shared_replay
            self.agents[self.learning_player_id] = learner
        for i in range(self.config["num_players"]):
            if i == self.learning_player_id:
                continue
            if not isinstance(self.agents[i], AlphaZeroAgent):
                repl = AlphaZeroAgent(player_id=i, model=self.model, config=self.config)
                if hasattr(repl, "logger"):
                    repl.logger = self.logger
                if self.shared_replay is not None:
                    repl.replay_buffer = self.shared_replay
                self.agents[i] = repl
        self.env.agents = self.agents
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
                if self.shared_replay is not None:
                    ag.replay_buffer = self.shared_replay

    # moved to utils.process_monitor.auto_replay_water_purge

    # -----------------------------------------------------
    # 自己対局 (データ収集)
    # -----------------------------------------------------
    def self_play(self, num_episodes: int = 1):
        return _orc.self_play(self, num_episodes=num_episodes)

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
        return _orc.train_concurrent(
            self,
            total_episodes=total_episodes,
            workers=workers,
            updates_per_iter=updates_per_iter,
            queue_maxsize=queue_maxsize,
            progress_print_every=progress_print_every,
        )

    # ---------------- 並列自己対局 ----------------
    def _self_play_parallel(self, num_episodes: int, workers: int):
        return _orc._self_play_parallel(self, num_episodes=num_episodes, workers=workers)

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
            # v4 特徴: 59N + 125。旧式 (56N+22) に固定していたため mismatch が発生していたので、
            # 現行モデルの in_features をそのまま利用してスナップショットを生成する。
            full_dim = None
            if use_full:
                try:
                    full_dim = int(getattr(self.model.backbone[0], 'in_features'))  # type: ignore[index]
                except Exception:
                    full_dim = None
            if use_full and full_dim is None:
                # それでも取得不能な場合は v4 期待式で推定 (警告付き)
                try:
                    est = 59 * int(self.config.get("num_players", 4)) + 125
                    print(f"[snapshot][WARN] full_dim 推定 fallback -> {est}")
                    full_dim = est
                except Exception:
                    pass
            try:
                snap = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device="cpu",
                    use_full_features=use_full,
                    full_feature_dim=full_dim if use_full else None,
                )
                snap.load_state_dict(self.model.state_dict(), strict=True)  # type: ignore[arg-type]
            except Exception as e_build:
                # 最終フォールバック: 既存モデルを deepcopy して CPU へ移動（互換重視 / shape mismatch 無視）
                print(f"[snapshot][WARN] 標準スナップショット再構築失敗 -> deepcopy fallback ({type(e_build).__name__}: {e_build})")
                snap = _cp.deepcopy(self.model)
                try:
                    snap.to('cpu')
                except Exception:
                    pass
            self.past_models.append(snap)
            self._previous_model = snap  # 互換
            # 上限超過なら古いものから削除
            if self.past_model_pool_size > 0 and len(self.past_models) > self.past_model_pool_size:
                overflow = len(self.past_models) - self.past_model_pool_size
                if overflow > 0:
                    del self.past_models[0:overflow]
            # ログ抑制設定
            try:
                do_log = bool(self.config.get("snapshot_log_enable", False))
            except Exception:
                do_log = True
            if do_log and self.logger:
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
        # Worker へ委譲（ロギング含む処理は worker 内で完結）
        sample_q = _queue.Queue()
        event_q = _queue.Queue()
        stop_ev = mp.Event()
        model_path = self.config.get("checkpoint_path", os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "policy_value_latest.pt"))
        worker = SelfplayDaemonWorker(
            worker_id=0,
            config=self.config,
            model_path=model_path,
            sample_queue=sample_q,
            event_queue=event_q,
            stop_event=stop_ev,
            control_queue=None,
            request_q=None,
            response_q=None,
        )
        # 既存インスタンス共有
        worker.env = self.env
        worker.agents = self.agents
        worker.model = self.model
        worker.device = getattr(self, "device", "cpu") if hasattr(self, "device") else "cpu"
        worker.max_steps = int(self.config.get("max_episode_steps", 1000) or 1000)
        worker.logger = self.logger  # ロガー注入
        # flush=False でサンプルは残し、ロギングのみ反映
        _ = worker.play_one_episode(flush=False)

    # -----------------------------------------------------
    # モデル学習 (ダミー)
    # -----------------------------------------------------
    def train_updates(self, num_updates: int = 1):
        start_time = time.time()
        last_print = 0
        successful_updates = 0  # 実際に損失が計算できた学習回数
        # --- 評価ゲート用: 学習前のベースラインモデルを保持（存在すれば ckpt、なければ現モデルのコピー） ---
        gate_enable = bool(self.config.get("eval_gate_enable", False))
        baseline_model = None
        if gate_enable:
            try:
                from agents.models import PolicyValueNet as _PVN
                # ベースラインは最新 ckpt
                base_ckpt = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                if os.path.exists(base_ckpt):
                    # resolve device 'auto' -> actual
                    def _resolve_device_str(dev_str: str | None) -> str:
                        if not dev_str or dev_str == "auto":
                            try:
                                import torch as _t
                                return "cuda" if _t.cuda.is_available() else "cpu"
                            except Exception:
                                return "cpu"
                        return dev_str
                    dev_str = _resolve_device_str(self.config.get("device", None))
                    baseline_model = _PVN.load(base_ckpt, map_location=dev_str)
                    # align device if needed
                    try:
                        if dev_str:
                            baseline_model.to(dev_str)  # type: ignore[arg-type]
                    except Exception:
                        pass
                else:
                    # フォールバック: 現在モデルの shallow コピー（同一参照回避）
                    import copy as _copy
                    baseline_model = _copy.deepcopy(self.model)
            except Exception as _e:
                print(f"[WARN] eval-gate baseline load failed: {_e}")
                baseline_model = None
        # 事前検証: 最初の学習行から val_* を必ず埋めるため、設定に関わらず一度実施
        if self.logger:
            try:
                vinfo0 = self.agents[0].validate_step(batch_size=self.config.get("val_batch_size") or self.config.get("batch_size", 256))
            except Exception:
                vinfo0 = {"policy_loss": None, "value_loss": None, "entropy": None}
            if isinstance(vinfo0, dict) and (vinfo0.get("policy_loss") is not None or vinfo0.get("value_loss") is not None):
                self.logger.log_validation(vinfo0)

        for i in range(num_updates):
            loss_info = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
            # ログ用にエポック情報を付与（存在する辞書に無害に追加）
            if isinstance(loss_info, dict):
                loss_info.setdefault("total_epochs", num_updates)
            # サンプル不足/偏り警告
            if loss_info.get("reason") == "no_data":
                print("[WARN] train_step skipped: no_data (consider increasing episodes or buffer)")
            else:
                if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                    successful_updates += 1
                # 共有バッファ存在時に学習プレイヤーサンプルの割合を軽くチェック
                if self.config.get("use_shared_replay", False) and self.shared_replay is not None:
                    total = len(self.shared_replay)
                    if total > 0:
                        own = sum(1 for _ in self.shared_replay.iter_all(owner_pid=self.learning_player_id))
                        ratio = own / total
                        if ratio < 0.15:  # しきい値は暫定
                            print(f"[WARN] low data share for learner: {own}/{total} ({ratio:.2%})")
            # 進捗出力: 動的バーは廃止し、間欠ログのみ
            if (i + 1) % self.config.get("log_interval", 50) == 0:
                _msg = f"[TRAIN] epoch={i+1}/{num_updates} loss={loss_info}"
                try:
                    if self.logger is not None:
                        self.logger.log_text(_msg, also_print=False)
                except Exception:
                    pass
                print(_msg)
            # 検証: 指定間隔で検証バッチの損失を測定しログへ（同一行に反映させるため、train行の直前に実施）
            try:
                # 既定は設定値（agents.config）に従う。0 明示設定で無効化可。
                val_every = int(self.config.get("val_eval_every_updates", 200))
            except Exception:
                val_every = 200
            if self.logger and val_every > 0 and ((i + 1) % val_every == 0):
                try:
                    vinfo = self.agents[0].validate_step(batch_size=self.config.get("val_batch_size") or self.config.get("batch_size", 256))
                except Exception:
                    vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None):
                    self.logger.log_validation(vinfo)

            # ロガーへ (train_step 内で既に push されている場合は二重記録を避ける)
            if self.logger and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                try:
                    payload = dict(loss_info)
                    payload["train_count"] = i + 1
                except Exception:
                    payload = loss_info
                self.logger.log_train(payload)
            # 追加: 各学習更新後に最新のゲート結果を毎回ログ（非並行モード用）
            try:
                thr_val = float(self.config.get("eval_gate_threshold", self._last_gate_threshold)) if hasattr(self, "_last_gate_threshold") else 0.6
            except Exception:
                thr_val = 0.6
            try:
                last_res = getattr(self, "_last_gate_result", None)
                if self.logger and isinstance(last_res, dict) and (last_res.get("win_rate") is not None):
                    wr = float(last_res.get("win_rate", 0.0))
                    msg_gate = f"[GATE] win_rate={wr:.2%} threshold={thr_val:.0%} result={'ACCEPT' if wr >= thr_val else 'REJECT'}"
                    self.logger.log_text(msg_gate)
            except Exception:
                pass

            # 即時 checkpoint + purge オプション (単独 train_updates 用 / concurrent とは別経路)
            if self.config.get('purge_replay_after_each_update'):
                try:
                    # purge は _save_checkpoint 内の設定に依存 (purge_replay_after_checkpoint)
                    # 事前にサイズ計測
                    old_size = None
                    try:
                        if self.shared_replay is not None:
                            old_size = len(self.shared_replay)
                        else:
                            ag0 = self.agents[0]
                            rb = getattr(ag0, 'replay_buffer', None)
                            if rb is not None and hasattr(rb, '__len__'):
                                old_size = len(rb)
                    except Exception:
                        old_size = None
                    self._save_checkpoint()
                    if self.logger:
                        self.logger.log_text(f"[replay] immediate_purge_after_update(train_updates) prev_size={old_size}")
                except Exception as e:
                    print(f"[WARN] immediate checkpoint after update failed: {e}")
        # --- 学習後: ゲート評価（有効時） ---
        gated_pass = True
        gate_result = None
        if gate_enable and (self.model is not None) and (baseline_model is not None):
            try:
                from evaluation.gating import evaluate_candidate
                games = int(self.config.get("eval_gate_games", 20) or 20)
                thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                seed = self.config.get("eval_gate_seed", None)
                gate_result = evaluate_candidate(self.model, baseline_model, self.config, games=games, seed=seed)
                gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                msg = f"[GATE] win_rate={gate_result.get('win_rate'):.2%} threshold={thr:.0%} result={'ACCEPT' if gated_pass else 'REJECT'}"
                # 最新ゲート結果を保持
                try:
                    self._last_gate_threshold = float(thr)
                    self._last_gate_result = dict(gate_result)
                except Exception:
                    pass
                if self.logger:
                    self.logger.log_text(msg)
                else:
                    print(msg)
            except Exception as e:
                print(f"[WARN] eval-gate failed to run: {e}")
                gated_pass = True  # フォールバックで通す

        # 許可された場合のみ保存。拒否ならロールバックして旧モデルを維持
        if not gated_pass and baseline_model is not None:
            try:
                # ロールバック
                self.model = baseline_model
                # 学習プレイヤーへも反映
                learner = self.agents[self.learning_player_id]
                if isinstance(learner, AlphaZeroAgent):
                    learner.set_model(self.model)
                if self.logger:
                    self.logger.log_text("[GATE] reverted to baseline model")
            except Exception as e:
                print(f"[WARN] revert to baseline failed: {e}")

        # 保存実行（gated_pass に応じて self.model は適切な方がセットされている）
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
        # 終了サマリを events.log に出力
        try:
            if self.logger:
                extra = ''
                try:
                    if gate_result is not None:
                        extra = f" gate_win_rate={gate_result.get('win_rate'):.4f} gate_games={gate_result.get('games')}"
                except Exception:
                    extra = ''
                self.logger.log_text(f"[summary] train_updates_requested={num_updates} train_updates_successful={successful_updates}{extra}")
                # バッファをフラッシュして確実にディスクへ書き込む
                if hasattr(self.logger, 'flush_buffers'):
                    self.logger.flush_buffers(force=True)
        except Exception:
            pass

    

    # -----------------------------------------------------
    # チェックポイント
    # -----------------------------------------------------
    def _save_checkpoint(self):  # backward compatibility name
        self._save_checkpoint(version_tag=None)

    def _save_checkpoint(self, version_tag: str | None = None, *, model_path_override: str | None = None, skip_model_save: bool = False):
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
        latest_path = self.config["checkpoint_path"] if model_path_override is None else model_path_override
        if (not skip_model_save) and self.model is not None:
            _atomic_save(latest_path, lambda p: self.model.save(p, force_sync=True))
        # バージョン付き保存
        if version_tag:
            ver_path = os.path.join(self.config["checkpoint_dir"], f"policy_value_{version_tag}.pt")
            if (not skip_model_save) and self.model is not None:
                _atomic_save(ver_path, lambda p: self.model.save(p, force_sync=True))
        # リプレイ保存: 共有モードなら共有バッファを保存、そうでなければ従来通り代表エージェント
        replay_path = self.config.get("replay_path", "replay_buffer.joblib")
        purge_after_ckpt = bool(self.config.get("purge_replay_after_checkpoint", False))
        if self.shared_replay is not None:
            try:
                # フル特徴量モードで legacy 混入していたらフィルタ
                if self.config.get('use_full_features'):
                    try:
                        before = len(self.shared_replay)
                        # 共有リプレイ内部構造アクセス (互換維持: data 属性が存在しない将来版対策)
                        data_attr = getattr(self.shared_replay, '_data', None)
                        if data_attr is not None:
                            # _data が deque の場合は直接フィルタせず (コスト高)、ここではスキップ
                            pass
                        else:
                            # 旧バージョン data フィールド互換 (安全性低いため try 内で)
                            self.shared_replay.data = [s for s in self.shared_replay.data if s.get('feature_version',1)==1]  # type: ignore[attr-defined]
                        after = len(self.shared_replay)
                        if after < before:
                            print(f"[INFO] shared replay purge legacy {before-after} samples (full mode)")
                    except Exception:
                        pass
                self.shared_replay.save(replay_path, purge=purge_after_ckpt)
                if purge_after_ckpt:
                    if self.logger:
                        try:
                            self.logger.log_text(f"[replay] checkpoint_saved_and_purged prev_size={before} new_size=0 path={replay_path}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WARN] shared replay save failed: {e}")
        else:
            try:
                # 単独エージェント経由保存
                ag0 = self.agents[0]
                if hasattr(ag0, 'save_replay'):
                    prev_size = None
                    try:
                        rb_local = getattr(ag0, 'replay_buffer', None)
                        if rb_local is not None and hasattr(rb_local, '__len__'):
                            prev_size = len(rb_local)
                    except Exception:
                        prev_size = None
                    ag0.config['purge_replay_after_save'] = purge_after_ckpt
                    ag0.save_replay(replay_path)
                    if purge_after_ckpt and self.logger:
                        self.logger.log_text(f"[replay] agent_replay_saved_and_purged prev_size={prev_size} path={replay_path}")
            except Exception as e:
                print(f"[WARN] agent replay save failed: {e}")
        # Optimizer 保存（学習エージェントの Adam 状態）
        try:
            ag0 = self.agents[self.learning_player_id]
            opt_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'optimizer_latest.pt')
            def _save_opt(pth):
                if hasattr(ag0, 'save_optimizer'):
                    ag0.save_optimizer(pth)
                else:
                    import torch as _t
                    opt = getattr(ag0, '_optimizer', None)
                    if opt is not None:
                        _t.save(opt.state_dict(), pth)
            _atomic_save(opt_path, _save_opt)
        except Exception as e:
            print(f"[WARN] optimizer save failed: {e}")
        # Scheduler 保存（学習率スケジューラの状態）
        try:
            ag0 = self.agents[self.learning_player_id]
            # スケジューラが存在する場合のみ保存
            if hasattr(ag0, '_scheduler') and getattr(ag0, '_scheduler') is not None:
                sch_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'scheduler_latest.pt')
                def _save_sch(pth):
                    if hasattr(ag0, 'save_scheduler'):
                        ag0.save_scheduler(pth)
                _atomic_save(sch_path, _save_sch)
        except Exception as e:
            print(f"[WARN] scheduler save failed: {e}")
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
    # Allow overriding checkpoint path via environment for quick experiments.
    # If ALPHA_ZERO_CHECKPOINT_PATH is set to empty string, trainer will build a fresh model.
    try:
        env_ckpt = os.environ.get('ALPHA_ZERO_CHECKPOINT_PATH', None)
        if env_ckpt is not None:
            cfg['checkpoint_path'] = env_ckpt
    except Exception:
        pass
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
    # 非並行モードは廃止し、常に並行モードで実行
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
    #print("[INFO] checkpoints ->", cfg.get("checkpoint_path"))
    #print("[INFO] replay ->", cfg.get("replay_path"))
