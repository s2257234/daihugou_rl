"""AlphaZero-like agent with PUCT MCTS and variable-length action handling.

本実装での報酬設計:
    - フェーズ (誰かが新たに上がるまでの区間) ごとに、
            その区間で『最初に上がったプレイヤー』= 1, その時点でまだ残っていた他プレイヤー = 0
            既に以前に上がっていたプレイヤーは評価対象外
    - 最終順位ベースの後付け報酬は付与しない (次に上がる確率を直接教師ラベル化)

Implements:
    - Replay samples as dict {state, legal_actions, pi, value or None(未確定)}
    - Action set taken directly from MCTS root children order (variable length)
    - Temperature schedule (early high exploration -> later low)
    - train_step with policy/value losses + entropy regularization (value が None のサンプルは除外)
"""
from __future__ import annotations

import random
import copy
from collections import deque
from typing import Any, Dict, List, Optional, Iterable
import joblib
import time
import uuid

try:  # model
    from agents.models import PolicyValueNet  # noqa: F401
except Exception:  # pragma: no cover
    PolicyValueNet = Any  # type: ignore

try:  # config
    from agents.config import ALPHA_ZERO_CONFIG
except Exception:  # pragma: no cover
    ALPHA_ZERO_CONFIG = {
        "num_simulations": 64,
        "puct_c": 1.4,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "temperature": 1.0,
        "buffer_size": 50000,
        "checkpoint_path": "checkpoints/policy_value_latest.pt",
        "replay_path": "replay_buffer.joblib",
    }

from agents.mcts import run_puct_mcts, PUCTNode

from agents.agent_utills.mcts_runner import run_mcts as _run_mcts_external
from agents.agent_utills.action_guard import clamp_action_to_env as _clamp_action_to_env
from agents.agent_utills.feature_extractor import extract_state as _extract_state_external
from agents.agent_utills.mcts_tt import (
    AlphaZeroTTView,
    apply_from_det_pool as _apply_from_det_pool_impl,
    build_single_determinization as _build_single_determinization_impl,
    determinization_worker as _determinization_worker_impl,
    inline_determinize as _inline_determinize_impl,
    maybe_start_det_pool as _maybe_start_det_pool_impl,
    shutdown_det_pool as _shutdown_det_pool_impl,
)
from utils.policy import softmax_temperature_policy

from agents.replay_buffer import store_replay_sample as _store_replay_sample
from agents.replay_buffer import canonicalize_state as _canonicalize_state_impl

# train_step / value encoding / DataLoader helpers are delegated to keep this file smaller.
# Backward compatibility: keep names importable from agents.drl_agent.
from agents.agent_utills.training import (
    VALUE_U8_NONE,
    _decode_value_u8,
    _extract_value,
    _has_value_label,
    ReplaySnapshotDataset,
    RawSampleDataset,
    PreselectedSampler,
    collate_prepared,
    collate_prepare_batch,
    TrainStepMixin,
    InferenceClient,
    LocalInferenceClient,
    RemoteInferenceClient,
)


def _is_pass_only(legal_actions) -> bool:
    if legal_actions is None:
        return False
    try:
        return all(a is None for a in legal_actions)
    except Exception:
        return False


def _choose_opening_action(agent: "AlphaZeroAgent", actions: List[Any], pi_action: List[float], training: bool) -> Any:
    opening_random_enable = bool(agent.config.get("opening_random_enable", False))
    opening_random_moves = int(agent.config.get("opening_random_moves", 0))
    opening_include_pass = bool(agent.config.get("opening_random_include_pass", False))
    use_opening_random = (
        training
        and opening_random_enable
        and opening_random_moves > 0
        and agent.move_count < opening_random_moves
    )
    if use_opening_random and actions:
        random_pool = actions
        if not opening_include_pass:
            filtered = [a for a in actions if a != "pass"]
            if filtered:
                random_pool = filtered
        return random.choice(random_pool) if random_pool else "pass"
    return random.choices(actions, weights=pi_action, k=1)[0] if actions else "pass"


class AlphaZeroAgent(TrainStepMixin):
    """大富豪用 AlphaZero 風エージェント。

    主機能:
      - MCTS(PUCT) により行動方策分布(pi) を推定
      - フェーズ勝利確率 value を同時学習 (BCE)
      - リプレイバッファへ (状態, 方策, valueラベル) を蓄積
    """

    # 安全デフォルト（マルチプロセスでの未初期化参照対策）
    _dup_enabled = False

    def __init__(self, player_id: int, model=None, config=None):
        # --- 基本設定 / ハイパーパラメータ ---
        self.player_id = player_id
        self.config = config or ALPHA_ZERO_CONFIG
        self.model = model
        
        # Device 設定（model から取得、なければ config、なければ cpu）
        if model is not None and hasattr(model, 'device'):
            self.device = model.device
        else:
            import torch
            device_str = config.get('device', 'cpu') if isinstance(config, dict) else 'cpu'
            # Allow 'auto' in config: prefer CUDA if available, otherwise CPU
            if isinstance(device_str, str) and device_str.lower() == 'auto':
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device_str)
        
        # --- 推論エンジンの初期化 (遅延初期化: 実際の使用時に確定) ---
        self._inference_client: Optional[InferenceClient] = None

        # リプレイバッファ (共有 or ローカル deque)。共有時は trainer から差し込まれる想定。
        # 非共有の場合は O(1) で先頭追い出しが可能な deque(maxlen) を用いる。
        self.max_buffer_size = self.config.get("buffer_size", 50000)
        self._use_shared = self.config.get("use_shared_replay", False)
        if not self._use_shared:
            # ローカル: deque (FIFO 自動エビクション)
            self.replay_buffer = deque(maxlen=self.max_buffer_size)
        else:
            # 共有バッファは trainer 側で後からセットされる。ここでは空の list プレースホルダ。
            self.replay_buffer = []

        # MCTS 関連パラメータ
        self.num_simulations = self.config.get("num_simulations", 64)
        self.puct_c = self.config.get("puct_c", 1.4)
        self.dirichlet_alpha = self.config.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = self.config.get("dirichlet_epsilon", 0.25)
        # バッチ推論/ルート並列用設定
        self.mcts_batch_eval_size = int(self.config.get("mcts_batch_eval_size", 1))
        self.enable_mcts_tt = bool(self.config.get("enable_mcts_tt", False))
        self.mcts_tt_capacity = int(self.config.get("mcts_tt_capacity", 10000))
        self._mcts_tt = None  # lazy init LRU 風辞書

        # 温度スケジュール (序盤探索重視 → 後半確定的)
        self.temperature = self.config.get("temperature", 1.0)
        self.temperature_low = self.config.get("temperature_low", 0.05)
        self.temperature_decay_moves = self.config.get("temperature_decay_moves", 20)
        self.move_count = 0  # エピソード内手数カウンタ
        # 自己対戦エピソードカウンタ（温度スケジュール短縮用）
        self.episodes_played = 0
        # 学習関連
        self.lr = self.config.get("lr", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.policy_loss_coef = self.config.get("policy_loss_coef", 1.0)
        self.value_loss_coef = self.config.get("value_loss_coef", 1.0)
        self.entropy_coef = self.config.get("entropy_coef", 1e-3)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self._optimizer = None  # 遅延初期化
        self._scheduler = None  # 学習率スケジューラ (遅延初期化)
        self._update_step = 0    # スケジューラ同期用の更新ステップ数
        # モデル世代 (Trainer 側で更新される想定)。データ多様性確保用にサンプルへ埋め込む。
        self.model_version = config.get("current_model_version", 0) if isinstance(config, dict) else 0

        # フェーズ中サンプル保持 (フェーズ確定時にラベル付与)
        self._phase_samples: List[Any] = []
        self._phase_value_preds: List[Optional[float]] = []
        # worker_zero_buffer モード用: エピソード内で確定したサンプルを一時保持
        self._episode_confirmed_samples: List[Dict[str, Any]] = []
        self.env_ref = None  # 直近参照環境
        self.logger = None   # 外部ロガー (TensorBoard 等)
        self._logged_inside = False  # 二重記録防止
        self._mcts_params_logged = False  # MCTSパラメータログ出力フラグ（初回のみ）
        # インクリメンタル Belief 用キャッシュ
        # 形式: {
        #   'num_players': int,
        #   'poss_sets': {pid: set(card_idx)},
        #   'hist_len': int,  # 適用済み action_history 長
        # }
        self._belief_cache = None
        # 直近状態（差分用ヒント）
        self._last_state = None

        # --- 学習中の統計・動的重み付け関連 ---
        self.total_value_samples = 0
        self.total_positive = 0
        self.phase_total = 0
        self.phase_correct = 0
        self.episode_phase_total = 0
        self.episode_phase_correct = 0
        self.lost_phase_samples = 0
        # バッチ推論使用フラグ（学習フェーズで1回だけログ出力するため）
        self._batch_inference_used = False
        self.tt_hits = 0
        self.tt_misses = 0
        self.pos_weight = float(self.config.get("value_pos_weight", 1.0))
        # pos_weight 調整ログのスロットリング用
        self._posw_last_log_samples = 0
        self._posw_last_log_value = float(self.pos_weight)
        # クラス別ミックスの過学習対策: 直近使用した陽性UIDを避けるバッファ
        try:
            from collections import deque as _dq
            self._recent_pos_uids = _dq(maxlen=int(self.config.get('value_pos_recent_buffer', 2000)))
        except Exception:
            self._recent_pos_uids = []

        # --- MCTS 実効シミュレーション統計（低頻度ログ用に集計） ---
        try:
            from collections import deque as _dq
            self._mcts_sims_recent = _dq(maxlen=4096)
        except Exception:
            self._mcts_sims_recent = []  # フォールバック
        self._mcts_moves = 0
        self._mcts_early_stop = 0
        # MCTS パフォーマンス計測用（1手あたりのNN推論時間の集計）
        self._perf_infer_ms_accum = 0.0
        self._perf_infer_calls = 0
        self._forward_time_ms_ema = None
        # ルートノードに属性を付けられない(__slots__)場合のフォールバック保持
        self._last_perf_clone_ms: Optional[float] = None
        self._last_perf_det_mode: Optional[str] = None

        # --- 重複サンプルフィルタ構造 (シグネチャ頻度カウント) ---
        self._dup_enabled = bool(self.config.get('enable_duplicate_filter', False))
        if self._dup_enabled:
            from collections import deque as _dq
            self._dup_sig_queue = _dq(maxlen=int(self.config.get('duplicate_window_size', 5000) or 5000))
            self._dup_sig_counts = {}
            self._dup_skipped = 0
            self._dup_kept = 0
            self._dup_last_log = 0
        else:
            self._dup_sig_queue = None
            self._dup_sig_counts = None
            self._dup_skipped = 0
            self._dup_kept = 0
            self._dup_last_log = 0

        # 行動履歴 (determinization 制約用): list of dict {pid, action, field_before, revo}
        self._action_history: List[Dict[str, Any]] = []

        # ---- 並列 determinization プール関連 (lazy init) ----
        self._det_pool = None            # deque of determinization assignments
        self._det_pool_lock = None       # threading.Lock
        self._det_pool_thread = None     # worker thread
        self._det_stop_event = None      # threading.Event
        self._det_cfg = {
            'capacity': int(self.config.get('det_pool_capacity', 128)),
            'refill_ratio': float(self.config.get('det_pool_refill_threshold', 0.3)),
            'workers': int(self.config.get('det_workers', 1)),  # 未来拡張 (現状 1 のみ)
            'sampling': str(self.config.get('det_pool_sampling', 'fifo')),  # fifo|random
            'retry_max': int(self.config.get('det_retry_max', 8)),
        }
        self._det_stats = {
            'generated': 0,
            'pool_hits': 0,
            'pool_fallback_inline': 0,
            'discard_mismatch': 0,
            'retries_total': 0,
        }

        # ---- 推論結果の再利用（評価用余分な forward 削減） ----
        self._capture_eval_value = False
        self._last_eval_value: Optional[float] = None
        # フォールバックログの重複抑制
        self._fallback_logged = set()

    # ---------------- Batch Preparation ----------------
    def _prepare_batch(self, batch: List[Dict[str, Any]], uid2weight: Optional[Dict[int, float]] = None, fast_full_input_np=None) -> Dict[str, Any]:
        """バッチ前処理は [agents/agent_utills/training.py](agents/agent_utills/training.py) に委譲。"""
        return TrainStepMixin._prepare_batch(self, batch, uid2weight=uid2weight, fast_full_input_np=fast_full_input_np)

    def _canonicalize_state(self, st: Optional[dict]) -> dict:
        """Canonicalize state; implementation lives in agents/replay_buffer.py."""
        return _canonicalize_state_impl(self, st)

    def _log_fallback_once(self, key: str, msg: str, exc: Exception | None = None) -> None:
        try:
            if key in self._fallback_logged:
                return
            self._fallback_logged.add(key)
        except Exception:
            pass
        try:
            text = f"{msg} ({type(exc).__name__}: {exc})" if exc is not None else msg
            if getattr(self, 'logger', None):
                self.logger.log_text(text)
            else:
                print(text)
        except Exception:
            pass

    # ---------------- Public API ----------------
    def set_model(self, model):
        """後から学習済みモデルを差し替える."""
        self.model = model
        # モデル更新時は TT を無効化（古い出力を使わないように）
        try:
            if self._mcts_tt is not None:
                self._mcts_tt.clear()
                self._mcts_tt_tick = 0
        except Exception:
            pass

    # 旧: 推論専用モデルは廃止（self.model を常に使用）

    def set_model_version(self, version: int):
        """Trainer 側からモデル世代を更新するフック（TTは世代で分離）."""
        try:
            self.model_version = int(version)
        except (ValueError, TypeError):
            # 変換できない場合はそのまま設定
            self.model_version = version

    def set_env_ref(self, env):
        """環境参照をセット (select_action が obs だけ来た場合に使用)。"""
        self.env_ref = env

    def select_action(self, env_or_obs, training: bool = True, **kwargs):
        """現在手番で行動を選択し、リプレイサンプルをバッファへ格納。

        1) MCTS 実行 → ルート子ノードの訪問回数取得
        2) 温度付き softmax で π 計算
        3) 方策サンプルを保存 (value は未確定 None)
        4) 選択行動を環境へ返す
        """
        # 学習/推論モードをインスタンス変数に保存（_get_legal_actionsで参照）
        self._current_training_mode = training
        
        # Resolve env
        if hasattr(env_or_obs, "game"):
            env = env_or_obs
            self.env_ref = env
        else:
            env = getattr(self, "env_ref", None)

        # pass-only: if only pass is legal, store sample and return immediately.
        # (Changed: previously skipped sample storage for pass-only situations,
        #  but we need these samples for training to learn value in forced-pass positions)
        pass_only_detected = False
        try:
            if env is not None and hasattr(env, "_generate_legal_actions"):
                g0 = env.game
                cur0 = g0.players[g0.turn]
                hand0 = cur0.hand
                field0 = (g0.current_field or [])[:]
                legal0 = env._generate_legal_actions(hand0, field0)
                if _is_pass_only(legal0):
                    pass_only_detected = True
                    if training:
                        # Store pass-only sample with legal_actions=[None], pi=fixed-length 1059 (PASS=1.0), value=None
                        state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))
                        # Create fixed-length pi with PASS action at index corresponding to 'PASS' key
                        L = 1059
                        pi_full_pass = [0.0] * L
                        # Find PASS index in canonical keys
                        if hasattr(self.model, 'canonical_action_keys'):
                            try:
                                canonical = self.model.canonical_action_keys()
                                pass_idx = canonical.index('PASS') if 'PASS' in canonical else (L - 1)
                                pi_full_pass[pass_idx] = 1.0
                            except Exception:
                                pi_full_pass[-1] = 1.0  # Fallback: assume PASS is last
                        else:
                            pi_full_pass[-1] = 1.0  # Fallback: assume PASS is last
                        stored = self._store_sample(state_repr, [None], pi_full_pass, None)
                        self._phase_samples.append(stored)
                        # PASSのみの場合は MCTS/NN 推論を省略して即座にパスする。
                        # 予測値は作らない（教師データではなく統計用だが、混同・汚染を避ける）。
                        self._phase_value_preds.append(None)
                        self._last_state = state_repr
                    # pass-only は正常な最適化動作なのでログ不要
                    return None
                # 合法手が1つだけ（パス以外）の場合も探索をスキップ
                elif legal0 and len(legal0) == 1:
                    single_action = legal0[0]
                    if single_action is not None:
                        if training:
                            # Store single-action sample with fixed-length 1059
                            try:
                                serialized_legal = [[str(c) for c in single_action]]
                            except Exception:
                                serialized_legal = [single_action]
                            state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))
                            # Create fixed-length pi with single action at 1.0
                            L = 1059
                            pi_full_single = [0.0] * L
                            if hasattr(self.model, 'canonical_action_keys'):
                                try:
                                    from agents.agent_utills.feature_extractor import CardEncoder
                                    ce = CardEncoder()
                                    parts = [int(ce.card_index(c)) for c in single_action]
                                    parts.sort()
                                    action_key = '|'.join(str(p) for p in parts)
                                    canonical = self.model.canonical_action_keys()
                                    action_idx = canonical.index(action_key) if action_key in canonical else 0
                                    pi_full_single[action_idx] = 1.0
                                except Exception:
                                    pi_full_single[0] = 1.0  # Fallback
                            else:
                                pi_full_single[0] = 1.0  # Fallback
                            stored = self._store_sample(state_repr, serialized_legal, pi_full_single, None)
                            self._phase_samples.append(stored)
                            # バリューは計算しない（学習時の正解ラベルはゲーム終了時の勝敗を使うため）
                            self._phase_value_preds.append(None)
                            self._last_state = state_repr
                        # 合法手が1つだけの場合は正常な最適化動作なのでログ不要
                        # 探索をスキップしてその合法手を返す
                        return single_action
        except Exception as e:
            self._log_fallback_once(
                "pre_mcts_legal_check_exception",
                "[az-fallback] pre-MCTS legal check failed; continuing",
                e
            )

        # If env is unavailable, fall back to pass or a legal action (when provided).
        if env is None:
            legal_actions_kw = kwargs.get("legal_actions")
            if _is_pass_only(legal_actions_kw):
                if training:
                    # Cannot extract state without env, skip sample storage
                    pass
                self._log_fallback_once(
                    "env_missing_pass_only",
                    f"[az-fallback] env is None; pass-only legal actions -> return pass (training={training})"
                )
                return None
            # 合法手が1つだけ（パス以外）の場合はその手を返す
            if legal_actions_kw and len(legal_actions_kw) == 1:
                single_action_kw = legal_actions_kw[0]
                if single_action_kw is not None:
                    self._log_fallback_once(
                        "env_missing_single_action",
                        f"[az-fallback] env is None; single legal action -> return it (training={training})"
                    )
                    try:
                        return [str(c) for c in single_action_kw]
                    except Exception:
                        return single_action_kw
            if legal_actions_kw is not None:
                self._log_fallback_once(
                    "env_missing_choose_legal",
                    f"[az-fallback] env is None; choosing from legal_actions without MCTS (training={training})"
                )
                for cand in legal_actions_kw:
                    if cand is None:
                        continue
                    try:
                        return [str(c) for c in cand]
                    except Exception:
                        return cand
            return None

        # If env.step provided legal_actions and it is pass-only, skip MCTS and pass.
        legal_actions_kw = kwargs.get("legal_actions")
        if _is_pass_only(legal_actions_kw):
            if training and not pass_only_detected:
                # Store pass-only sample with fixed-length 1059 (PASS=1.0)
                state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))
                L = 1059
                pi_full_pass = [0.0] * L
                if hasattr(self.model, 'canonical_action_keys'):
                    try:
                        canonical = self.model.canonical_action_keys()
                        pass_idx = canonical.index('PASS') if 'PASS' in canonical else (L - 1)
                        pi_full_pass[pass_idx] = 1.0
                    except Exception:
                        pi_full_pass[-1] = 1.0
                else:
                    pi_full_pass[-1] = 1.0
                stored = self._store_sample(state_repr, [None], pi_full_pass, None)
                self._phase_samples.append(stored)
                # PASSのみの場合は MCTS/NN 推論を省略して即座にパスする。
                # 予測値は作らない（教師データではなく統計用だが、混同・汚染を避ける）。
                self._phase_value_preds.append(None)
                self._last_state = state_repr
            self._log_fallback_once(
                "pass_only_skip_mcts",
                f"[az-fallback] pass-only legal_actions hint; skip MCTS (training={training})"
            )
            return None
        
        # 合法手が1つだけ（パス以外）の場合も探索をスキップ
        if legal_actions_kw and len(legal_actions_kw) == 1:
            single_action_kw = legal_actions_kw[0]
            if single_action_kw is not None:
                if training and not pass_only_detected:
                    # Store single-action sample with fixed-length 1059
                    try:
                        serialized_legal = [[str(c) for c in single_action_kw]]
                    except Exception:
                        serialized_legal = [single_action_kw]
                    state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))
                    L = 1059
                    pi_full_single = [0.0] * L
                    if hasattr(self.model, 'canonical_action_keys'):
                        try:
                            from agents.agent_utills.feature_extractor import CardEncoder
                            ce = CardEncoder()
                            parts = [int(ce.card_index(c)) for c in single_action_kw]
                            parts.sort()
                            action_key = '|'.join(str(p) for p in parts)
                            canonical = self.model.canonical_action_keys()
                            action_idx = canonical.index(action_key) if action_key in canonical else 0
                            pi_full_single[action_idx] = 1.0
                        except Exception:
                            pi_full_single[0] = 1.0
                    else:
                        pi_full_single[0] = 1.0
                    stored = self._store_sample(state_repr, serialized_legal, pi_full_single, None)
                    self._phase_samples.append(stored)
                    # バリューは計算しない（学習時の正解ラベルはゲーム終了時の勝敗を使うため）
                    self._phase_value_preds.append(None)
                    self._last_state = state_repr
                self._log_fallback_once(
                    "single_action_skip_mcts",
                    f"[az-fallback] single legal action hint; skip MCTS (training={training})"
                )
                # 探索をスキップしてその合法手を返す
                return single_action_kw

        # デバッグ: MCTS実行前の環境状態を記録
        _pre_mcts_revo = None
        _pre_mcts_field = None
        _pre_mcts_turn = None
        if self.config.get('debug_action_mismatch', False):
            try:
                _pre_mcts_revo = bool(getattr(getattr(env.game, 'rule_checker', None), 'revolution', False))
                _pre_mcts_field = [str(c) for c in (getattr(env.game, 'current_field', []) or [])]
                _pre_mcts_turn = getattr(env.game, 'turn', -1)
            except Exception as e:
                self._log_fallback_once(
                    "debug_action_mismatch_pre_exception",
                    "[az-fallback] debug_action_mismatch(pre) failed; continuing",
                    e
                )

        # ---- MCTS 実行 ----
        if not training:
            # 評価時はルート状態のNN出力を次ステップで再利用できるようキャプチャ
            self._capture_eval_value = True
            self._last_eval_value = None
        else:
            # 学習時はキャプチャ不要
            self._capture_eval_value = False
            self._last_eval_value = None
        import time
        
        # 推論時の複数回手札サンプリング（Multiple Determinization Averaging）
        multi_det_count = 1
        if not training:
            multi_det_count = max(1, int(self.config.get("inference_multi_determinization", 1)))
        
        if multi_det_count > 1 and not training:
            # 複数回MCTS実行して訪問回数を統合
            mcts_start = time.time()
            aggregated_visits = {}  # {action: total_visit_count}
            aggregated_q_sum = {}   # {action: total_q_value_sum}
            aggregated_q_count = {} # {action: count for averaging}
            
            for det_idx in range(multi_det_count):
                # 各反復でMCTSを実行（determinizationが有効なら異なる手札サンプリング）
                root_iter = self._run_mcts(env, training=False)
                
                # 訪問回数とQ値を集約
                for action, child in root_iter.children.items():
                    if action not in aggregated_visits:
                        aggregated_visits[action] = 0
                        aggregated_q_sum[action] = 0.0
                        aggregated_q_count[action] = 0
                    
                    aggregated_visits[action] += child.visit_count
                    # Q値を重み付き平均するため、訪問回数 × Q値の合計を保持
                    aggregated_q_sum[action] += child.value * child.visit_count
                    aggregated_q_count[action] += child.visit_count
            
            # 最初の実行のrootノードを使って統合結果を格納
            root = self._run_mcts(env, training=False)  # 最後の1回を再利用してもよいが、新規作成
            
            # root.childrenを統合結果で上書き
            for action in aggregated_visits.keys():
                if action not in root.children:
                    # 新しいノードを作成（既存のノードがない場合）
                    prior = 1.0 / max(1, len(aggregated_visits))
                    to_play = getattr(root, 'to_play', getattr(env.game, 'turn', self.player_id))
                    root.children[action] = PUCTNode(prior=prior, parent=root, action=action, to_play=to_play)
                
                # 訪問回数を統合値に上書き
                root.children[action].visit_count = aggregated_visits[action]
                
                # value_sumを設定（value = value_sum / visit_count で計算される）
                if aggregated_q_count[action] > 0:
                    avg_q_value = aggregated_q_sum[action] / aggregated_q_count[action]
                    root.children[action].value_sum = avg_q_value * aggregated_visits[action]
            
            # 統合に使われなかったchildrenを削除
            actions_to_remove = []
            for action in root.children.keys():
                if action not in aggregated_visits:
                    actions_to_remove.append(action)
            for action in actions_to_remove:
                del root.children[action]
            
            mcts_time = time.time() - mcts_start
            
            # パフォーマンスログ（複数サンプリング時）
            if mcts_time > 1.0:
                if not hasattr(self, '_mcts_log_counter'):
                    self._mcts_log_counter = 0
                self._mcts_log_counter += 1
                if self._mcts_log_counter % 10 == 0:
                    print(f"[PERF] AlphaZero Multi-Det MCTS: {mcts_time:.2f}s (sims={self.num_simulations}, runs={multi_det_count})")
        else:
            # 通常の1回実行
            
            root = self._run_mcts(env, training=training)
            

        # デバッグ: 環境コピー時のズレを検証
        if self.config.get('debug_action_mismatch', False):
            try:
                # MCTS実行後の実環境の状態を取得
                _post_mcts_revo = bool(getattr(getattr(env.game, 'rule_checker', None), 'revolution', False))
                _post_mcts_field = [str(c) for c in (getattr(env.game, 'current_field', []) or [])]
                _post_mcts_turn = getattr(env.game, 'turn', -1)
                
                # 実環境の変更を検出（MCTSは元環境を変更してはいけない）
                if _pre_mcts_revo != _post_mcts_revo:
                    print(f"[ENV-MUTATED-REVO] pid={self.player_id} before={_pre_mcts_revo} after={_post_mcts_revo}")
                if _pre_mcts_field != _post_mcts_field:
                    print(f"[ENV-MUTATED-FIELD] pid={self.player_id} before={_pre_mcts_field} after={_post_mcts_field}")
                if _pre_mcts_turn != _post_mcts_turn:
                    print(f"[ENV-MUTATED-TURN] pid={self.player_id} before={_pre_mcts_turn} after={_post_mcts_turn}")
                
                # MCTSで使用した環境コピーの状態と実環境を比較
                if root is not None:
                    _copy_revo = getattr(root, '_debug_env_copy_revolution', None)
                    _copy_field = getattr(root, '_debug_env_copy_field', None)
                    
                    if _copy_revo is not None and _copy_revo != _pre_mcts_revo:
                        print(f"[ENV-COPY-MISMATCH-REVO] pid={self.player_id} real={_pre_mcts_revo} copy={_copy_revo}")
                    if _copy_field is not None and _copy_field != _pre_mcts_field:
                        print(f"[ENV-COPY-MISMATCH-FIELD] pid={self.player_id} real={_pre_mcts_field} copy={_copy_field}")
            except Exception as e:
                self._log_fallback_once(
                    "debug_action_mismatch_post_exception",
                    "[az-fallback] debug_action_mismatch(post) failed; continuing",
                    e
                )

        # --- Root children hard alignment with real env legal actions ---
        try:
            real_root_legal = self._get_legal_actions(env)
        except Exception:
            real_root_legal = None
        if real_root_legal is not None:
            try:
                # リストはhashableではないため、tuple化してsetに変換
                real_set = set()
                for act in real_root_legal:
                    if act == "pass":
                        real_set.add("pass")
                    elif isinstance(act, list):
                        real_set.add(tuple(sorted(act)))  # 順序非依存のキー
                    else:
                        real_set.add(act)
            except Exception:
                real_set = None
            
            # ===== デバッグ: MCTS children と実環境の合法手を比較 =====
            if self.config.get('debug_action_mismatch', False) and real_set:
                try:
                    mcts_keys = set()
                    for act in root.children.keys():
                        if act == "pass":
                            mcts_keys.add("pass")
                        elif isinstance(act, (list, tuple)):
                            mcts_keys.add(tuple(sorted(str(c) for c in act)))
                        else:
                            mcts_keys.add(act)
                    
                    # 不一致を検出
                    only_in_mcts = mcts_keys - real_set
                    only_in_real = real_set - mcts_keys
                    
                    if only_in_mcts or only_in_real:
                        if hasattr(self, 'logger') and self.logger:
                            try:
                                _revo_state = bool(getattr(getattr(env.game, 'rule_checker', None), 'revolution', False))
                                _field = [str(c) for c in getattr(env.game, 'current_field', [])]
                                self.logger.debug(
                                    f"[MCTS-LEGAL-MISMATCH] pid={self.player_id} "
                                    f"field={_field} revo={_revo_state} "
                                    f"only_in_mcts={list(only_in_mcts)[:5]} "
                                    f"only_in_real={list(only_in_real)[:5]}"
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
            
            if real_set:
                # root.childrenのキーもtuple化して比較
                for act in list(root.children.keys()):
                    act_key = None
                    if act == "pass":
                        act_key = "pass"
                    elif isinstance(act, (list, tuple)):
                        act_key = tuple(sorted(str(c) for c in act))
                    else:
                        act_key = act
                    
                    if act_key not in real_set:
                        try:
                            del root.children[act]
                        except Exception:
                            pass
                
                if not root.children:
                    try:
                        n_leg = len(real_root_legal)
                        if n_leg > 0:
                            prior = 1.0 / float(n_leg)
                            to_play = getattr(root, 'to_play', getattr(env.game, 'turn', self.player_id))
                            for act in real_root_legal:
                                root.children[act] = PUCTNode(prior=prior, parent=root, action=act, to_play=to_play)
                    except Exception:
                        pass

        actions = list(root.children.keys())
        visits = [child.visit_count for child in root.children.values()]

        # 推論時のみ: Q値による足切り（Veto）戦略
        if not training:
            q_veto_threshold = self.config.get("inference_q_value_veto_threshold")
            if q_veto_threshold is not None and q_veto_threshold > 0.0:
                filtered_actions = []
                filtered_visits = []
                filtered_children = {}
                for act, child in root.children.items():
                    q_value = child.value  # Q値（平均勝率）= value_sum / visit_count
                    if q_value > q_veto_threshold:
                        filtered_actions.append(act)
                        filtered_visits.append(child.visit_count)
                        filtered_children[act] = child
                # 除外後の手が1つ以上ある場合のみ適用
                if filtered_actions:
                    actions = filtered_actions
                    visits = filtered_visits
                    root.children = filtered_children

        # 温度決定 (手数/進行に応じたスケジュール)
        tau_action = self._select_temperature(training=training)
        pi_action = self._apply_temperature_to_visits(visits, tau_action)
        tau_target = float(self.config.get("policy_target_tau", 1.0))
        pi_target = self._apply_temperature_to_visits(visits, tau_target)

        chosen = _choose_opening_action(self, actions, pi_action, training)
        action_env = None if chosen == "pass" else chosen
        if isinstance(action_env, tuple):
            action_env = list(action_env)
        action_env = self._validate_action(env, action_env)

        # ActionGuard: clamp to env legality
        action_env = _clamp_action_to_env(env, action_env, legal_actions_hint=kwargs.get('legal_actions'))

        state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))

        # リプレイサンプル保存 (value=None : 未確定)
        # サンプル保存の判定:
        # 既存実装は learning_player_id 固定で保存していたため、
        # Arena (最新モデルを複数プレイヤーに割当て) の場合に保存数が変動しない問題があった。
        # factory で割当時に `is_using_latest_model` を設定しているため、
        # こちらのフラグを優先して保存する（最新モデルを使用するプレイヤーを保存対象にする）。
        is_latest_model_user = bool(self.config.get('is_using_latest_model', False)) or (self.player_id == self.config.get('learning_player_id', 0))
        if training and is_latest_model_user:
            try:
                import random as _r
                val_ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                is_val_sample = (_r.random() < val_ratio) if val_ratio > 0.0 else False
            except Exception:
                is_val_sample = False

            try:
                _, value_scalar_for_store = self._policy_value(env)
                if isinstance(value_scalar_for_store, dict):
                    value_scalar_for_store = float(value_scalar_for_store.get(self.player_id, 0.5))
                elif isinstance(value_scalar_for_store, (list, tuple)):
                    pid = getattr(self, 'player_id', 0)
                    value_scalar_for_store = float(value_scalar_for_store[pid]) if 0 <= pid < len(value_scalar_for_store) else 0.5
                else:
                    value_scalar_for_store = float(value_scalar_for_store)
            except Exception:
                value_scalar_for_store = 0.5

            if is_val_sample:
                cfg = self.config
                _old_inf_dir = cfg.get('inference_dirichlet', False)
                _old_open_rand = cfg.get('opening_random_enable', False)
                _old_det_eval = cfg.get('determinization_mode_eval', None)
                try:
                    cfg['inference_dirichlet'] = False
                    cfg['opening_random_enable'] = False
                    cfg['determinization_mode_eval'] = 'none'
                except Exception:
                    pass
                try:
                    root_val = self._run_mcts(env, training=False)
                    actions_val = list(root_val.children.keys())
                    visits_val = [child.visit_count for child in root_val.children.values()]
                    tau_target = float(self.config.get("policy_target_tau", 1.0))
                    pi_target_val = self._apply_temperature_to_visits(visits_val, tau_target)
                    serialized_legal_val = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions_val]
                    # Attempt to map variable-length pi into canonical fixed-head ordering
                    try:
                        if hasattr(self, 'model') and getattr(self.model, 'canonical_action_keys', None) is not None:
                            canonical = self.model.canonical_action_keys()
                            key_to_idx = {k: i for i, k in enumerate(canonical)}
                            L = len(canonical)
                            pi_full = [0.0] * L
                            # helper: convert action -> canonical key string
                            def _act_to_key(a):
                                try:
                                    if a is None or a == 'pass':
                                        return 'PASS'
                                    cards = a if isinstance(a, (list, tuple)) else [a]
                                    from agents.agent_utills.feature_extractor import CardEncoder
                                    ce = CardEncoder()
                                    parts = []
                                    for c in cards:
                                        try:
                                            parts.append(int(ce.card_index(c)))
                                        except Exception:
                                            pass
                                    parts.sort()  # Integer sort to match action_vocab.py
                                    return '|'.join(str(x) for x in parts)
                                except Exception:
                                    try:
                                        return '|'.join(sorted(repr(x) for x in (a if isinstance(a, (list, tuple)) else [a])))
                                    except Exception:
                                        return str(a)
                            for i, act in enumerate(actions_val):
                                k = _act_to_key(act)
                                idx = key_to_idx.get(k, None)
                                try:
                                    v = float(pi_target_val[i]) if i < len(pi_target_val) else 0.0
                                except Exception:
                                    v = 0.0
                                if idx is not None and 0 <= idx < L:
                                    pi_full[idx] += v
                                else:
                                    # log unmapped actions for diagnosis (skip PASS)
                                    try:
                                        if act not in (None, 'pass'):
                                            self._log_fallback_once(
                                                "canonical_map_miss",
                                                f"[WARN] canonical map miss: action={act} key={k}",
                                            )
                                    except Exception:
                                        pass
                            # normalize if possible
                            s = sum(pi_full)
                            if s > 0:
                                pi_full = [p / s for p in pi_full]
                            else:
                                # fallback: uniform distribution over all canonical keys
                                L = len(pi_full)
                                pi_full = [1.0 / L if L > 0 else 0.0] * L
                            stored = self._store_sample(state_repr, serialized_legal_val, pi_full, None)
                        else:
                            # No canonical_action_keys available: create uniform distribution
                            L = 1059  # Known vocab size
                            pi_full_fallback = [1.0 / L] * L
                            stored = self._store_sample(state_repr, serialized_legal_val, pi_full_fallback, None)
                    except Exception as _map_e:
                        try:
                            if getattr(self, 'logger', None) and hasattr(self.logger, 'log_text'):
                                self.logger.log_text(f"[WARN] canonical mapping failed: {_map_e}")
                            else:
                                print(f"[WARN] canonical mapping failed: {_map_e}")
                        except Exception:
                            pass
                        # Fallback: uniform distribution over canonical keys
                        L = 1059
                        pi_full_fallback = [1.0 / L] * L
                        stored = self._store_sample(state_repr, serialized_legal_val, pi_full_fallback, None)
                    try:
                        stored['split'] = 'val'
                    except Exception:
                        pass
                except Exception:
                    serialized_legal = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions]
                    # Fallback: uniform distribution
                    L = 1059
                    pi_full_fallback = [1.0 / L] * L
                    stored = self._store_sample(state_repr, serialized_legal, pi_full_fallback, None)
                finally:
                    try:
                        cfg['inference_dirichlet'] = _old_inf_dir
                        cfg['opening_random_enable'] = _old_open_rand
                        if _old_det_eval is None:
                            try:
                                del cfg['determinization_mode_eval']
                            except Exception:
                                pass
                        else:
                            cfg['determinization_mode_eval'] = _old_det_eval
                    except Exception:
                        pass
                self._phase_samples.append(stored)
                self._phase_value_preds.append(value_scalar_for_store)
            else:
                serialized_legal = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions]
                try:
                    if hasattr(self, 'model') and getattr(self.model, 'canonical_action_keys', None) is not None:
                        canonical = self.model.canonical_action_keys()
                        key_to_idx = {k: i for i, k in enumerate(canonical)}
                        L = len(canonical)
                        pi_full = [0.0] * L
                        def _act_to_key(a):
                            try:
                                if a is None or a == 'pass':
                                    return 'PASS'
                                cards = a if isinstance(a, (list, tuple)) else [a]
                                from agents.agent_utills.feature_extractor import CardEncoder
                                ce = CardEncoder()
                                parts = []
                                for c in cards:
                                    try:
                                        parts.append(int(ce.card_index(c)))
                                    except Exception:
                                        pass
                                parts.sort()  # Integer sort to match action_vocab.py
                                return '|'.join(str(x) for x in parts)
                            except Exception:
                                try:
                                    return '|'.join(sorted(repr(x) for x in (a if isinstance(a, (list, tuple)) else [a])))
                                except Exception:
                                    return str(a)
                        for i, act in enumerate(actions):
                            k = _act_to_key(act)
                            idx = key_to_idx.get(k, None)
                            try:
                                v = float(pi_target[i]) if i < len(pi_target) else 0.0
                            except Exception:
                                v = 0.0
                                if idx is not None and 0 <= idx < L:
                                    pi_full[idx] += v
                                else:
                                    try:
                                        if act not in (None, 'pass'):
                                            self._log_fallback_once(
                                                "canonical_map_miss",
                                                f"[WARN] canonical map miss: action={act} key={k}",
                                            )
                                    except Exception:
                                        pass
                        s = sum(pi_full)
                        if s > 0:
                            pi_full = [p / s for p in pi_full]
                        else:
                            # fallback: uniform distribution over canonical keys
                            L = len(pi_full)
                            pi_full = [1.0 / L if L > 0 else 0.0] * L
                        stored = self._store_sample(state_repr, serialized_legal, pi_full, None)
                    else:
                        # No canonical_action_keys available: uniform distribution
                        L = 1059
                        pi_full_fallback = [1.0 / L] * L
                        stored = self._store_sample(state_repr, serialized_legal, pi_full_fallback, None)
                except Exception as _map_e:
                    try:
                        if getattr(self, 'logger', None) and hasattr(self.logger, 'log_text'):
                            self.logger.log_text(f"[WARN] canonical mapping failed: {_map_e}")
                        else:
                            print(f"[WARN] canonical mapping failed: {_map_e}")
                    except Exception:
                        pass
                    # Fallback: uniform distribution
                    L = 1059
                    pi_full_fallback = [1.0 / L] * L
                    stored = self._store_sample(state_repr, serialized_legal, pi_full_fallback, None)
                self._phase_samples.append(stored)
                self._phase_value_preds.append(value_scalar_for_store)

        self._last_state = state_repr
        self.move_count += 1

        # --- 行動履歴へ記録 (determinization 用) ---
        try:
            g = env.game
            combo_type = None
            try:
                rc = getattr(g, 'rule_checker', None)
                if rc and hasattr(rc, 'classify_combo') and g.current_field:
                    cinfo = rc.classify_combo(g.current_field)
                    if cinfo:
                        combo_type = cinfo.get('type')
            except Exception:
                combo_type = None
            hist_entry = {
                "pid": g.turn,
                "action": action_env if action_env is not None else "pass",
                "field_before": [str(c) for c in getattr(g, 'current_field', [])],
                "revo": bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False)),
                "combo_type": combo_type,
            }
            self._action_history.append(hist_entry)
            # Game 側が公式に _action_history を管理する。無い場合のみフォールバックで埋める。
            if not hasattr(g, '_action_history'):
                g._action_history = [hist_entry]  # type: ignore
        except Exception:
            pass

        return action_env

    # ---------------- Temperature schedule helpers ----------------
    def _compute_high_temp_window(self) -> int:
        """高温(τ=high)を適用する手数の上限を返す。

        仕様:
          - 初期は 10 手まで高温
          - 自己対戦エピソード数が増えるごとに段階的に短縮
          - 最低 2 手までは高温を維持

        短縮レートは簡易に「decay_every エピソードごとに 1 手短縮」。
        """
        base = int(self.config.get("temp_high_moves_initial", 10))
        min_win = int(self.config.get("temp_high_moves_min", 2))
        decay_every = int(self.config.get("temp_high_moves_decay_every", 500))  # 例: 500epごとに-1
        episodes = int(getattr(self, "episodes_played", 0))
        reduction = (episodes // decay_every) if decay_every > 0 else 0
        return max(min_win, base - reduction)

    def _select_temperature(self, training: bool = True) -> float:
        """現在手数と進行に応じて温度τを返す。

        - 学習時: 先頭 high_window 手は τ=1.0、それ以降は τ=0.1
        - 評価時: ほぼgreedy (τ≈0)
        """
        if not training:
            return 1e-6
        # 値は要件に合わせたデフォルト。必要なら config で上書き可能。
        high_temp = float(self.config.get("temp_high_value", 1.0))
        low_temp = float(self.config.get("temp_low_value", 0.1))
        high_window = self._compute_high_temp_window()
        return high_temp if self.move_count < high_window else low_temp

    def _apply_temperature_to_visits(self, visits, tau: float):
        """訪問数に温度τを適用して確率分布を生成する。

        Args:
            visits: ルート子ノードの訪問回数リスト
            tau: 温度パラメータ（0に近いほどシャープ、大きいほど平坦）

        Returns:
            温度適用後の確率分布（合計=1.0）
        """
        import numpy as np
        v = np.asarray(visits, dtype=np.float64)
        if tau <= 1e-6:
            # 低温極限: argmax にほぼ1.0
            pi = np.zeros_like(v, dtype=np.float64)
            pi[int(v.argmax())] = 1.0
            return pi.tolist()
        # v^(1/τ) で温度適用
        v_pow = np.power(v + 1e-10, 1.0 / tau)
        z = v_pow.sum()
        if z <= 0:
            # フォールバック: 一様分布
            return [1.0 / len(v)] * len(v)
        pi = v_pow / z
        return pi.tolist()

    # ---------------- Core (MCTS) ----------------
    def _run_mcts(self, env, training: bool = True) -> PUCTNode:
        """環境を軽量コピーし PUCT MCTS を実行してルートノードを返す."""
        return _run_mcts_external(self, env, training=training)

    def _get_tt_view(self):
        # MCTSRunner へ TT を供給する責務（TT本体の分離は phase-2）
        return AlphaZeroTTView(self)

    def _get_policy_value_batch_fn(self, *, training: bool):
        # MCTSRunner へ batch policy/value 関数を供給する責務（remote inference 分離は phase-2）
        def policy_value_batch_fn(env_list, legal_list=None):
            return self._policy_value_batch_impl(env_list, legal_list=legal_list, training=training)
        return policy_value_batch_fn

    def _policy_value_batch_impl(self, env_list, legal_list=None, training: bool = True):
        """バッチ推論: 複数環境をまとめて GPU forward し、結果を各環境へ分配する。
        
        Args:
            env_list: 環境のリスト
            legal_list: 各環境の legal_actions リスト (Optional)
            training: 学習モードかどうか
            
        Returns:
            List of (policy_dict, value_dict) tuples for each environment
        """
        if not env_list:
            return []
        
        # 単一環境の場合は従来通り（オーバーヘッド回避）
        if len(env_list) == 1:
            return [self._policy_value(env_list[0])]
        
        try:
            import torch
            
            # 1. 全環境の状態を抽出
            states = []
            legals = []
            for i, env in enumerate(env_list):
                state = self._extract_state(env)
                states.append(state)
                
                # legal_actions を取得（legal_list または env から）
                if legal_list is not None and i < len(legal_list):
                    legal = legal_list[i]
                else:
                    try:
                        legal = env.get_legal_actions() if hasattr(env, 'get_legal_actions') else None
                    except Exception:
                        legal = None
                legals.append(legal if legal else [])
            
            # 2. モデルでバッチ推論（GPU上で一括処理）
            with torch.no_grad():
                # device の type 属性を安全に取得（model.device へフォールバック）
                device = getattr(self, 'device', None) or getattr(self.model, 'device', None)
                use_amp = bool(getattr(device, 'type', None) == 'cuda')
                with torch.amp.autocast('cuda', enabled=use_amp):
                    policy_logits_batch, value_logits_batch = self.model.forward_batch(states)
            

            self._batch_inference_used = True
            results = []
            for i, (state, legal) in enumerate(zip(states, legals)):
                try:
                    # policy ロジットを legal 数に合わせて切り出し
                    logits_full = policy_logits_batch[i]
                    n_legal = len(legal)
                    
                    if n_legal > 0:
                        # legal の長さに合わせて切り出し（GPU上で）
                        if logits_full.shape[0] < n_legal:
                            pad = torch.zeros(n_legal - logits_full.shape[0], 
                                            device=logits_full.device, 
                                            dtype=logits_full.dtype)
                            logits_sel = torch.cat([logits_full, pad], dim=0)
                        else:
                            logits_sel = logits_full[:n_legal]
                        
                        # softmax は GPU 上で実行
                        policy_probs = torch.softmax(logits_sel, dim=0)
                        # CPU 転送は最小限（1回のみ）
                        policy_probs_list = policy_probs.cpu().tolist()
                        # リスト型のアクションを tuple に変換してキーとして使用可能にする
                        policy_dict = {(tuple(action) if isinstance(action, list) else action): prob 
                                      for action, prob in zip(legal, policy_probs_list)}
                    else:
                        policy_dict = {}
                    
                    # value: スカラー出力（学習対象プレイヤーの勝率）
                    # BCEWithLogitsLoss 使用のため、推論時は sigmoid で確率化
                    value_logits = value_logits_batch[i]
                    value_prob = torch.sigmoid(value_logits)
                    # スカラー値を取得（shape [] or [1] → float）
                    value_scalar = float(value_prob.item())
                    
                    results.append((policy_dict, value_scalar))
                    
                except Exception as e:
                    # 個別環境の処理エラー時はフォールバック
                    self._log_fallback_once(
                        f"batch_impl_env_{i}",
                        f"[batch-fallback] env {i} failed in batch processing; fallback to single",
                        e
                    )
                    results.append(self._policy_value(env_list[i]))
            
            return results
            
        except Exception as e:
            # バッチ推論全体が失敗した場合は従来の逐次処理にフォールバック
            self._log_fallback_once(
                "batch_impl_all",
                "[batch-fallback] Batch inference failed; fallback to sequential processing",
                e
            )
            return [self._policy_value(e) for e in env_list]

    def _ensure_inference_client(self) -> InferenceClient:
        """InferenceClient の遅延初期化。リモートキューの有無に応じて適切なクライアントを返す。"""
        if self._inference_client is not None:
            return self._inference_client
        
        # リモート推論キューが設定されているか確認
        rq = getattr(self, '_remote_request_q', None)
        rsp_q = getattr(self, '_remote_response_q', None)
        wid = getattr(self, '_remote_worker_id', None)
        
        if rq is not None and rsp_q is not None and wid is not None:
            self._inference_client = RemoteInferenceClient(self)
        else:
            self._inference_client = LocalInferenceClient(self)
        
        return self._inference_client

    def _policy_value(self, env):
        """1 環境に対する policy 分布と value を返す（InferenceClient 経由）。"""
        import time as _tperf
        
        # 1. 状態抽出
        state = self._extract_state(env)
        
        # 2. ターミナル状態のショートカット（この実装の value は「区間(phase/stage)の勝者=1、それ以外=0」）
        # NOTE: 多人数のためスカラー 1.0 を返すのは誤り（全プレイヤーが勝ちになる）。
        # ここでは「手番プレイヤーの手札が0」という異常/境界状態に対しても
        # per-player dict を返すことで MCTS/ログを壊さないようにする。
        try:
            hand_sz = None
            if isinstance(state, dict):
                hand_sz = state.get('hand_size', None)
                if hand_sz is None:
                    h = state.get('hand', None)
                    if isinstance(h, (list, tuple)):
                        hand_sz = len(h)
            if hand_sz is None:
                try:
                    g = getattr(env, 'game', None)
                    if g is not None:
                        pid = getattr(g, 'turn', None)
                        if pid is not None:
                            pl = getattr(g, 'players', None)
                            if pl is not None and 0 <= int(pid) < len(pl):
                                hand_attr = getattr(pl[int(pid)], 'hand', None)
                                if isinstance(hand_attr, (list, tuple)):
                                    hand_sz = len(hand_attr)
                except Exception:
                    hand_sz = None

            if hand_sz == 0:
                g = getattr(env, 'game', None)
                turn_pid = int(getattr(g, 'turn', getattr(self, 'player_id', 0))) if g is not None else int(getattr(self, 'player_id', 0))
                try:
                    n_players = int(getattr(g, 'num_players', int(self.config.get('num_players', 4)))) if g is not None else int(self.config.get('num_players', 4))
                except Exception:
                    n_players = 4
                # stage winner として turn_pid を採用（手札0ならそのプレイヤーの上がりとして扱う）
                already_won = set(getattr(env, 'already_won_players', set()) or set())
                values = {}
                for pid in range(n_players):
                    if pid == turn_pid:
                        values[pid] = 1.0
                    elif pid not in already_won:
                        values[pid] = 0.0
                return {}, values
        except Exception:
            pass
        
        # 3. 合法手取得
        legal = self._get_legal_actions(env)
        if not legal:
            self._log_fallback_once(
                "policy_value_no_legal",
                "[az-fallback] no legal actions; returning empty policy/value"
            )
            return {}, 0.0
        
        # 4. InferenceClient 経由で推論実行（パフォーマンス計測付き）
        _t0 = _tperf.time()
        try:
            client = self._ensure_inference_client()
            policy_dict, value_scalar = client.infer_policy_value(state, legal)
        except Exception as e:
            # フォールバック: 均等分布
            # リスト型のアクションを tuple に変換してキーとして使用可能にする
            self._log_fallback_once(
                "policy_value_infer_exception",
                f"[az-fallback] infer_policy_value failed; using uniform policy (training={getattr(self, '_current_training_mode', False)})",
                e
            )
            n = len(legal)
            policy_dict = {(tuple(a) if isinstance(a, list) else a): 1.0 / n for a in legal}
            value_scalar = 0.5
        
        # 5. パフォーマンス統計の更新
        try:
            _dt_ms = (_tperf.time() - _t0) * 1000.0
            self._perf_infer_ms_accum += _dt_ms
            self._perf_infer_calls += 1
            if bool(self.config.get('measure_forward_time', False)):
                alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                if self._forward_time_ms_ema is None:
                    self._forward_time_ms_ema = float(_dt_ms)
                else:
                    self._forward_time_ms_ema = float(alpha * _dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
        except (AttributeError, ValueError, TypeError) as e:
            if self.logger:
                try:
                    self.logger.debug(f"Perf stats update failed: {e}")
                except Exception:
                    pass
        
        self._record_eval_value(value_scalar)
        return policy_dict, value_scalar

    def _value_scalar_for_self(self, value_scalar: Any) -> Optional[float]:
        pid = int(getattr(self, 'player_id', 0) or 0)
        try:
            if isinstance(value_scalar, dict):
                if pid in value_scalar:
                    return float(value_scalar[pid])
                return None
            if isinstance(value_scalar, (list, tuple)):
                if 0 <= pid < len(value_scalar):
                    return float(value_scalar[pid])
                return None
            return float(value_scalar)
        except Exception as e:
            self._log_fallback_once(
                "validate_action_exception",
                "[az-fallback] validate_action failed; returning pass",
                e
            )
            return None

    def _record_eval_value(self, value_scalar: Any) -> None:
        if not getattr(self, '_capture_eval_value', False):
            return
        if getattr(self, '_last_eval_value', None) is not None:
            return
        val = self._value_scalar_for_self(value_scalar)
        if val is None:
            return
        self._last_eval_value = val
        self._capture_eval_value = False

    def get_last_eval_value(self) -> Optional[float]:
        """直近 select_action 呼び出しでキャプチャした value（推論結果）を返す。"""
        return getattr(self, '_last_eval_value', None)

    # ---------------- Env helpers ----------------
    def _copy_env(self, env):
        """環境を shallow copy し、Game の最小限状態だけ複製した高速シミュレーション用コピーを生成.
        
        重要: 革命状態・場の状態・passed状態など、合法手判定に関わるすべての状態を確実にコピーする。
        """
        base = env
        new_env = copy.copy(base)  # シェルコピー
        g = base.game
        g_new = copy.copy(g)       # Game オブジェクト浅いコピー (__init__ 不呼び出し)
        
        # ===== 1. 現在の手番・ターン情報を記録（検証用） =====
        _original_turn = getattr(g, 'turn', 0)
        _original_revolution = False
        try:
            _original_revolution = bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False))
        except Exception:
            pass
        
        # Player hand はリスト + Card をコピー（参照共有しない）
        # MCTS/合法手生成が Joker 代用状態などを変更すると、本 env に漏れて合法性がズレるため。
        try:
            from game.card import Card

            def _clone_card(c):
                try:
                    if getattr(c, 'is_joker', False):
                        nc = Card(is_joker=True)
                        try:
                            nc.joker_as_rank = getattr(c, 'joker_as_rank', None)
                            nc.joker_as_suit = getattr(c, 'joker_as_suit', None)
                        except Exception:
                            pass
                        return nc
                    return Card(suit=getattr(c, 'suit', None), rank=getattr(c, 'rank', None), is_joker=False)
                except Exception:
                    # 最終フォールバック
                    return Card(is_joker=True)
        except Exception:
            Card = None  # type: ignore

            def _clone_card(c):
                return c
        new_players = []
        for p in g.players:
            p_new = copy.copy(p)
            try:
                p_new.hand = [_clone_card(c) for c in list(p.hand)]
            except Exception:
                p_new.hand = list(p.hand)
            new_players.append(p_new)
        g_new.players = new_players
        
        # ===== 2. 場の状態を完全にコピー =====
        try:
            g_new.current_field = [_clone_card(c) for c in list(getattr(g, 'current_field', []) or [])]
        except Exception:
            g_new.current_field = list(getattr(g, 'current_field', []) or [])
        
        # ===== 3. passed状態・rankings・手番を確実にコピー =====
        g_new.passed = list(g.passed)
        g_new.rankings = list(getattr(g, 'rankings', []))
        try:
            g_new.turn = int(getattr(g, 'turn', 0))
        except Exception:
            pass
        # ===== 4. RuleChecker / 革命フラグを確実にコピー =====
        try:
            from game.rules import RuleChecker
            rc_src = getattr(g, 'rule_checker', None)
            rc = RuleChecker()
            if rc_src is not None:
                try:
                    rc.revolution = bool(getattr(rc_src, 'revolution', False))
                except Exception:
                    rc.revolution = False
                # すべてのルールチェッカー属性を確実にコピー
                for name in (
                    'strict_straight_progression',
                    'rev_enable_four_kind', 'rev_four_kind_allow_joker', 'rev_four_kind_exact',
                    'rev_enable_straight', 'rev_straight_min_len', 'rev_straight_allow_joker',
                    'enable_joker', 'joker_is_strongest', 'enable_eight_reverse',
                ):
                    try:
                        setattr(rc, name, getattr(rc_src, name))
                    except Exception:
                        pass
            g_new.rule_checker = rc
        except Exception:
            pass
        try:
            g_new.silent = True  # シミュレーション時の print 抑制
        except Exception:
            pass
        
        # ===== 5. 環境レベルの状態をコピー =====
        try:
            # 環境が持つ追加の状態（あれば）をコピー
            if hasattr(base, 'last_action'):
                new_env.last_action = getattr(base, 'last_action', None)
            if hasattr(base, 'done'):
                new_env.done = bool(getattr(base, 'done', False))
        except Exception:
            pass
        
        new_env.game = g_new
        
        # ===== 6. コピー検証（デバッグモード時のみ） =====
        if self.config.get('debug_copy_verification', False):
            try:
                _verify_turn = getattr(g_new, 'turn', 0)
                _verify_revo = bool(getattr(getattr(g_new, 'rule_checker', None), 'revolution', False))
                if _verify_turn != _original_turn:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"[COPY-WARNING] turn mismatch: orig={_original_turn} copy={_verify_turn}")
                if _verify_revo != _original_revolution:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"[COPY-WARNING] revolution mismatch: orig={_original_revolution} copy={_verify_revo}")
            except Exception:
                pass
        
        return new_env

    # ------------ Determinization helpers (parallel pool) ------------
    def _maybe_start_det_pool(self, env):
        return _maybe_start_det_pool_impl(self, env)

    def _determinization_worker(self):
        return _determinization_worker_impl(self)

    def _apply_from_det_pool(self, e_clone, original_env, root_pid) -> bool:
        return _apply_from_det_pool_impl(self, e_clone, original_env, root_pid)

    # --- Inline fallback determinization (直接適用) ---
    def _inline_determinize(self, e_clone, original_env, root_pid):
        return _inline_determinize_impl(self, e_clone, original_env, root_pid)

    # コア生成: apply_direct=False なら (ok, assignment_dict) を返す
    def _build_single_determinization(self, e_clone, original_env, root_pid, apply_direct=False):
        return _build_single_determinization_impl(self, e_clone, original_env, root_pid, apply_direct=apply_direct)

    def _filter_high_rank_waste(self, env, legal_actions: List[Any]) -> List[Any]:
        """無駄な高ランク出し禁止フィルタ。
        
        最も弱いカード + max_rank_gap までのカードしか出せないように制限。
        例: max_rank_gap=3, 最弱が4(rank=2) なら rank 2+3=5 まで許可。
        
        Args:
            env: 環境オブジェクト
            legal_actions: フィルタリング前の合法手リスト
        
        Returns:
            フィルタリング後の合法手リスト（最低1手は残す）
        """
        if not legal_actions:
            return legal_actions
        
        try:
            from game.card import Card
            
            # 場のカードと革命状態を取得
            field = (env.game.current_field or [])[:]  
            if not field:
                # 場が空ならフィルタ無効（最初の手出し）
                return legal_actions
            
            rc = getattr(env.game, 'rule_checker', None)
            if rc is None:
                return legal_actions
            
            # 場の役を分類
            field_combo = rc.classify_combo(field)
            if not field_combo:
                return legal_actions
            
            field_type = field_combo.get('type')
            field_strength = field_combo.get('strength', 0)
            field_size = field_combo.get('size', 0)
            
            # パス以外の合法手を収集
            non_pass_actions = [a for a in legal_actions if a != "pass"]
            if len(non_pass_actions) <= 1:
                # 手が1つ以下ならフィルタ不要
                return legal_actions
            
            # 各合法手を解析
            action_info = []
            for action in non_pass_actions:
                try:
                    cards = [Card.from_string(s) for s in action] if isinstance(action, list) else []
                    if not cards:
                        continue
                    
                    combo = rc.classify_combo(cards)
                    if not combo:
                        continue
                    
                    action_type = combo.get('type')
                    action_strength = combo.get('strength', 0)
                    action_size = combo.get('size', 0)
                    
                    # 場の役と同じ種類でサイズも一致する手のみ比較対象
                    if action_type != field_type or action_size != field_size:
                        continue
                    
                    # 場より強い手のみが合法手のはず
                    if action_strength <= field_strength:
                        continue
                    
                    action_info.append({
                        'action': action,
                        'strength': action_strength,
                        'cards': cards,
                    })
                except Exception:
                    continue
            
            if len(action_info) <= 1:
                # 比較可能な手が1つ以下ならフィルタ不要
                return legal_actions
            
            # 強さでソート（昇順: 弱い手から強い手へ）
            action_info.sort(key=lambda x: x['strength'])
            
            # 最弱手の強さを取得
            min_strength = action_info[0]['strength']
            
            # 許可される最大強さを計算
            max_rank_gap = int(self.config.get('filter_high_rank_max_gap', 3))
            max_allowed_strength = min_strength + max_rank_gap
            
            # フィルタリング後の合法手を構築
            filtered = []
            for action in legal_actions:
                if action == "pass":
                    # パスは常に残す
                    filtered.append(action)
                    continue
                
                # このアクションの強さを検索
                action_strength = None
                for info in action_info:
                    if info['action'] == action:
                        action_strength = info['strength']
                        break
                
                # action_infoにない場合（別の役種類など）はそのまま残す
                if action_strength is None:
                    filtered.append(action)
                    continue
                
                # 強さが許可範囲内なら残す
                if action_strength <= max_allowed_strength:
                    filtered.append(action)
            
            # 安全装置: 全ての手を除外してしまった場合は元のリストを返す
            if not filtered or (len(filtered) == 1 and filtered[0] == "pass"):
                return legal_actions
            
            return filtered
        
        except Exception as e:
            # エラー時は元のリストをそのまま返す（フィルタリング失敗）
            self._log_fallback_once(
                "filter_high_rank_exception",
                "[az-fallback] filter_high_rank_waste failed; returning unfiltered legal actions",
                e
            )
            return legal_actions

    def _get_legal_actions(self, env) -> List[Any]:
        """現在手番プレイヤーの合法手集合を可変長リストで返す。

        戻り値: List[List[str]] - 各要素はカード文字列のリスト ['♣6'] など
                   パスの場合は "pass" 文字列
        
        変更点:
          - 合法手は正規化・重複排除後、カノニカル順に安定ソートする
          - 戻り値はリストのリスト形式（tupleではなく）
        """
        try:
            cur = env.game.players[env.game.turn]
            hand = cur.hand
            field = (env.game.current_field or [])[:]
            if hasattr(env, "_generate_legal_actions"):
                raw = env._generate_legal_actions(hand, field)
            else:
                raw = []
        except Exception as e:
            self._log_fallback_once(
                "get_legal_actions_exception",
                "[az-fallback] get_legal_actions failed; returning empty",
                e
            )
            raw = []
        
        acts: List[Any] = []
        seen = set()
        field_is_empty = False
        try:
            field_is_empty = (len(env.game.current_field or []) == 0)
        except Exception:
            field_is_empty = False
        
        for a in raw:
            if a is None:
                continue
            try:
                # 重複排除のためのキー（順序非依存）
                tup = tuple(sorted(str(c) for c in a))
            except Exception:
                continue
            
            if tup not in seen:
                seen.add(tup)
                # 実際のアクションとしてはリスト形式で保存
                try:
                    acts.append([str(c) for c in a])
                except Exception:
                    acts.append(list(a))
        
        # Canonical sorting（pass は最後に付与）
        def _canon_key(action):
            # pass は最も後ろに配置
            if action == "pass":
                return (99, 0, 0, 1, ("ZZZ",))
            # 役分類ベースの安定キー
            try:
                from game.card import Card
                rc = getattr(env.game, 'rule_checker', None)
                cards = [Card.from_string(s) for s in action] if isinstance(action, list) else []
                combo = rc.classify_combo(cards) if (rc and cards) else None
                type_order = {
                    'single': 0,
                    'pair': 1,
                    'triple': 2,
                    'four': 3,
                    'straight': 4,
                    'joker_single': 5,
                }
                ctype = combo.get('type') if combo else None
                size = combo.get('size') if combo else len(cards)
                strength = combo.get('strength') if combo else 0
                jokers = combo.get('jokers') if combo else 0
                # タイブレーク: 文字列IDの辞書順（順序非依存化のため内部はソート）
                id_key = tuple(sorted(action)) if isinstance(action, list) else (str(action),)
                return (type_order.get(ctype, 98), int(size or 0), int(strength or 0), int(jokers or 0), id_key)
            except Exception:
                id_key = tuple(sorted(action)) if isinstance(action, list) else (str(action),)
                return (97, 0, 0, 0, id_key)

        acts_sorted = sorted(acts, key=_canon_key)
        # パスは「場が空でない場合」にのみ合法手として許可する（最後に付加）
        if not field_is_empty and ("pass" not in acts_sorted):
            acts_sorted.append("pass")
        
        # 無駄な高ランク出し禁止フィルタ（推論時のみ適用）
        filter_enable = self.config.get('filter_high_rank_enable', False)
        filter_training = self.config.get('filter_high_rank_training', False)
        is_training = getattr(self, '_current_training_mode', False)
        
        if filter_enable and (not is_training or filter_training):
            acts_sorted = self._filter_high_rank_waste(env, acts_sorted)
        
        return acts_sorted

    def _validate_action(self, env, action):
        """選択行動が実際の手札で再現可能か最低限の検証を行い、不正なら None(=パス扱い)。"""
        try:
            if action is None:
                return None
            cur = env.game.players[env.game.turn]
            hand_set = {str(c) for c in cur.hand}
            if isinstance(action, list) and all(card in hand_set for card in action):
                return action
            return None
        except Exception:
            return None

    def _extract_state(self, env, prev_state=None):
        return _extract_state_external(self, env, prev_state=prev_state)

    def _extract_state_impl(self, env, prev_state=None):
        """後方互換のため残す（旧コードパス互換）。

        実体は agents/agent_utills/feature_extractor.py の
        FeatureExtractor 実装へ委譲する。
        """
        return _extract_state_external(self, env, prev_state=prev_state)

    

    # ---------------- Replay buffer ----------------
    def _store_sample(self, state, legal_actions, pi, value):
        """リプレイサンプル1件を保存。

        本体処理は [agents/replay_buffer.py](agents/replay_buffer.py) の
        `store_replay_sample()` に移管。
        """
        return _store_replay_sample(self, state, legal_actions, pi, value)

    def assign_values(self, samples: List[Any], value: float):
        """フェーズ確定時に保留サンプルへ 0/1 ラベルを一括適用。統計も更新."""
        for s in samples:
            if not isinstance(s, dict):  # 後方互換: index の可能性
                if 0 <= s < len(self.replay_buffer):
                    rec = self.replay_buffer[s]
                else:
                    continue
            else:
                rec = s
            try:
                st = rec.get('state') if isinstance(rec, dict) else None
                has_hl = isinstance(st, dict) and ('hand_labels' in st)
                hl_dim = st.get('hand_labels_dim') if has_hl else None
            except Exception:
                pass
            prev_labeled = _has_value_label(rec)
            if not prev_labeled:
                self.total_value_samples += 1
                if value > 0.5:
                    self.total_positive += 1
            # Store raw float value (no quantization)
            rec["value"] = float(value)
            
            # 学習/検証スプリットを一度だけ付与
            
            if rec.get('split') is None:
                ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                import random as _r
                rec['split'] = 'val' if (_r.random() < ratio) else 'train'
            

        # ---------------- 動的 pos_weight 調整 ----------------
        # 目的: クラス不均衡 (陽性率 pos_rate) の変動に応じて BCE の正例重みを (1-pos_rate)/pos_rate へ近づける。
        # 段階的に更新して過剰振動を防ぐ。
        try:
            dyn_enable = self.config.get('dynamic_pos_weight_enable', True)
            if dyn_enable and self.total_value_samples >= int(self.config.get('dynamic_pos_weight_warmup_samples', 1500)):
                pos_rate = self.total_positive / max(1, self.total_value_samples)
                # 逆出現率 (inverse prevalence) をターゲットとする
                target = (1.0 - pos_rate) / max(pos_rate, 1e-6)
                # クリップ範囲
                min_w = float(self.config.get('dynamic_pos_weight_min', 1.0))
                max_w = float(self.config.get('dynamic_pos_weight_max', 3.0))
                target = max(min_w, min(max_w, target))
                old = self.pos_weight
                # 更新間隔（サンプル数ベース）
                interval = int(self.config.get('dynamic_pos_weight_update_every_samples', 1000))
                if interval <= 0 or (self.total_value_samples % interval) == 0:
                    # 最大ステップ幅 (一度に跳ね上げない)
                    step_max = float(self.config.get('dynamic_pos_weight_step_max', 0.2))
                    delta = target - old
                    if abs(delta) >= 1e-3:  # 有意差のみ更新
                        adj = old + max(-step_max, min(step_max, delta))
                        self.pos_weight = adj
                        # config にも書き戻し (train_step 側が参照するケースに備え)
                        try:
                            self.config['value_pos_weight'] = self.pos_weight
                        except Exception:
                            pass
                        # ログ（スロットリング対応）
                        try:
                            log_enable = bool(self.config.get('dynamic_pos_weight_log_enable', True))
                            log_every = int(self.config.get('dynamic_pos_weight_log_every_samples', 10000) or 0)
                            delta_min = float(self.config.get('dynamic_pos_weight_log_delta_min', 0.3) or 0.0)
                            last_s = getattr(self, '_posw_last_log_samples', 0)
                            last_v = getattr(self, '_posw_last_log_value', float('nan'))
                            samples_ok = (log_every <= 0) or ((self.total_value_samples - int(last_s)) >= log_every)
                            delta_ok = (not (last_v != last_v)) or (abs(self.pos_weight - float(last_v)) >= delta_min)  # last_v!=last_v is NaN check
                            if log_enable and (samples_ok or delta_ok):
                                if self.logger:
                                    self.logger.log_text(
                                        f"[pos-weight-adjust] samples={self.total_value_samples} pos_rate={pos_rate:.3f} old={old:.3f} target={target:.3f} new={self.pos_weight:.3f}")
                                else:
                                    import os, datetime
                                    log_dir = self.config.get('log_dir', 'logs')
                                    os.makedirs(log_dir, exist_ok=True)
                                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                                        f.write(f"[{ts}] [pos-weight-adjust] samples={self.total_value_samples} pos_rate={pos_rate:.3f} old={old:.3f} target={target:.3f} new={self.pos_weight:.3f}\n")
                                # 更新
                                self._posw_last_log_samples = int(self.total_value_samples)
                                self._posw_last_log_value = float(self.pos_weight)
                        except Exception:
                            pass
        except Exception:
            pass

    def assign_values_by_winner(self, samples: List[Any], winner_player_id: int):
        """Assign 0/1 value labels per-sample based on absolute winner_player_id.

        This ensures that samples which have been canonicalized (self-centric)
        but still carry their originating `player_id` receive value labels
        that reflect whether their owner won the game.
        """
        for s in samples:
            if not isinstance(s, dict):
                if 0 <= s < len(self.replay_buffer):
                    rec = self.replay_buffer[s]
                else:
                    continue
            else:
                rec = s
            try:
                # Determine owner player id for this sample. Prefer explicit field,
                # fall back to sample's stored player_id or agent's player_id.
                owner = None
                try:
                    owner = rec.get('player_id') if isinstance(rec, dict) else None
                    if owner is None:
                        st = rec.get('state') if isinstance(rec, dict) else None
                        if isinstance(st, dict) and ('player_id' in st):
                            owner = st.get('player_id')
                except Exception:
                    owner = None
                if owner is None:
                    owner = int(getattr(self, 'player_id', 0))
                val = 1.0 if int(owner) == int(winner_player_id) else 0.0
            except Exception:
                val = 0.0
            if not _has_value_label(rec):
                self.total_value_samples += 1
                if val > 0.5:
                    self.total_positive += 1
            try:
                # Store raw float value (no quantization)
                rec['value'] = float(val)
            except Exception as e:
                print(f"[ERROR-value] {e}")
                pass
            try:
                if rec.get('split') is None:
                    ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                    import random as _r
                    rec['split'] = 'val' if (_r.random() < ratio) else 'train'
            except Exception:
                pass

    def _reset_phase_buffers(self):
        self._phase_samples = []
        self._phase_value_preds = []

    def finalize_phase(self, winner_player_id: int, was_active: bool):
        """フェーズ終端処理: 勝者IDに基づき 0/1 ラベル付与 + 予測精度集計."""
        if not was_active or not self._phase_samples:
            self._reset_phase_buffers()
            return
        val = 1.0 if self.player_id == winner_player_id else 0.0
        lost = sum(1 for s in self._phase_samples if isinstance(s, dict) and s.get("in_buffer") is False)
        if lost:
            self.lost_phase_samples += lost
        try:
            last_rec = self._phase_samples[-1]
            pred = self._phase_value_preds[-1] if self._phase_value_preds else None
            if pred is not None:
                self.phase_total += 1
                self.episode_phase_total += 1
                hit = (pred > 0.5) == (val > 0.5)
                if hit:
                    self.phase_correct += 1
                    self.episode_phase_correct += 1
        except Exception:
            pass
        # Assign values per-sample relative to each sample's owner (robust to canonicalization)
        try:
            self.assign_values_by_winner(self._phase_samples, winner_player_id)
        except Exception:
            # fallback to legacy behavior
            self.assign_values(self._phase_samples, val)
        # worker_zero_buffer モード: 確定サンプルを一時リストに保存
        if self.replay_buffer is None:
            for s in self._phase_samples:
                has_label = _has_value_label(s)
                if isinstance(s, dict) and has_label:
                    # Deep copy to prevent later modifications
                    copied = copy.deepcopy(s)
                    self._episode_confirmed_samples.append(copied)
        self._reset_phase_buffers()

    def flush_unfinished_phase(self):
        """未確定フェーズを 0 扱いで確定 (エピソード終了/中断時)。"""
        if self._phase_samples:
            lost = sum(1 for s in self._phase_samples if isinstance(s, dict) and s.get("in_buffer") is False)
            if lost:
                self.lost_phase_samples += lost
            try:
                last_rec = self._phase_samples[-1]
                pred = self._phase_value_preds[-1] if self._phase_value_preds else None
                if pred is not None:
                    self.phase_total += 1
                    self.episode_phase_total += 1
                    if pred <= 0.5:  # 0 と予測していたら的中
                        self.phase_correct += 1
                        self.episode_phase_correct += 1
            except Exception:
                pass
            self.assign_values(self._phase_samples, 0.0)
            # worker_zero_buffer モード: 確定サンプルを一時リストに保存
            if self.replay_buffer is None:
                for s in self._phase_samples:
                    if isinstance(s, dict) and _has_value_label(s):
                        # Deep copy to prevent later modifications
                        self._episode_confirmed_samples.append(copy.deepcopy(s))
            self._reset_phase_buffers()

    def finalize_game(self, *_args, **_kwargs):  # 互換維持用 no-op
        """ゲーム終端フック (最終順位報酬を使わないので何もしない)。"""
        # フェーズ一時サンプル破棄
        self._reset_phase_buffers()
        # ゲーム単位データをクリア
        try:
            if hasattr(self, '_action_history'):
                self._action_history.clear()
        except Exception:
            pass
        # determinization プールをゲーム境界でリセット (設定で無効化可能)
        try:
            if self.config.get('reset_det_pool_each_game', True):
                self.shutdown_det_pool()
        except Exception:
            pass

    # ---------------- Persistence ----------------
    # --- Optimizer helpers ---
    def ensure_optimizer(self):
        """Ensure that the optimizer exists (lazy init with current hyperparams).
        
        Value Headには強めのWeight Decayと低めの学習率を適用して過学習を抑制する。
        """
        if self._optimizer is None:
            try:
                import torch
                if self.model is None or not hasattr(self.model, 'parameters'):
                    self._optimizer = None
                    return
                # Debug: dump named parameter names for runtime inspection
                try:
                    if bool(self.config.get('debug_optimizer_named_params', False)) or bool(self.config.get('debug_value_flow', False)):
                        try:
                            names = [n for n, _ in self.model.named_parameters()]
                            print(f"[DEBUG_OPT_NAMED_PARAMS] names={names}")
                        except Exception:
                            try:
                                print(f"[DEBUG_OPT_NAMED_PARAMS] failed to list named_parameters")
                            except Exception:
                                pass
                except Exception:
                    pass
                
                lr = float(self.config.get("lr", getattr(self, 'lr', 1e-4)))
                wd = float(self.config.get("weight_decay", getattr(self, 'weight_decay', 1e-4)))
                
                # Value Head専用の設定
                value_head_wd = float(self.config.get("value_head_weight_decay", wd * 5))
                value_head_lr_scale = float(self.config.get("value_head_lr_scale", 0.5))
                
                # パラメータをグループ分け
                value_head_params = []
                other_params = []
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if 'value_head' in name:
                        value_head_params.append(param)
                    else:
                        other_params.append(param)

                # Debug: value_head delta snapshot (ensure_optimizerはvalue_head可視な地点)
                try:
                    if bool(self.config.get('debug_value_flow', False)) and bool(self.config.get('debug_value_delta_in_optimizer', True)):
                        vh_numel = 0
                        vh_sample0 = None
                        vh_norm = None
                        if value_head_params:
                            with torch.no_grad():
                                flat = torch.cat([p.detach().reshape(-1).to('cpu') for p in value_head_params])
                                vh_numel = int(flat.numel())
                                vh_sample0 = float(flat[0].item()) if vh_numel > 0 else None
                                vh_norm = float(torch.linalg.vector_norm(flat).item())

                        prev = getattr(self, '_debug_vh_snapshot', None)
                        if prev is None:
                            print(
                                f"[DEBUG_VALUE_DELTA_OPT] init numel={vh_numel} norm={vh_norm} sample0={vh_sample0}"
                            )
                        else:
                            try:
                                dn = None
                                ds0 = None
                                if vh_norm is not None and prev.get('norm') is not None:
                                    dn = float(abs(vh_norm - prev.get('norm')))
                                if vh_sample0 is not None and prev.get('sample0') is not None:
                                    ds0 = float(vh_sample0 - prev.get('sample0'))
                                print(
                                    f"[DEBUG_VALUE_DELTA_OPT] numel={vh_numel} norm={vh_norm} sample0={vh_sample0} "
                                    f"abs_dnorm={dn} dsample0={ds0}"
                                )
                            except Exception:
                                print(
                                    f"[DEBUG_VALUE_DELTA_OPT] numel={vh_numel} norm={vh_norm} sample0={vh_sample0}"
                                )

                        setattr(self, '_debug_vh_snapshot', {'numel': vh_numel, 'norm': vh_norm, 'sample0': vh_sample0})
                except Exception:
                    pass

                # Debug: show whether value_head is trainable / included
                try:
                    if bool(self.config.get('debug_value_flow', False)):
                        vh_all = 0
                        vh_trainable = 0
                        for name, param in self.model.named_parameters():
                            if 'value_head' in name:
                                vh_all += 1
                                if bool(getattr(param, 'requires_grad', True)):
                                    vh_trainable += 1
                        print(f"[DEBUG_OPT] value_head_named_params={vh_all} trainable={vh_trainable} grouped={len(value_head_params)}")
                except Exception:
                    pass
                
                # パラメータグループを作成
                param_groups = []
                if other_params:
                    param_groups.append({
                        'params': other_params,
                        'lr': lr,
                        'weight_decay': wd,
                    })
                if value_head_params:
                    param_groups.append({
                        'params': value_head_params,
                        'lr': lr * value_head_lr_scale,
                        'weight_decay': value_head_wd,
                        'name': 'value_head',  # デバッグ用
                    })
                
                if param_groups:
                    self._optimizer = torch.optim.Adam(param_groups)
                    try:
                        if bool(self.config.get('debug_value_flow', False)):
                            # summarize param group sizes
                            gsz = []
                            for g in self._optimizer.param_groups:
                                try:
                                    nm = g.get('name', 'main')
                                except Exception:
                                    nm = 'main'
                                try:
                                    n = sum(int(p.numel()) for p in g.get('params', []) if p is not None)
                                except Exception:
                                    n = None
                                try:
                                    glr = g.get('lr', None)
                                except Exception:
                                    glr = None
                                try:
                                    gwd = g.get('weight_decay', None)
                                except Exception:
                                    gwd = None
                                gsz.append(f"{nm}:n={n}:lr={glr}:wd={gwd}")
                            print(f"[DEBUG_OPT] param_groups=" + ", ".join(gsz))
                    except Exception:
                        pass
                else:
                    self._optimizer = None
            except Exception:
                self._optimizer = None

    def save_optimizer(self, path: Optional[str] = None) -> bool:
        """Save optimizer state_dict to file. Returns True on success."""
        try:
            import os
            import torch
            if path is None:
                path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "optimizer_latest.pt")
            # Respect config: optionally disable checkpoint saves
            if self.config.get('disable_checkpoint_saving', False):
                return False
            self.ensure_optimizer()
            if self._optimizer is None:
                return False
            torch.save(self._optimizer.state_dict(), path)
            return True
        except Exception:
            return False

    # --- Scheduler helpers (Warmup + Cosine) ---
    def ensure_scheduler(self):
        """Ensure LR scheduler (warmup+cosine without restarts) is created if enabled.

        設定キー:
          - lr_scheduler: 'none' | 'warmup_cosine' (既定: warmup_cosine)
          - lr_warmup_steps: int (既定: 0)
          - lr_cosine_T_max_updates: int (既定: 0 = 無効)
          - lr_min: float (既定: 0.0)
        """
        try:
            import math
            import torch
            if self._scheduler is not None:
                return
            # scheduler 有効条件
            sched_type = str(self.config.get('lr_scheduler', 'warmup_cosine')).lower()
            if sched_type not in ('warmup_cosine',):
                return
            if self._optimizer is None:
                self.ensure_optimizer()
            if self._optimizer is None:
                return
            base_lr = float(self.config.get('lr', getattr(self, 'lr', 7e-5)))
            warmup = int(self.config.get('lr_warmup_steps', 0) or 0)
            tmax = int(self.config.get('lr_cosine_T_max_updates', 0) or 0)
            lr_min = float(self.config.get('lr_min', 0.0) or 0.0)
            min_scale = (lr_min / base_lr) if base_lr > 0 else 0.0

            def _lr_lambda(step: int):
                # step は 0 始まりの更新カウンタ
                if warmup > 0 and step < warmup:
                    return max(1e-8, float(step + 1) / float(warmup))
                if tmax <= warmup or tmax <= 0:
                    return min_scale
                prog = min(1.0, float(step - warmup) / float(max(1, tmax - warmup)))
                cos_factor = 0.5 * (1.0 + math.cos(math.pi * prog))
                return min_scale + (1.0 - min_scale) * cos_factor

            self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=_lr_lambda, last_epoch=max(-1, self._update_step - 1)
            )
        except Exception:
            self._scheduler = None

    def save_scheduler(self, path: Optional[str] = None) -> bool:
        """Save scheduler state_dict to file. Returns True on success."""
        try:
            import os
            import torch
            if self.config.get('disable_checkpoint_saving', False):
                return False
            if self._scheduler is None:
                return False
            if path is None:
                path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "scheduler_latest.pt")
            torch.save(self._scheduler.state_dict(), path)
            return True
        except Exception:
            return False

    def load_scheduler(self, path: Optional[str] = None) -> bool:
        """Load scheduler state_dict from file. Returns True on success.

        事前に optimizer / scheduler を ensure 済みにしておく必要があります。
        """
        try:
            import os
            import torch
            if path is None:
                path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "scheduler_latest.pt")
            if not os.path.isfile(path):
                return False
            # ensure scheduler exists
            self.ensure_optimizer()
            self.ensure_scheduler()
            if self._scheduler is None:
                return False
            # PyTorch FutureWarning 回避と安全性向上: weights_only=True を使用
            try:
                state = torch.load(path, map_location='cpu', weights_only=True)
            except TypeError:
                # 古い PyTorch 互換
                state = torch.load(path, map_location='cpu')
            self._scheduler.load_state_dict(state)
            # LambdaLR の state_dict には last_epoch が含まれるため、内部カウンタを同期させる
            try:
                last_ep = getattr(self._scheduler, 'last_epoch', None)
                if isinstance(last_ep, int):
                    # 次回 step() 後に last_epoch == 保存値 + 1 となるように自前の更新カウンタも揃える
                    self._update_step = max(int(self._update_step or 0), last_ep + 1)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def reset_scheduler_warmup(self) -> bool:
        """Warmup を最初からやり直すために LR とスケジューラをワンショットでリセットする。

        - optimizer の param_group['lr'] を基準学習率へ戻す
        - 自前カウンタ _update_step を 0 に戻す
        - スケジューラを再構築 (last_epoch = -1) し、次の step() から warmup を再開
        """
        try:
            self.ensure_optimizer()
            if self._optimizer is None:
                return False
            # 学習率を基準値へ戻す
            base_lr = float(self.config.get('lr', getattr(self, 'lr', 7e-5)))
            try:
                for g in self._optimizer.param_groups:
                    g['lr'] = base_lr
            except Exception:
                pass
            # カウンタとスケジューラを再構築
            self._update_step = 0
            self._scheduler = None
            self.ensure_scheduler()
            if self.logger:
                try:
                    self.logger.log_text("[resume] scheduler warmup reset once")
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def load_optimizer(self, path: Optional[str] = None, map_location: Optional[str] = None, override_lr: Optional[float] = None) -> bool:
        """Load optimizer state_dict from file.

        - If override_lr is provided, set loaded param group lrs to this value.
        - Returns True on success, False otherwise.
        """
        try:
            import os
            import torch
            if path is None:
                path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "optimizer_latest.pt")
            if not os.path.isfile(path):
                return False
            self.ensure_optimizer()
            if self._optimizer is None:
                return False
            # PyTorch FutureWarning 回避と安全性向上: weights_only=True を使用
            try:
                state = torch.load(
                    path,
                    map_location=map_location or (getattr(getattr(self.model, 'device', None), 'type', None) or 'cpu'),
                    weights_only=True,
                )
            except TypeError:
                # 古い PyTorch 互換
                state = torch.load(
                    path,
                    map_location=map_location or (getattr(getattr(self.model, 'device', None), 'type', None) or 'cpu'),
                )
            self._optimizer.load_state_dict(state)
            if override_lr is not None:
                try:
                    for g in self._optimizer.param_groups:
                        g['lr'] = float(override_lr)
                except (KeyError, ValueError, TypeError) as e:
                    if self.logger:
                        try:
                            self.logger.debug(f"Override LR failed: {e}")
                        except Exception:
                            pass
            return True
        except Exception:
            return False

    def save_replay(self, path: Optional[str] = None):
        """リプレイバッファを保存し、必要ならメモリ解放(purge)。

        動作:
          - 共有リプレイ (ReplayBuffer インスタンス) の場合: ReplayBuffer.save を使用し、
            config.purge_replay_after_save=True なら purge=True で呼ぶ。
          - ローカル deque/list の場合: joblib.dump で直接保存し、purge 指定時は要素をクリア。

        設定キー:
          purge_replay_after_save: bool (デフォルト False)
          trainer_only_replay_save: bool (True の場合、config.is_trainer_process が True でなければ何もしない)
        """
        path = path or self.config.get("replay_path", "replay_buffer.joblib")
        # Respect config option to disable data writes (replay / data files)
        if self.config.get('disable_data_writes', False):
            return
        # --- Trainer ガード ---
        if self.config.get('trainer_only_replay_save') and not self.config.get('is_trainer_process'):
            return  # 非トレーナープロセス/スレッドでは保存しない
        # purge のデフォルトは安全側(False)
        purge = bool(self.config.get("purge_replay_after_save", False))
        rb = getattr(self, 'replay_buffer', None)
        # 共有 ReplayBuffer
        if rb is not None and rb.__class__.__name__ == 'ReplayBuffer':
            try:
                # 新しい save API (purge 対応)BCEWithLogitsLoss 化
                rb.save(path, purge=purge)
            except TypeError:
                # 互換: 古いバージョン (purgeパラメータ無し)
                try:
                    rb.save(path)
                except Exception:
                    joblib.dump(rb, path, compress=3)
                if purge:
                    try:
                        rb.clear()
                    except Exception:
                        pass
            except Exception:
                # 最終フォールバック
                try:
                    joblib.dump(rb, path, compress=3)
                except Exception:
                    joblib.dump(rb, path)
                if purge:
                    try:
                        rb.clear()
                    except Exception:
                        pass
            return
        # ローカル deque/list 格納形式
        # 非同期 I/O 経由
        async_used = False
        try:
            if self.config.get('enable_async_io', False):
                from utils.async_io import get_async_io
                aio = get_async_io(self.config)
                if aio:
                    aio.enqueue_joblib_dump(rb, path, compress=3)
                    async_used = True
        except Exception:
            async_used = False
        if not async_used:
            try:
                joblib.dump(rb, path, compress=3)
            except Exception:
                joblib.dump(rb, path)
        if purge and hasattr(rb, 'clear'):
            rb.clear()
            

    def load_replay(self, path: Optional[str] = None):
        """joblib からリプレイバッファを読み込み (無ければ空)。"""
        path = path or self.config.get("replay_path", "replay_buffer.joblib")
        try:
            data = joblib.load(path)
            # feature_version フィルタ (フル特徴量モード時)
            if self.config.get('use_full_features'):
                if isinstance(data, list):
                    legacy = sum(1 for s in data if isinstance(s, dict) and s.get('feature_version',0)==0)
                    if legacy > 0:
                        # バックアップ
                        
                        bk = path + '.legacy_backup'
                        joblib.dump(data, bk, compress=3)
                        
                        data = [s for s in data if isinstance(s, dict) and s.get('feature_version',0)==1]
                        print(f"[INFO] purged legacy replay samples={legacy} kept={len(data)}")
            self.replay_buffer = data
        except FileNotFoundError:
            self.replay_buffer = []

    # ---------------- Training ----------------
    def train_step(self, batch_size: int = 64):
        """学習処理は [agents/agent_utills/training.py](agents/agent_utills/training.py) に委譲。

        互換性のため、このメソッド名は維持する。
        """
        return TrainStepMixin.train_step(self, batch_size=batch_size)

    # ------------------ Measurement helpers ------------------
    def measure_select_action_time(self, env_or_obs, iterations: int = 10, warmup: int = 2, training: bool = True, **kwargs):
        """エージェントの select_action を複数回呼んで所要時間統計を返すユーティリティ。

        戻り値: dict(count, total_s, mean_s, std_s, min_s, max_s)
        注意: MCTS を含むため実行は重い。必要に応じて iterations を調整してください。
        """
        import time as _time
        try:
            import statistics as _stats
        except Exception:
            _stats = None

        # warmup
        for _ in range(max(0, int(warmup or 0))):
            try:
                self.select_action(env_or_obs, training=training, **kwargs)
            except Exception:
                # warmup 中の例外は無視して続行
                pass

        times = []
        for _ in range(max(0, int(iterations or 0))):
            t0 = _time.time()
            try:
                self.select_action(env_or_obs, training=training, **kwargs)
            except Exception:
                # 計測中の例外は時間計測は継続しつつ記録しない
                continue
            times.append(_time.time() - t0)

        if not times:
            return {"count": 0, "total_s": 0.0, "mean_s": None, "std_s": None, "min_s": None, "max_s": None}

        total = sum(times)
        mean = total / len(times)
        mn = min(times)
        mx = max(times)
        std = None
        if _stats is not None and len(times) > 1:
            try:
                std = float(_stats.pstdev(times))
            except Exception:
                try:
                    std = float(_stats.stdev(times))
                except Exception:
                    std = None

        out = {
            "count": len(times),
            "total_s": float(total),
            "mean_s": float(mean),
            "std_s": float(std) if std is not None else None,
            "min_s": float(mn),
            "max_s": float(mx),
        }
        # Log to events.log (use agent.logger if available)
        try:
            line = (f"[MEASURE][select_action] pid={self.player_id} count={out['count']} "
                    f"total_s={out['total_s']:.6f} mean_s={out['mean_s']:.6f} std_s={out['std_s'] if out['std_s'] is not None else 'n/a'} "
                    f"min_s={out['min_s']:.6f} max_s={out['max_s']:.6f}")
            if self.logger and hasattr(self.logger, 'log_text'):
                self.logger.log_text(line, also_print=False)
                
            else:
                import os, datetime
                log_dir = self.config.get('log_dir', 'logs')
                os.makedirs(log_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}] {line}\n")
                
        except Exception:
            pass
        return out

    def measure_train_step_time(self, batch_size: int = 64, iterations: int = 10, warmup: int = 2):
        """train_step を複数回実行して所要時間統計を返すユーティリティ。

        事前にデータが無い場合は reason を含む辞書を返します。
        戻り値: dict(count, total_s, mean_s, std_s, min_s, max_s, last_result)
        """
        import time as _time
        try:
            import statistics as _stats
        except Exception:
            _stats = None

        # 簡易チェック: データやモデルが無い場合は早期リターン
        
        check = self.train_step(batch_size=batch_size)
        if isinstance(check, dict) and check.get('reason') is not None:
            return {"count": 0, "total_s": 0.0, "mean_s": None, "std_s": None, "min_s": None, "max_s": None, "reason": check.get('reason')}
        

        # warmup
        for _ in range(max(0, int(warmup or 0))):
            self.train_step(batch_size=batch_size)
            

        times = []
        last_result = None
        for _ in range(max(0, int(iterations or 0))):
            t0 = _time.time()
            try:
                last_result = self.train_step(batch_size=batch_size)
            except Exception:
                # 実行エラーが起きても計測は継続
                last_result = {"loss": None, "reason": "exception"}
            times.append(_time.time() - t0)

        if not times:
            return {"count": 0, "total_s": 0.0, "mean_s": None, "std_s": None, "min_s": None, "max_s": None, "last_result": last_result}

        total = sum(times)
        mean = total / len(times)
        mn = min(times)
        mx = max(times)
        std = None
        if _stats is not None and len(times) > 1:
            try:
                std = float(_stats.pstdev(times))
            except Exception:
                try:
                    std = float(_stats.stdev(times))
                except Exception:
                    std = None

        out = {
            "count": len(times),
            "total_s": float(total),
            "mean_s": float(mean),
            "std_s": float(std) if std is not None else None,
            "min_s": float(mn),
            "max_s": float(mx),
            "last_result": last_result,
        }
        # 計測ログ出力はユーザ要望により無効化（結果のみ返す）
        return out

    def validate_step(self, batch_size: Optional[int] = None):
        """検証用: 検証splitのサンプルで損失を計算して返す（勾配・更新なし）。

        実装は agents.validation.validate_on_agent に委譲しており、
        API は従来通り維持します。
        """
        try:
            from agents.validation import validate_on_agent
            return validate_on_agent(self, batch_size=batch_size)
        except Exception as e:
            try:
                import traceback as _tb
                tb = _tb.format_exc()
            except Exception:
                tb = None
            out = {"policy_loss": None, "value_loss": None, "entropy": None, "reason": f"validate_error: {type(e).__name__}: {e}"}
            if tb:
                out['traceback'] = tb
            return out

    def reset_episode(self):
        """エピソード開始時にカウンタ類を初期化."""
        self.move_count = 0
        # 進行に応じた温度スケジュール短縮のため、エピソード数をカウント
        try:
            self.episodes_played += 1
        except (AttributeError, TypeError):
            self.episodes_played = int(getattr(self, "episodes_played", 0)) + 1
        self.episode_phase_total = 0
        self.episode_phase_correct = 0
        # Belief 増分キャッシュ/直近状態をリセット
        try:
            self._belief_cache = None
            self._last_state = None
        except (AttributeError, KeyError) as e:
            if self.logger:
                try:
                    self.logger.debug(f"Belief cache reset failed: {e}")
                except Exception:
                    pass
        # worker_zero_buffer モード用の確定サンプルリストをクリア
        # (エピソード開始前にクリアすることで、前エピソードの送信済みサンプルを保持しない)
        if hasattr(self, '_episode_confirmed_samples'):
            self._episode_confirmed_samples.clear()
        # 行動履歴のクリア（メモリ削減オプション）
        if self.config.get('clear_action_history_per_episode', True):
            try:
                if hasattr(self, '_action_history'):
                    self._action_history.clear()
            except (AttributeError, RuntimeError) as e:
                if self.logger:
                    try:
                        self.logger.debug(f"Action history clear failed: {e}")
                    except Exception:
                        pass
        # Phase サンプルの積極的クリア（メモリ削減オプション）
        if self.config.get('aggressive_phase_clear', False):
            try:
                self._reset_phase_buffers()
            except (AttributeError, RuntimeError) as e:
                if self.logger:
                    try:
                        self.logger.debug(f"Phase buffer clear failed: {e}")
                    except Exception:
                        pass
        # determinization プールはゲームを跨ぐと署名ミスマッチが増えるためデフォルトで再初期化
        try:
            if self.config.get('reset_det_pool_each_game', True):
                self.shutdown_det_pool()
        except (AttributeError, RuntimeError) as e:
            if self.logger:
                try:
                    self.logger.debug(f"Det pool reset failed: {e}")
                except Exception:
                    pass
        # per-episode perf counters
        try:
            self._perf_infer_ms_accum = 0.0
            self._perf_infer_calls = 0
        except (AttributeError, TypeError) as e:
            if self.logger:
                try:
                    self.logger.debug(f"Perf counter reset failed: {e}")
                except Exception:
                    pass

    # ------------ Determinization pool teardown ------------
    def shutdown_det_pool(self):
        """テスト/終了時に determinization ワーカーを安全に停止する補助メソッド."""
        return _shutdown_det_pool_impl(self)


DRLAgent = AlphaZeroAgent
