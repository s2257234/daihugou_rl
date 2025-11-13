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
from typing import Any, Dict, List, Optional
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


def softmax_temperature_policy(visits: List[int], temperature: float) -> List[float]:
    """訪問回数リストに温度付きソフトマックスを適用して確率分布を返す。

    temperature が極小(≈0) の場合は argmax を one-hot で返し決定的選択に近づける。
    """
    if not visits:
        return []
    if temperature <= 1e-6:  # 低温度: 決定的 (最大訪問のみ 1)
        m = max(visits)
        return [1.0 if v == m else 0.0 for v in visits]
    # 温度スケーリング: v^(1/T)
    scaled = [v ** (1.0 / max(temperature, 1e-6)) for v in visits]
    s = sum(scaled)
    return [x / s for x in scaled] if s > 0 else [1.0 / len(scaled)] * len(scaled)


class AlphaZeroAgent:
    """大富豪用 AlphaZero 風エージェント。

    主機能:
      - MCTS(PUCT) により行動方策分布(pi) を推定
      - フェーズ勝利確率 value を同時学習 (BCE)
      - リプレイバッファへ (状態, 方策, valueラベル) を蓄積
    """

    def __init__(self, player_id: int, model=None, config=None):
        # --- 基本設定 / ハイパーパラメータ ---
        self.player_id = player_id
        self.config = config or ALPHA_ZERO_CONFIG
        self.model = model

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
        # worker_zero_buffer モード用: エピソード内で確定したサンプルを一時保持
        self._episode_confirmed_samples: List[Dict[str, Any]] = []
        self.env_ref = None  # 直近参照環境
        self.logger = None   # 外部ロガー (TensorBoard 等)
        self._logged_inside = False  # 二重記録防止
        # インクリメンタル Belief 用キャッシュ
        # 形式: {
        #   'num_players': int,
        #   'poss_sets': {pid: set(card_idx)},
        #   'hist_len': int,  # 適用済み action_history 長
        # }
        self._belief_cache = None
        # 直近状態（差分用ヒント）
        self._last_state = None
        self.total_value_samples = 0
        self.total_positive = 0
        self.phase_total = 0
        self.phase_correct = 0
        self.episode_phase_total = 0
        self.episode_phase_correct = 0
        self.lost_phase_samples = 0
        self.tt_hits = 0
        self.tt_misses = 0
        self.pos_weight = float(self.config.get("value_pos_weight", 1.5))
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
        except Exception:
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
        if hasattr(env_or_obs, "game"):
            env = env_or_obs
            self.env_ref = env
        else:
            env = self.env_ref
        if env is None:
            return None
    # ---- MCTS 実行 & 性能計測 ----
        import time as _perf_t
        _t0 = _perf_t.time()
        # training フラグを MCTS 実行へ伝播
        root = self._run_mcts(env, training=training)
        _mcts_ms = (_perf_t.time() - _t0) * 1000.0
        # 実際のシミュレーション数 (Early Stop 計測)
        # mcts-sims ログはユーザ要望により無効化（以前は平均/直近シミュレーション数を一定間隔で記録）
        # ここでは何も行わず静粛化。
        # 実効シミュレーション数と early-stop を集計
        try:
            sims = int(getattr(root, '_actual_simulations', 0) or 0)
            es = bool(getattr(root, '_early_stopped', False))
            self._mcts_moves += 1
            if es:
                self._mcts_early_stop += 1
            try:
                self._mcts_sims_recent.append(sims)
            except Exception:
                # list フォールバック
                self._mcts_sims_recent.append(sims) if hasattr(self._mcts_sims_recent, 'append') else None
        except Exception:
            pass
        # 1手あたりのMCTSパフォーマンスログ（任意で有効化）
        try:
            if bool(self.config.get('mcts_perf_log_enable', False)):
                every = int(self.config.get('mcts_perf_log_every', 30) or 1)
                if every <= 1 or (self.move_count % every) == 0:
                    # sims を堅牢に算出（_actual_simulations が0/未設定なら子 visit_count 合計）
                    try:
                        sims_logged = int(getattr(root, '_actual_simulations', 0) or 0)
                    except Exception:
                        sims_logged = 0
                    if sims_logged <= 0:
                        try:
                            sims_logged = int(sum(int(getattr(ch, 'visit_count', 0) or 0) for ch in root.children.values()))
                        except Exception:
                            sims_logged = 0
                    # clone_ms: ルート属性→self保持→None
                    _cm = getattr(root, '_perf_clone_ms', None)
                    if _cm is None:
                        _cm = getattr(self, '_last_perf_clone_ms', None)
                    clone_ms = float(_cm) if _cm is not None else None
                    # det モード: ルート属性→self保持→config から推定
                    det_mode = getattr(root, '_perf_det_mode', None)
                    if not det_mode:
                        det_mode = getattr(self, '_last_perf_det_mode', None)
                    if not det_mode:
                        det_mode = (self.config.get('determinization_mode_train', 'fixed_once') if training
                                    else self.config.get('determinization_mode_eval', 'stochastic'))
                    infer_ms = float(getattr(self, '_perf_infer_ms_accum', 0.0) or 0.0)
                    try:
                        legal_n = len(list(root.children.keys()))
                    except Exception:
                        legal_n = 0
                    early = 1 if bool(getattr(root, '_early_stopped', False)) else 0
                    clone_str = f"{clone_ms:.1f}" if clone_ms is not None else "n/a"
                    line = (
                        f"[MCTS-PERF] pid={self.player_id} move={self.move_count} sims={sims_logged} "
                        f"legal={legal_n} det={det_mode} early={early} "
                        f"clone_ms={clone_str} infer_ms={infer_ms:.1f} total_ms={_mcts_ms:.1f}"
                    )
                    if self.logger:
                        self.logger.log_text(line)
                    else:
                        # フォールバック: ロガーが無い場合でも events.log へ追記
                        try:
                            import os, datetime
                            log_dir = self.config.get('log_dir', 'logs')
                            os.makedirs(log_dir, exist_ok=True)
                            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                                f.write(f"[{ts}] {line}\n")
                        except Exception:
                            # 最終手段として標準出力
                            print(line)
        except Exception:
            pass
        actions = list(root.children.keys())
        visits = [child.visit_count for child in root.children.values()]

        # 温度決定 (手数/進行に応じたスケジュール)
        # 行動サンプリング用: 現在のスケジュールに従う（高温→低温）
        tau_action = self._select_temperature(training=training)
        pi_action = self._apply_temperature_to_visits(visits, tau_action)
        tau_target = float(self.config.get("policy_target_tau", 1.0))
        pi_target = self._apply_temperature_to_visits(visits, tau_target)

        # MCTS 統計のサンプリングログ (低確率で記録)
        if actions:
            try:
                import math
                priors = [root.children[a].prior for a in actions]
                s_p = sum(priors)
                priors_n = [p / s_p for p in priors] if s_p > 0 else [1/len(priors)]*len(priors)
                s_v = sum(visits)
                visit_probs = [v / s_v for v in visits] if s_v > 0 else [1/len(visits)]*len(visits)
                def _entropy(vec):
                    return -sum(p*math.log(max(p,1e-12)) for p in vec)
                prior_ent = _entropy(priors_n)
                visit_ent = _entropy(visit_probs)
                kl = sum(p*(math.log(max(p,1e-12)) - math.log(max(q,1e-12))) for p,q in zip(priors_n, visit_probs))
                top1_same = 1 if priors_n.index(max(priors_n)) == visit_probs.index(max(visit_probs)) else 0
                rate = self.config.get("mcts_log_sample_rate", 0.15)
                if self.logger and random.random() < rate:
                    self.logger.log_mcts_sample({
                        "player": self.player_id,
                        "move_index": self.move_count,
                        "legal_count": len(actions),
                        "prior_entropy": prior_ent,
                        "visit_entropy": visit_ent,
                        "kl_prior_visit": kl,
                        "top1_same": top1_same,
                        "temperature": tau_action,  # 行動用温度を記録
                    })
            except Exception:
                pass
        # per-move [perf] ログは削除済 (必要なら Git 履歴から復元可能)

        # 序盤完全ランダムオプション (config.opening_random_enable)
        opening_random_enable = bool(self.config.get("opening_random_enable", False))
        opening_random_moves = int(self.config.get("opening_random_moves", 0))
        opening_include_pass = bool(self.config.get("opening_random_include_pass", False))
        use_opening_random = (
            training
            and opening_random_enable
            and opening_random_moves > 0
            and self.move_count < opening_random_moves
        )
        random_pool = actions
        if use_opening_random and actions:
            if not opening_include_pass:
                filtered = [a for a in actions if a != "pass"]
                if filtered:
                    random_pool = filtered
            chosen = random.choice(random_pool) if random_pool else "pass"
        else:
            # π_action に従い行動サンプリング (行動なしなら pass)
            chosen = random.choices(actions, weights=pi_action, k=1)[0] if actions else "pass"
        action_env = None if chosen == "pass" else chosen
        if isinstance(action_env, tuple):  # tuple を list 化
            action_env = list(action_env)
        action_env = self._validate_action(env, action_env)
        state_repr = self._extract_state(env, prev_state=getattr(self, '_last_state', None))

        # リプレイサンプル保存 (value=None : 未確定)
        # 評価/推論(training=False) 時はサンプルを蓄積しない（評価ワーカーのメモリ膨張を防止）
        if training:
            # 重要: 保存するのは pi_target（学習用、エントロピーの低い分布）
            # 検証サンプルは Dirichlet/Opening Random 無効で収集する
            try:
                import random as _r
                val_ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                is_val_sample = (_r.random() < val_ratio) if val_ratio > 0.0 else False
            except Exception:
                is_val_sample = False

            try:
                # ルート価値を再取得 (MCTS 内で破棄されるため再計算)
                _, value_scalar_for_store = self._policy_value(env)
                # ベクター/辞書の場合は自分視点に射影
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
                # 一時的に推論時ノイズを強制オフにして MCTS を再実行（検証用 π を収集）
                cfg = self.config
                _old_inf_dir = cfg.get('inference_dirichlet', False)
                _old_open_rand = cfg.get('opening_random_enable', False)
                # 追加: 検証サンプル再生成時は determinization を学習時と同じ none に固定し分布揺らぎを抑える
                _old_det_eval = cfg.get('determinization_mode_eval', None)
                try:
                    cfg['inference_dirichlet'] = False
                    cfg['opening_random_enable'] = False
                    # eval再MCTS用の det モードを 'none' に一時上書き（stochastic の揺らぎ除去）
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
                    stored = self._store_sample(state_repr, serialized_legal_val, pi_target_val, None, value_pred=value_scalar_for_store)
                    # split を検証に固定
                    try:
                        stored['split'] = 'val'
                    except Exception:
                        pass
                except Exception:
                    # 失敗時は通常サンプルとして格納
                    serialized_legal = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions]
                    stored = self._store_sample(state_repr, serialized_legal, pi_target, None, value_pred=value_scalar_for_store)
                finally:
                    # 設定を元に戻す
                    try:
                        cfg['inference_dirichlet'] = _old_inf_dir
                        cfg['opening_random_enable'] = _old_open_rand
                        if _old_det_eval is None:
                            # 元々キーが無かった場合は削除して後方互換を保つ
                            try: del cfg['determinization_mode_eval']
                            except Exception: pass
                        else:
                            cfg['determinization_mode_eval'] = _old_det_eval
                    except Exception:
                        pass
                self._phase_samples.append(stored)
            else:
                serialized_legal = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions]
                stored = self._store_sample(state_repr, serialized_legal, pi_target, None, value_pred=value_scalar_for_store)
                # split は assign_values 側で設定（後方互換）
                self._phase_samples.append(stored)
        # 次回差分用に直近状態を保持
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
                "pid": g.turn,  # 行動直前の手番 (action 適用前に抽出済 root.turn)
                "action": action_env if action_env is not None else "pass",
                "field_before": [str(c) for c in getattr(g, 'current_field', [])],
                "revo": bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False)),
                "combo_type": combo_type,
            }
            self._action_history.append(hist_entry)
            # 環境へも埋め込み (クローンが参照できるよう)
            if not hasattr(g, '_action_history'):
                g._action_history = []  # type: ignore
            g._action_history.append(hist_entry)  # type: ignore
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
        import time as _tperf
        _t_clone0 = _tperf.time()
        env_copy = self._copy_env(env)
        _clone_ms = (_tperf.time() - _t_clone0) * 1000.0
        # フォールバック用に保持
        try:
            self._last_perf_clone_ms = float(_clone_ms)
        except Exception:
            pass
        # NN推論累積タイマーをリセット
        try:
            self._perf_infer_ms_accum = 0.0
        except Exception:
            pass

        # 並列 determinization プール lazy 起動（determinization 有効かつ 'none' 以外の時のみ）
        try:
            if self.config.get('enable_parallel_determinization') and bool(self.config.get('enable_determinization', True)):
                _det_mode_now = (
                    self.config.get('determinization_mode_train', 'fixed_once') if training
                    else self.config.get('determinization_mode_eval', 'stochastic')
                )
                if _det_mode_now != 'none':
                    self._maybe_start_det_pool(env)
        except Exception:
            pass

        # ノード展開時に prior と value を取得
        def policy_value_fn(e):
            return self._policy_value(e)

        def legal_fn(e):  # 合法手生成
            return self._get_legal_actions(e)

        # トランスポジションテーブル準備（必要時のみ）
        TT = None
        if self.enable_mcts_tt:
            if self._mcts_tt is None:
                # 使用回数で簡易LRU: {key: (value, last_tick)}
                self._mcts_tt = {}
                self._mcts_tt_tick = 0
            TT = _AlphaZeroTTView(self)

        # バッチ版 policy_value 関数（固定アクションヘッド時のみ活用、未対応なら None）
        # legal_list: 事前計算済みの合法手（run_puct_mcts から渡される）。再計算を避けるために使用。
        def policy_value_batch_fn(env_list, legal_list=None):
            try:
                mdl = self.model
                if mdl is None:
                    outs = []
                    for i, e in enumerate(env_list):
                        legal = (legal_list[i] if legal_list is not None else self._get_legal_actions(e))
                        if not legal:
                            outs.append(({}, 0.0))
                        else:
                            p = 1.0 / len(legal)
                            outs.append(({a: p for a in legal}, 0.0))
                    return outs
                # If remote inference is available, delegate per-request and remap to legal actions
                rq = getattr(self, '_remote_request_q', None)
                rsp_q = getattr(self, '_remote_response_q', None)
                wid = getattr(self, '_remote_worker_id', None)
                if rq is not None and rsp_q is not None and wid is not None:
                    import uuid, time as _tmo
                    n_env = len(env_list)
                    outs = [None] * n_env
                    req_ids = []
                    start_lat = _tmo.time()
                    # send requests
                    for i, e in enumerate(env_list):
                        try:
                            payload = self._extract_state(e)
                        except Exception:
                            payload = None
                        rid = uuid.uuid4().hex
                        req_ids.append(rid)
                        try:
                            rq.put((int(wid), rid, payload))
                        except Exception:
                            pass
                    # collect with timeout
                    timeout = float(self.config.get('remote_infer_timeout', 5.0) or 5.0)
                    start_ts = _tmo.time()
                    received = 0
                    while received < n_env and (_tmo.time() - start_ts) < timeout:
                        try:
                            rrid, out = rsp_q.get(timeout=0.5)
                        except Exception:
                            continue
                        try:
                            if rrid in req_ids:
                                idx = req_ids.index(rrid)
                                outs[idx] = out
                                received += 1
                        except Exception:
                            pass
                    # 計測: バッチ遅延を per-env 平均として加算
                    try:
                        dt_ms = float((_tmo.time() - start_lat) * 1000.0)
                        self._perf_infer_ms_accum += dt_ms
                        self._perf_infer_calls += max(1, n_env)
                        if bool(self.config.get('measure_forward_time', False)):
                            alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                            per_env_ms = dt_ms / max(1, n_env)
                            if self._forward_time_ms_ema is None:
                                self._forward_time_ms_ema = float(per_env_ms)
                            else:
                                self._forward_time_ms_ema = float(alpha * per_env_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                    except Exception:
                        pass
                    # map to legal actions, fallback locally if missing
                    results = []
                    for i, e in enumerate(env_list):
                        legal = (legal_list[i] if legal_list is not None else self._get_legal_actions(e))
                        if outs[i] is None:
                            # local fallback
                            results.append(self._policy_value(e))
                            continue
                        out = outs[i]
                        try:
                            # out could be (pol,val) or (pol,val,extras)
                            pol_raw, val_raw = out[0], out[1]
                            extras = out[2] if (isinstance(out, tuple) and len(out) >= 3) else None
                            # cache hand probs if present (last one wins)
                            if isinstance(extras, dict) and isinstance(extras.get('hand_probs'), (list, tuple)):
                                self._remote_hand_probs = list(extras.get('hand_probs'))
                            # remap index->prob to legal actions if keys are indices
                            mapped = {}
                            if legal and pol_raw:
                                if all(isinstance(k, int) for k in getattr(pol_raw, 'keys', lambda: [])()):
                                    # index mapping
                                    for j, a in enumerate(legal):
                                        try:
                                            mapped[a] = float(pol_raw.get(j, 0.0))
                                        except Exception:
                                            mapped[a] = 0.0
                                else:
                                    mapped = pol_raw
                            else:
                                mapped = {}
                            results.append((mapped, val_raw))
                        except Exception:
                            # final fallback local
                            results.append(self._policy_value(e))
                    return results
                # 可変長アクションモデルはバッチ困難 → 個別に処理
                if getattr(mdl, 'supports_variable_actions', False) and hasattr(mdl, 'evaluate'):
                    return [self._policy_value(e) for e in env_list]
                # forward_batch があれば使う
                if hasattr(mdl, 'forward_batch'):
                    states = [self._extract_state(e) for e in env_list]
                    # 推論は勾配不要 + CUDA では AMP を利用
                    try:
                        import torch as _t
                        _dev = getattr(mdl, 'device', None)
                        _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
                        with _t.no_grad():
                            with _t.amp.autocast('cuda', enabled=_use_amp):
                                import time as _tperf
                                _t0 = _tperf.time()
                                logits_b, value_vec_b = mdl.forward_batch(states)
                        try:
                            self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
                        except Exception:
                            pass
                    except Exception:
                        import time as _tperf
                        _t0 = _tperf.time()
                        logits_b, value_vec_b = mdl.forward_batch(states)
                        try:
                            self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )  # best-effort
                        except Exception:
                            pass
                    outs = []
                    for i, e in enumerate(env_list):
                        # 事前計算した合法手があれば再利用
                        legal = (legal_list[i] if legal_list is not None else self._get_legal_actions(e))
                        if not legal:
                            outs.append(({}, 0.0))
                            continue
                        n = len(legal)
                        logits_t = logits_b[i]
                        if hasattr(logits_t, 'shape'):
                            import torch as _t
                            if logits_t.shape[0] < n:
                                pad = _t.full((n - logits_t.shape[0],), -1e9, device=logits_t.device, dtype=logits_t.dtype)
                                logits_t = _t.cat([logits_t, pad], dim=0)
                            logits_t = logits_t[:n]
                            logits = logits_t.tolist()
                        else:
                            logits = list(logits_t)[:n]
                            if len(logits) < n:
                                logits += [-1e9] * (n - len(logits))
                        # value（固定ヘッド）: ロジットベクター → Sigmoid で確率ベクターへ
                        v_vec = value_vec_b[i]
                        try:
                            import torch as _t
                            if hasattr(v_vec, 'sigmoid'):
                                v_probs = v_vec.sigmoid().detach().cpu().tolist()
                            else:
                                vv = v_vec.tolist() if hasattr(v_vec, 'tolist') else list(v_vec)
                                import math as _m
                                v_probs = [float(1.0/(1.0+_m.exp(-float(x)))) for x in vv]
                        except Exception:
                            try:
                                vv = v_vec.tolist() if hasattr(v_vec, 'tolist') else list(v_vec)
                                import math as _m
                                v_probs = [float(1.0/(1.0+_m.exp(-float(x)))) for x in vv]
                            except Exception:
                                # 最終フォールバック: 自分視点のみ
                                pid = getattr(self, 'player_id', 0)
                                try:
                                    v_probs = [float(v_vec[pid])]
                                except Exception:
                                    v_probs = [0.5]
                        # softmax
                        import math as _m
                        mx = max(logits) if logits else 0.0
                        exps = [_m.exp(x - mx) for x in logits]
                        s = sum(exps)
                        probs = [e_ / s for e_ in exps] if s > 0 else [1.0 / n] * n
                        outs.append(({legal[j]: probs[j] for j in range(n)}, v_probs))
                    return outs
            except Exception:
                pass
            # 失敗時は逐次版へフォールバック
            return [self._policy_value(e) for e in env_list]

        

        # Determinization (imperfect information + pass 制約)
        det_enable = bool(self.config.get('enable_determinization', True))
        root_pid = getattr(env.game, "turn", 0)
        # 学習/推論でモード切替
        det_mode = (self.config.get('determinization_mode_train', 'fixed_once') if training
                    else self.config.get('determinization_mode_eval', 'stochastic'))

        # fixed_once 用テンプレート（1回サンプルした割当を全シミュレーションに固定適用）
        template_assignment = None
        if det_enable and det_mode == 'fixed_once':
            try:
                ok, assign = self._build_single_determinization(env_copy, env, root_pid, apply_direct=False)
                if ok:
                    template_assignment = assign
            except Exception:
                template_assignment = None

        def _apply_assignment_fixed(e_clone, assignment, root_pid_):
            try:
                g_new = e_clone.game
                from game.card import Card
                for pid, cards_str in assignment.get('hands', {}).items():
                    if pid == root_pid_:
                        continue
                    try:
                        g_new.players[pid].hand = [Card.from_string(s) if hasattr(Card,'from_string') else Card(s) for s in cards_str]
                    except Exception:
                        g_new.players[pid].hand = list(cards_str)
            except Exception:
                pass

        def _determinize(e_clone, original_env, root_pid_):
            if not det_enable or det_mode == 'none':
                return
            if det_mode == 'fixed_once' and template_assignment is not None:
                _apply_assignment_fixed(e_clone, template_assignment, root_pid_)
                return
            # stochastic: プール→フォールバックの順で適用
            if self.config.get('enable_parallel_determinization', True):
                used = self._apply_from_det_pool(e_clone, original_env, root_pid_)
                if used:
                    return
            self._det_stats['pool_fallback_inline'] = self._det_stats.get('pool_fallback_inline',0) + 1
            self._inline_determinize(e_clone, original_env, root_pid_)

        # 推論時は Dirichlet を既定で無効化
        add_dirichlet_flag = True if training else bool(self.config.get('inference_dirichlet', False))

        # --- 学習時だけ: 学習対象プレイヤー以外のMCTS探索を間引く ---
        # 既定: 学習プレイヤーは self.num_simulations のまま。
        #       それ以外は scale(例: 1/8) もしくは固定値(opponent_num_simulations)を使用。
        sims_to_run = int(self.num_simulations)
        try:
            if training:
                learn_pid = int(self.config.get("learning_player_id", 0) or 0)
                if int(self.player_id) != learn_pid:
                    # 絶対指定があれば優先
                    opp_abs = int(self.config.get("opponent_num_simulations", 0) or 0)
                    if opp_abs > 0:
                        sims_to_run = max(1, opp_abs)
                    else:
                        scale = float(self.config.get("opponent_sim_scale", 0.125) or 0.125)
                        min_sim = int(self.config.get("opponent_sim_min", 1) or 1)
                        sims_to_run = max(min_sim, int(round(sims_to_run * max(0.0, scale))))
        except Exception:
            # 何かあっても既定の探索数で継続
            sims_to_run = int(self.num_simulations)

        root = run_puct_mcts(
            root_env_copy=env_copy,
            num_simulations=sims_to_run,
            policy_value_fn=policy_value_fn,
            policy_value_batch_fn=policy_value_batch_fn,
            get_legal_actions_fn=legal_fn,
            c_puct=self.puct_c,
            add_dirichlet=add_dirichlet_flag,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            root_player_id=root_pid,
            batch_eval_size=self.mcts_batch_eval_size,
            transposition_table=TT,
            determinize_fn=_determinize if det_enable else None,
            early_stop_enable=bool(self.config.get('mcts_early_stop_enable', False)),
            early_stop_min_sims=int(self.config.get('mcts_early_stop_min_sims', 16)),
            early_stop_visit_ratio=float(self.config.get('mcts_early_stop_visit_ratio', 0.75)),
            early_stop_gap_ratio=float(self.config.get('mcts_early_stop_gap_ratio', 0.10)),
            early_stop_log_sample_rate=float(self.config.get('mcts_early_stop_log_sample_rate', 0.0)),
            early_stop_post_min_batch=int(self.config.get('mcts_early_stop_post_min_batch', 0) or 0),
            early_stop_debug=bool(self.config.get('mcts_early_stop_debug', False)),
            early_stop_logger=self.logger,
        )
        # 付加情報（計測）: ルートに付与できない場合があるため self.* にも保存
        _det_mode_now = (
            self.config.get('determinization_mode_train', 'fixed_once') if training
            else self.config.get('determinization_mode_eval', 'stochastic')
        )
        try:
            setattr(root, '_perf_clone_ms', float(_clone_ms))
        except Exception:
            pass
        try:
            setattr(root, '_perf_infer_ms', float(getattr(self, '_perf_infer_ms_accum', 0.0)))
        except Exception:
            pass
        try:
            setattr(root, '_perf_det_mode', _det_mode_now)
        except Exception:
            pass
        # フォールバック保持
        try:
            self._last_perf_det_mode = str(_det_mode_now)
        except Exception:
            pass
        return root

    def _policy_value(self, env):
        """1 環境に対する policy 分布と value を返す。"""
        # If remote inference queue is attached, delegate single-eval to master
        try:
            rq = getattr(self, '_remote_request_q', None)
            rsp_q = getattr(self, '_remote_response_q', None)
            wid = getattr(self, '_remote_worker_id', None)
            if rq is not None and rsp_q is not None and wid is not None:
                # build lightweight payload (feature/state dict)
                try:
                    payload = self._extract_state(env)
                except Exception:
                    payload = None
                try:
                    # generate a robust unique request id
                    req_id = uuid.uuid4().hex
                    import time as _tperf
                    _t0 = _tperf.time()
                    try:
                        rq.put((int(wid), req_id, payload))
                    except Exception:
                        pass
                    # wait for response
                    timeout = float(self.config.get('remote_infer_timeout', 5.0) or 5.0)
                    start = time.time()
                    while True:
                        try:
                            rid, out = rsp_q.get(timeout=0.5)
                            if rid == req_id:
                                # out expected shapes:
                                #   legacy: (policy_dict, value)
                                #   extended: (policy_dict, value, extras)
                                pol_raw, val_raw = None, None
                                extras = None
                                try:
                                    pol_raw = out[0]; val_raw = out[1]
                                    extras = out[2] if (isinstance(out, tuple) and len(out) >= 3) else None
                                except Exception:
                                    pass
                                # 計測: 遠隔推論の往復遅延を1コールとして加算
                                try:
                                    dt_ms = (_tperf.time() - _t0) * 1000.0
                                    self._perf_infer_ms_accum += float(dt_ms)
                                    self._perf_infer_calls += 1
                                    if bool(self.config.get('measure_forward_time', False)):
                                        alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                                        if self._forward_time_ms_ema is None:
                                            self._forward_time_ms_ema = float(dt_ms)
                                        else:
                                            self._forward_time_ms_ema = float(alpha * dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                                except Exception:
                                    pass
                                # cache hand_probs if present
                                try:
                                    if isinstance(extras, dict) and isinstance(extras.get('hand_probs'), (list, tuple)):
                                        self._remote_hand_probs = list(extras.get('hand_probs'))
                                except Exception:
                                    pass
                                # remap index->prob to legal actions if keys are indices
                                try:
                                    legal_local = self._get_legal_actions(env)
                                except Exception:
                                    legal_local = []
                                mapped = {}
                                try:
                                    if legal_local and pol_raw:
                                        if hasattr(pol_raw, 'keys') and all(isinstance(k, int) for k in pol_raw.keys()):
                                            for j, a in enumerate(legal_local):
                                                try:
                                                    mapped[a] = float(pol_raw.get(j, 0.0))
                                                except Exception:
                                                    mapped[a] = 0.0
                                        else:
                                            mapped = pol_raw
                                except Exception:
                                    mapped = {}
                                return (mapped, val_raw)
                            # else ignore
                        except Exception:
                            # timeout on get -> check global timeout
                            if (time.time() - start) >= timeout:
                                break
                    # fallback to local if remote failed/timeout
                except Exception:
                    pass
        except Exception:
            pass
        state = self._extract_state(env)
        legal = self._get_legal_actions(env)
        n = len(legal)
        if n == 0:
            return {}, 0.0
        mdl = self.model
        if mdl is None:
            p = 1.0 / n
            return {a: p for a in legal}, 0.0

        # 可変長アクション対応モデル
        try:
            import torch as _t
            _dev = getattr(mdl, 'device', None)
            _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
            with _t.no_grad():
                if getattr(mdl, 'supports_variable_actions', False) and hasattr(mdl, 'evaluate'):
                    # 可変長アクションモデルの逐次評価
                    # 期待: value は各プレイヤーのロジット/確率ベクター
                    # 後方互換: スカラーしか返らない場合は N 人分に複製
                    import time as _tperf
                    _t0 = _tperf.time()
                    logits, value_out = mdl.evaluate(state, legal)
                    try:
                        _dt_ms = (_tperf.time() - _t0) * 1000.0
                        self._perf_infer_ms_accum += _dt_ms
                        self._perf_infer_calls += 1
                        if bool(self.config.get('measure_forward_time', False)):
                            alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                            try:
                                if self._forward_time_ms_ema is None:
                                    self._forward_time_ms_ema = float(_dt_ms)
                                else:
                                    self._forward_time_ms_ema = float(alpha * _dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # value_out をベクターへ
                    try:
                        if isinstance(value_out, dict):
                            # dict のまま渡す（MCTS 側で to_play 成分を抽出）
                            value_scalar = {int(k): float(v) for k, v in value_out.items()}
                        elif isinstance(value_out, (list, tuple)):
                            # 既にベクターならそのまま（必要に応じ Sigmoid はモデル側）
                            value_scalar = [float(x) for x in value_out]
                        else:
                            # スカラー → N 人分に複製
                            try:
                                N = int(len(getattr(env.game, 'players', [])) or 4)
                            except Exception:
                                N = 4
                            value_scalar = [float(value_out)] * N
                    except Exception:
                        value_scalar = float(value_out)
                else:
                    # 固定ヘッドはAMPでバッチ/単発推論
                    import time as _tperf
                    with _t.amp.autocast('cuda', enabled=_use_amp):
                        _t0 = _tperf.time()
                        logits_t, value_vec_t = mdl.forward(state)  # tensors
                    try:
                        _dt_ms = (_tperf.time() - _t0) * 1000.0
                        self._perf_infer_ms_accum += _dt_ms
                        self._perf_infer_calls += 1
                        if bool(self.config.get('measure_forward_time', False)):
                            alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                            try:
                                if self._forward_time_ms_ema is None:
                                    self._forward_time_ms_ema = float(_dt_ms)
                                else:
                                    self._forward_time_ms_ema = float(alpha * _dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if hasattr(logits_t, 'shape') and logits_t.shape[0] < n:  # 念のためパディング（不可視化: -1e9）
                        pad = _t.full((n - logits_t.shape[0],), -1e9, device=getattr(logits_t, 'device', None), dtype=getattr(logits_t, 'dtype', None) or None)
                        logits_t = _t.cat([logits_t, pad], dim=0)
                    logits_t = logits_t[:n]
                    # value_vec_t はロジット。全プレイヤー分を Sigmoid に通して返す。
                    try:
                        value_scalar = value_vec_t.sigmoid().detach().cpu().tolist()
                    except Exception:
                        v_raw = value_vec_t.tolist() if hasattr(value_vec_t,'tolist') else list(value_vec_t)
                        import math as _m
                        value_scalar = [float(1.0/(1.0+_m.exp(-float(x)))) for x in v_raw]
                    logits = logits_t.tolist() if hasattr(logits_t, 'tolist') else list(logits_t)
        except Exception:
            # フォールバック（従来通り）
            if getattr(mdl, 'supports_variable_actions', False) and hasattr(mdl, 'evaluate'):
                import time as _tperf
                _t0 = _tperf.time()
                logits, value_out = mdl.evaluate(state, legal)
                try:
                    _dt_ms = (_tperf.time() - _t0) * 1000.0
                    self._perf_infer_ms_accum += _dt_ms
                    self._perf_infer_calls += 1
                    if bool(self.config.get('measure_forward_time', False)):
                        alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                        try:
                            if self._forward_time_ms_ema is None:
                                self._forward_time_ms_ema = float(_dt_ms)
                            else:
                                self._forward_time_ms_ema = float(alpha * _dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    if isinstance(value_out, dict):
                        value_scalar = {int(k): float(v) for k, v in value_out.items()}
                    elif isinstance(value_out, (list, tuple)):
                        value_scalar = [float(x) for x in value_out]
                    else:
                        N = int(len(getattr(env.game, 'players', [])) or 4)
                        value_scalar = [float(value_out)] * N
                except Exception:
                    value_scalar = float(value_out)
            else:
                import time as _tperf
                _t0 = _tperf.time()
                logits_t, value_vec_t = mdl.forward(state)
                try:
                    _dt_ms = (_tperf.time() - _t0) * 1000.0
                    self._perf_infer_ms_accum += _dt_ms
                    self._perf_infer_calls += 1
                    if bool(self.config.get('measure_forward_time', False)):
                        alpha = float(self.config.get('measure_ema_alpha', 0.3) or 0.3)
                        try:
                            if self._forward_time_ms_ema is None:
                                self._forward_time_ms_ema = float(_dt_ms)
                            else:
                                self._forward_time_ms_ema = float(alpha * _dt_ms + (1.0 - alpha) * self._forward_time_ms_ema)
                        except Exception:
                            pass
                except Exception:
                    pass
                if hasattr(logits_t, 'shape') and logits_t.shape[0] < n:
                    import torch as _t
                    pad = _t.full((n - logits_t.shape[0],), -1e9, device=getattr(logits_t, 'device', None), dtype=getattr(logits_t, 'dtype', None) or None)
                    logits_t = _t.cat([logits_t, pad], dim=0)
                logits_t = logits_t[:n]
                try:
                    value_scalar = value_vec_t.sigmoid().detach().cpu().tolist()
                except Exception:
                    v_raw = value_vec_t.tolist() if hasattr(value_vec_t,'tolist') else list(value_vec_t)
                    import math as _m
                    value_scalar = [float(1.0/(1.0+_m.exp(-float(x)))) for x in v_raw]
                logits = logits_t.tolist() if hasattr(logits_t, 'tolist') else list(logits_t)

        # softmax 正規化
        import math as _m
        mx = max(logits) if logits else 0.0
        exps = [_m.exp(x - mx) for x in logits]
        s = sum(exps)
        probs = [e / s for e in exps] if s > 0 else [1.0 / n] * n
        try:
            out = {legal[i]: probs[i] for i in range(n)}
        except Exception:
            # フォールバック: サイズ不一致など
            out = {a: (1.0/len(legal)) for a in legal}
            value_scalar = 0.5
        # value_scalar は list/dict/float を許容（MCTS 側で各ノードの to_play 視点を抽出）
        # If remote hand probs were cached and determinization wants them, expose via side-channel attr
        try:
            if getattr(self, '_remote_hand_probs', None) is not None:
                # Mark timestamp for freshness if needed later
                self._remote_hand_probs_ts = time.time()
        except Exception:
            pass
        return out, value_scalar

    # ---------------- Env helpers ----------------
    def _copy_env(self, env):
        """環境を shallow copy し、Game の最小限状態だけ複製した高速シミュレーション用コピーを生成."""
        base = env
        new_env = copy.copy(base)  # シェルコピー
        g = base.game
        g_new = copy.copy(g)       # Game オブジェクト浅いコピー (__init__ 不呼び出し)
        # Player hand はリストだけコピー (Card は参照共有で OK)
        new_players = []
        for p in g.players:
            p_new = copy.copy(p)
            p_new.hand = list(p.hand)
            new_players.append(p_new)
        g_new.players = new_players
        g_new.current_field = list(g.current_field)
        g_new.passed = list(g.passed)
        g_new.rankings = list(getattr(g, 'rankings', []))
        try:
            g_new.silent = True  # シミュレーション時の print 抑制
        except Exception:
            pass
        new_env.game = g_new
        return new_env

    # ------------ Determinization helpers (parallel pool) ------------
    def _maybe_start_det_pool(self, env):
        if self._det_pool is not None:
            return
        try:
            import threading
            from collections import deque as _dq
            self._det_pool = _dq(maxlen=self._det_cfg['capacity'])
            self._det_pool_lock = threading.Lock()
            self._det_stop_event = threading.Event()
            # ルート基準情報 (手番 / 革命) を記録し mismatch で破棄
            g = env.game
            self._det_root_signature = {
                'num_players': len(getattr(g, 'players', [])),
                'revo': bool(getattr(getattr(g,'rule_checker',None),'revolution',False)),
                'root_turn': getattr(g, 'turn', 0),
            }
            self._det_pool_thread = threading.Thread(target=self._determinization_worker, daemon=True)
            self._det_pool_thread.start()
            try:
                if self.logger:
                    self.logger.log_text(f"[det-pool-init] cap={self._det_cfg['capacity']} refill={self._det_cfg['refill_ratio']} sig={self._det_root_signature}")
            except Exception:
                pass
        except Exception as e:
            # 失敗時は無視してインライン動作へ (ただし可視化)
            try:
                print(f"[DetPool][ERROR] 起動失敗: {type(e).__name__}: {e}")
            except Exception:
                pass
            self._det_pool = None

    def _determinization_worker(self):
        import time, random as _r
        while self._det_stop_event and not self._det_stop_event.is_set():
            try:
                # 充足チェック
                with self._det_pool_lock:
                    cur_len = len(self._det_pool)
                    cap = self._det_cfg['capacity']
                refill_threshold = int(self._det_cfg['capacity'] * self._det_cfg['refill_ratio'])
                if cur_len >= cap or cur_len > refill_threshold and cur_len > 0:
                    time.sleep(0.002)
                    continue
                # 生成元となる env_ref から shallow copy して root 基準で determinization
                base_env = self.env_ref
                if base_env is None:
                    time.sleep(0.01)
                    continue
                env_clone = self._copy_env(base_env)
                ok, assignment = self._build_single_determinization(env_clone, base_env, getattr(base_env.game,'turn',0))
                if ok and assignment:
                    # 署名整合性判定
                    g = base_env.game
                    sig = {
                        'num_players': len(getattr(g, 'players', [])),
                        'revo': bool(getattr(getattr(g,'rule_checker',None),'revolution',False)),
                        'root_turn': getattr(g,'turn',0),
                    }
                    if sig != getattr(self, '_det_root_signature', sig):
                        self._det_stats['discard_mismatch'] += 1
                        # root_turn だけが変わっているケースでは署名を更新して再利用性を確保
                        try:
                            prev = getattr(self, '_det_root_signature', None)
                            if prev and prev.get('num_players') == sig['num_players'] and prev.get('revo') == sig['revo'] and prev.get('root_turn') != sig['root_turn']:
                                self._det_root_signature = sig  # ターン進行に追随
                                if self.logger and (self._det_stats['discard_mismatch'] % 50 == 1):
                                    self.logger.log_text(f"[det-pool-update] root_turn change prev={prev.get('root_turn')} new={sig['root_turn']} discards={self._det_stats['discard_mismatch']}")
                            else:
                                if self.logger and (self._det_stats['discard_mismatch'] % 100 == 1):
                                    self.logger.log_text(f"[det-pool-mismatch] discards={self._det_stats['discard_mismatch']} prev={prev} cur={sig}")
                        except Exception:
                            pass
                        time.sleep(0.001)
                        continue
                    with self._det_pool_lock:
                        if len(self._det_pool) < self._det_cfg['capacity']:
                            self._det_pool.append(assignment)
                            self._det_stats['generated'] += 1
                else:
                    # 軽く待機して再トライ (過剰ループ抑制)
                    time.sleep(0.001)
            except Exception as e:
                try:
                    print(f"[DetPool][ERROR] worker 例外: {type(e).__name__}: {e}")
                except Exception:
                    pass
                time.sleep(0.005)

    def _apply_from_det_pool(self, e_clone, original_env, root_pid) -> bool:
        if self._det_pool is None:
            return False
        try:
            import random as _r
            with self._det_pool_lock:
                if not self._det_pool:
                    return False
                if self._det_cfg['sampling'] == 'random':
                    idx = _r.randrange(len(self._det_pool))
                    # deque は index pop 不直接 -> 回転
                    for _ in range(idx):
                        self._det_pool.append(self._det_pool.popleft())
                    assignment = self._det_pool.popleft()
                else:
                    assignment = self._det_pool.popleft()
            # 適用
            g_new = e_clone.game
            from game.card import Card
            for pid, cards_str in assignment['hands'].items():
                if pid == root_pid:
                    continue
                try:
                    g_new.players[pid].hand = [Card.from_string(s) if hasattr(Card,'from_string') else Card(s) for s in cards_str]
                except Exception:
                    g_new.players[pid].hand = list(cards_str)
            self._det_stats['pool_hits'] += 1
            return True
        except Exception as e:
            try:
                print(f"[DetPool][ERROR] apply 失敗: {type(e).__name__}: {e}")
            except Exception:
                pass
            return False

    # --- Inline fallback determinization (直接適用) ---
    def _inline_determinize(self, e_clone, original_env, root_pid):
        self._build_single_determinization(e_clone, original_env, root_pid, apply_direct=True)

    # コア生成: apply_direct=False なら (ok, assignment_dict) を返す
    def _build_single_determinization(self, e_clone, original_env, root_pid, apply_direct=False):
        try:
            g_orig = original_env.game
            g_new = e_clone.game
            history = getattr(g_orig, '_action_history', []) or []
            root_hand_ids = {str(c) for c in g_orig.players[root_pid].hand}
            field_ids = {str(c) for c in g_orig.current_field}
            # デッキ総カード
            all_cards: list[str] = []
            try:
                if hasattr(g_orig, 'deck') and g_orig.deck:
                    all_cards = [str(c) for c in g_orig.deck]
            except Exception:
                pass
            if not all_cards:
                for p in g_orig.players:
                    all_cards.extend(str(c) for c in p.hand)
                all_cards.extend(field_ids)
                all_cards = list(dict.fromkeys(all_cards))
            known = set(root_hand_ids) | field_ids
            for rid in getattr(g_orig, 'rankings', []):
                if rid != root_pid:
                    known.update(str(c) for c in g_orig.players[rid].hand)
            for h in history:
                act = h.get('action')
                if isinstance(act, (list, tuple)):
                    known.update(str(c) for c in act)
            unknown_seed = [cid for cid in all_cards if cid not in known]
            # pass 制約抽出
            pass_reqs = []
            for h in history:
                if h.get('action') == 'pass':
                    fb = h.get('field_before') or []
                    if not fb:
                        continue
                    cnt = len(fb)
                    ranks = []
                    for cid in fb:
                        try:
                            core = cid[:-1]
                            ranks.append(int(core))
                        except Exception:
                            pass
                    base_rank = min(ranks) if ranks else 0
                    pass_reqs.append({
                        'pid': h.get('pid'), 'count': cnt, 'min_rank': base_rank,
                        'revo': bool(h.get('revo', False)), 'combo_type': h.get('combo_type'),
                        'size': cnt, 'ranks_ref': sorted(ranks),
                    })
            def rank_of(cid: str):
                core = cid[:-1]
                try:
                    return int(core)
                except Exception:
                    return 0
            import random as _r
            opp_ids = [i for i in range(len(g_new.players)) if i != root_pid]
            max_retry = max(1, int(self._det_cfg.get('retry_max', 8)))
            # Step4: 手札予測ヘッドによるガイド付き割当
            guided_enable = False
            hand_card_probs = None  # dict: opponent_pid -> list[53] (probabilities)
            try:
                mdl = self.model
                if bool(getattr(self.config, 'get', lambda k, d=None: d)('hand_pred_use_in_determinization', False)) or \
                   bool(self.config.get('hand_pred_use_in_determinization', False)):
                    # 予測ヘッドがモデルに存在するか確認
                    num_players_local = len(g_new.players)
                    remote_hp = getattr(self, '_remote_hand_probs', None)
                    fresh = True
                    try:
                        ttl = float(self.config.get('hand_pred_cache_ttl_sec', 0.5) or 0.5)
                        ts = float(getattr(self, '_remote_hand_probs_ts', 0.0) or 0.0)
                        import time as _time
                        if ttl > 0 and (ts <= 0 or (_time.time() - ts) > ttl):
                            fresh = False
                    except Exception:
                        fresh = True
                    if fresh and isinstance(remote_hp, (list, tuple)) and len(remote_hp) == 53 * (num_players_local - 1):
                        opponents_order = [i for i in range(num_players_local) if i != root_pid]
                        hand_card_probs = {}
                        for oi, pid_ in enumerate(opponents_order):
                            start = oi * 53
                            hand_card_probs[pid_] = list(remote_hp[start:start+53])
                        guided_enable = True
                    elif self.model is not None and getattr(self.model, 'enable_hand_prediction_head', False) and hasattr(self.model, 'forward_with_belief'):
                        state_root = self._extract_state(original_env)
                        import torch as _t
                        _dev = getattr(self.model, 'device', None)
                        _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
                        with _t.no_grad():
                            with _t.amp.autocast('cuda', enabled=_use_amp):
                                try:
                                    _, _, hand_logits_t = self.model.forward_with_belief(state_root)
                                except Exception:
                                    hand_logits_t = None
                        if hand_logits_t is not None:
                            try:
                                import torch as _t
                                if isinstance(hand_logits_t, _t.Tensor):
                                    hp = _t.sigmoid(hand_logits_t).detach().cpu().tolist()
                                else:
                                    import math as _m
                                    seq = (hand_logits_t.detach().cpu().tolist() if hasattr(hand_logits_t, 'detach') else (hand_logits_t.tolist() if hasattr(hand_logits_t, 'tolist') else list(hand_logits_t)))
                                    hp = [1.0/(1.0+_m.exp(-float(x))) for x in seq]
                            except Exception:
                                hp = list(hand_logits_t)
                            expected = 53 * (num_players_local - 1)
                            if len(hp) == expected:
                                opponents_order = [i for i in range(num_players_local) if i != root_pid]
                                hand_card_probs = {}
                                for oi, pid_ in enumerate(opponents_order):
                                    start = oi * 53
                                    hand_card_probs[pid_] = hp[start:start+53]
                                guided_enable = True
            except Exception:
                guided_enable = False
                hand_card_probs = None

            for attempt in range(max_retry):
                unknown = list(unknown_seed)
                guided_used = False
                if guided_enable and attempt == 0:
                    # ガイド付き: 各カードを opponent の確率に基づきサンプリング
                    try:
                        _r.shuffle(unknown)  # 順番ランダム化
                        # 容量 (各 opponent の所持枚数)
                        capacities = {i: len(g_new.players[i].hand) for i in opp_ids}
                        assignment_hands = {i: [] for i in opp_ids}
                        from game.card import Card
                        def _card_index_from_str(cid: str) -> int:
                            try:
                                # Card.from_string があれば利用
                                c_obj = Card.from_string(cid) if hasattr(Card, 'from_string') else Card(cid)
                                # _extract_state と同じロジック
                                if getattr(c_obj, 'is_joker', False):
                                    return 52
                                suit_order = {'\u2660':0,'\u2665':1,'\u2666':2,'\u2663':3, 'S':0,'H':1,'D':2,'C':3}
                                s = suit_order.get(getattr(c_obj,'suit','S'),0)
                                r = int(getattr(c_obj,'rank',1)) - 1
                                idx = s * 13 + r
                                if idx < 0 or idx >= 53:
                                    return 52
                                return idx
                            except Exception:
                                # 失敗時は Joker 扱いで 52 へフォールバック
                                return 52
                        for cid in unknown:
                            remaining_opps = [i for i in opp_ids if capacities[i] > 0]
                            if not remaining_opps:
                                break
                            idx = _card_index_from_str(cid)
                            # 生確率取得
                            probs_raw = []
                            total = 0.0
                            for pid_ in remaining_opps:
                                p_val = 0.0
                                try:
                                    p_list = hand_card_probs.get(pid_, None)
                                    if p_list and 0 <= idx < len(p_list):
                                        p_val = float(p_list[idx])
                                except Exception:
                                    p_val = 0.0
                                probs_raw.append(p_val)
                                total += p_val
                            if total <= 1e-12:
                                # 全ゼロ → 一様分布
                                probs_norm = [1.0/len(remaining_opps)] * len(remaining_opps)
                            else:
                                probs_norm = [p/total for p in probs_raw]
                            # サンプリング
                            r = _r.random()
                            cum = 0.0
                            chosen = remaining_opps[-1]
                            for j, pid_ in enumerate(remaining_opps):
                                cum += probs_norm[j]
                                if r <= cum:
                                    chosen = pid_
                                    break
                            assignment_hands[chosen].append(cid)
                            capacities[chosen] -= 1
                        # 足りない場合 (容量残り) → 余った unknown から再配分
                        leftover_unknown = [cid for cid in unknown if all(cid not in v for v in assignment_hands.values())]
                        if leftover_unknown:
                            for cid in leftover_unknown:
                                targets = [i for i in opp_ids if capacities[i] > 0]
                                if not targets:
                                    break
                                choice = _r.choice(targets)
                                assignment_hands[choice].append(cid)
                                capacities[choice] -= 1
                        # 各 opponent の枚数が期待通りか簡易検証; 不整合ならガイド失敗として通常ランダムへフォールバック
                        mismatch = False
                        for i in opp_ids:
                            if len(assignment_hands[i]) != len(g_new.players[i].hand):
                                mismatch = True
                                break
                        if not mismatch:
                            guided_used = True
                            self._det_stats['guided_assignments'] = self._det_stats.get('guided_assignments', 0) + 1
                        else:
                            guided_used = False
                    except Exception:
                        guided_used = False
                if not guided_used:
                    _r.shuffle(unknown)
                    cursor = 0
                    assignment_hands = {}
                    for i in opp_ids:
                        p_new = g_new.players[i]
                        sz = len(p_new.hand)
                        pick = unknown[cursor:cursor+sz]
                        cursor += sz
                        assignment_hands[i] = list(pick)
                # 矛盾検査
                consistent = True
                for req in pass_reqs:
                    pidc = req['pid']
                    if pidc == root_pid or pidc is None:
                        continue
                    hand_ids = set(assignment_hands.get(pidc, []))
                    counts = {}
                    for cid in hand_ids:
                        r = rank_of(cid)
                        counts[r] = counts.get(r, 0) + 1
                    combo_type = req.get('combo_type')
                    feas = []
                    if combo_type == 'straight':
                        size_req = req.get('size', req['count'])
                        min_rank_ref = req['min_rank']
                        ranks_sorted = sorted(counts.keys())
                        if ranks_sorted:
                            for idx in range(len(ranks_sorted)):
                                window = ranks_sorted[idx:idx+size_req]
                                if len(window) < size_req:
                                    break
                                if all(window[i+1]-window[i] == 1 for i in range(size_req-1)):
                                    if req['revo']:
                                        if window[0] <= min_rank_ref:
                                            feas.append(tuple(window))
                                    else:
                                        if window[0] >= min_rank_ref:
                                            feas.append(tuple(window))
                    else:
                        if not req['revo']:
                            feas = [r for r,cnt in counts.items() if r >= req['min_rank'] and cnt >= req['count']]
                        else:
                            feas = [r for r,cnt in counts.items() if r <= req['min_rank'] and cnt >= req['count']]
                    if feas:
                        consistent = False
                        break
                if consistent:
                    # 適用または返却
                    if apply_direct:
                        from game.card import Card
                        for pid, cards_str in assignment_hands.items():
                            if pid == root_pid:
                                continue
                            try:
                                g_new.players[pid].hand = [Card.from_string(s) if hasattr(Card,'from_string') else Card(s) for s in cards_str]
                            except Exception:
                                g_new.players[pid].hand = list(cards_str)
                        return True, None
                    else:
                        self._det_stats['retries_total'] += attempt
                        return True, {'hands': assignment_hands}
            # 全リトライ失敗 -> 最後を採用 (安全側)
            if apply_direct:
                from game.card import Card
                for pid, cards_str in assignment_hands.items():
                    if pid == root_pid:
                        continue
                    try:
                        g_new.players[pid].hand = [Card.from_string(s) if hasattr(Card,'from_string') else Card(s) for s in cards_str]
                    except Exception:
                        g_new.players[pid].hand = list(cards_str)
                return True, None
            else:
                return False, None
        except Exception as e:
            try:
                print(f"[DetBuild][ERROR] 例外: {type(e).__name__}: {e}")
            except Exception:
                pass
            return False, None

    def _get_legal_actions(self, env) -> List[Any]:
        """現在手番プレイヤーの合法手集合を可変長リストで返す。

        変更点:
                # If remote inference queue is attached, delegate to master in per-request fashion
                rq = getattr(self, '_remote_request_q', None)
                rsp_q = getattr(self, '_remote_response_q', None)
                wid = getattr(self, '_remote_worker_id', None)
                if rq is not None and rsp_q is not None and wid is not None:
                    outs = [None] * len(env_list)
                    req_ids = []
                    # send all requests with robust uuid ids
                    for i, e in enumerate(env_list):
                        try:
                            payload = self._extract_state(e)
                        except Exception:
                            payload = None
                        req_id = uuid.uuid4().hex
                        req_ids.append(req_id)
                        try:
                            rq.put((int(wid), req_id, payload))
                        except Exception:
                            pass
                    # collect responses with timeout
                    timeout = float(self.config.get('remote_infer_timeout', 5.0) or 5.0)
                    start = time.time()
                    received = 0
                    while received < len(env_list) and (time.time() - start) < timeout:
                        try:
                            rid, out = rsp_q.get(timeout=0.5)
                        except Exception:
                            continue
                        try:
                            # match response to local index by req id
                            if rid in req_ids:
                                idx = req_ids.index(rid)
                                outs[idx] = out
                                received += 1
                        except Exception:
                            pass
                    # For any missing responses, fallback to local evaluation
                    for i in range(len(env_list)):
                        if outs[i] is None:
                            try:
                                # compute locally
                                outs[i] = self._policy_value(env_list[i])
                            except Exception:
                                outs[i] = ({}, 0.0)
                    return outs
                if self.model is None:
          - 合法手は正規化・重複排除後、カノニカル順に安定ソートする。
        """
        try:
            cur = env.game.players[env.game.turn]
            hand = cur.hand  # noqa: F841 (説明目的: hand を使って合法手生成)
            field = env.game.current_field[:]  # noqa: F841
            if hasattr(env, "_generate_legal_actions"):
                raw = env._generate_legal_actions(hand, field)
            else:
                raw = []
        except Exception:
            raw = []
        acts: List[Any] = []
        seen = set()
        field_is_empty = False
        try:
            field_is_empty = (len(env.game.current_field) == 0)
        except Exception:
            field_is_empty = False
        for a in raw:
            if a is None:
                continue
            try:
                tup = tuple(str(c) for c in a)  # カードオブジェクトを文字列化して重複排除
            except Exception:
                tup = a
            if tup not in seen:
                seen.add(tup)
                acts.append(tup)
        # Canonical sorting（pass は最後に付与）
        def _canon_key(tup_action):
            # pass は最も後ろに配置
            if tup_action == "pass":
                return (99, 0, 0, 1, ("ZZZ",))
            # 役分類ベースの安定キー
            try:
                from game.card import Card
                rc = getattr(env.game, 'rule_checker', None)
                cards = [Card.from_string(s) for s in tup_action] if isinstance(tup_action, tuple) else []
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
                id_key = tuple(sorted(tup_action)) if isinstance(tup_action, tuple) else (str(tup_action),)
                return (type_order.get(ctype, 98), int(size or 0), int(strength or 0), int(jokers or 0), id_key)
            except Exception:
                id_key = tuple(sorted(tup_action)) if isinstance(tup_action, tuple) else (str(tup_action),)
                return (97, 0, 0, 0, id_key)

        acts_sorted = sorted(acts, key=_canon_key)
        # パスは「場が空でない場合」にのみ合法手として許可する（最後に付加）
        if not field_is_empty and ("pass" not in acts_sorted):
            acts_sorted.append("pass")
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
        """部分観測用特徴量 (Belief 入力削除後のフル特徴レイアウト)。

        新レイアウト (rank クラス 4 種: daifugo,fugo,hinmin,daihinmin):
            - Self(55): 自分手札53bit + pass1 + remain_norm1
            - OppSummary(5*(N-1)): 各 opponent の remain_norm1 + rank one-hot(4)
            - Field(22): revolution1 + combo7 + base_rank13 + field_size_norm1
            - FieldCards(53): 現在場に出ている具体カードビット
            - OpponentDiscards(53*(N-1)): 各相手がこれまでに場へ出したカードフラグ
            - PassMatrix(13*(N-1)): パスで示唆された“到達不能”ランクの近似フラグ
            - Turn(N): 手番 one-hot

        合計次元 (Belief と PlayHistory を除去し OpponentDiscards/PassMatrix を追加):
            expected_full_dim = 72 * num_players + 59

        備考:
            - hand_labels (53*(N-1)) は引き続き教師として生成し、モデルの hand_head 出力ロジットに対し
              BCEWithLogitsLoss で使用する。
            - 旧 Belief ベクトルは生成しない。
        """
        try:
            g = env.game
            pid = g.turn
            rule_checker = getattr(g, "rule_checker", None)
            revo = bool(getattr(rule_checker, "revolution", False)) if rule_checker else False
            use_full = bool(getattr(self, 'config', {}).get('use_full_features', False))
            base = {"turn": pid}
            me = g.players[pid]
            base.update({
                "hand_size": len(getattr(me,'hand',[])),
                "field_size": len(g.current_field),
                "revolution": revo,
            })
            if not use_full:
                return base
            num_players = len(g.players)
            # 追加ブロック: OpponentDiscards 53*(N-1) + PassMatrix 13*(N-1)
            opp_discards_dim = 53 * (num_players - 1)
            pass_matrix_dim = 13 * (num_players - 1)
            # 新期待次元（Belief と PlayHistory を除去）: 72N + 59
            expected_full_dim = 72 * num_players + 59

            def card_index(card):
                try:
                    if getattr(card,'is_joker',False):
                        return 52
                    suit_order = {'\u2660':0,'\u2665':1,'\u2666':2,'\u2663':3, 'S':0,'H':1,'D':2,'C':3}
                    return suit_order.get(getattr(card,'suit','S'),0)*13 + (int(getattr(card,'rank',1))-1)
                except Exception:
                    return 52

            # Self block
            self_bits = [0.0]*53
            for c in getattr(me,'hand',[]):
                try:
                    idx = card_index(c)
                    if 0 <= idx < 53:
                        self_bits[idx] = 1.0
                except Exception:
                    pass
            self_pass = 1.0 if (pid < len(getattr(g,'passed',[])) and getattr(g,'passed')[pid]) else 0.0
            self_remain = len(getattr(me,'hand',[]))/53.0
            feat = self_bits + [self_pass, self_remain]

            # Opponent summaries
            # 階級ラベル: 4 クラス (平民を除外) → one-hot 長さ 4
            rank_labels = ['daifugo','fugo','hinmin','daihinmin']
            def encode_rank(pl):
                one = [0.0]*4
                try:
                    rc = getattr(pl,'rank_class',None)
                    if rc is None and hasattr(pl,'rank'):
                        rc = getattr(pl,'rank')
                    if isinstance(rc,str) and rc.lower() in rank_labels:
                        one[rank_labels.index(rc.lower())] = 1.0
                    elif isinstance(rc,int) and 0 <= rc < 4:
                        one[rc] = 1.0
                except Exception:
                    pass
                return one
            opponents = [i for i in range(num_players) if i != pid]
            opp_remains = {i: len(g.players[i].hand) for i in opponents}
            # ラベル作成（学習用・相手手札の実ラベル）
            try:
                if bool(getattr(self, 'config', {}).get('enable_hand_prediction_head', False)) and \
                   float(getattr(self, 'config', {}).get('hand_pred_loss_coef', 0.0) or 0.0) > 0.0:
                    hand_labels = []  # 53*(N-1) の 0/1
                    for i in opponents:
                        vec = [0.0] * 53
                        for c in getattr(g.players[i], 'hand', []):
                            try:
                                idx = card_index(c)
                                if 0 <= idx < 53:
                                    vec[idx] = 1.0
                            except Exception:
                                pass
                        hand_labels.extend(vec)
                    base['hand_labels'] = hand_labels
                    base['hand_labels_dim'] = len(hand_labels)
            except Exception:
                pass
            for i in opponents:
                feat.append(opp_remains[i]/53.0)
                feat.extend(encode_rank(g.players[i]))

            # Field block
            field = g.current_field
            combo_type_onehot = [0.0]*7
            rank_onehot = [0.0]*13
            field_size = len(field)
            if field_size == 0:
                combo_type_onehot[0] = 1.0
                base_rank = None
            else:
                combo = rule_checker.classify_combo(field) if rule_checker else None
                ctype = combo['type'] if combo else None
                mapping = {'single':1,'pair':2,'triple':3,'four':4,'straight':5,'joker_single':6}
                if ctype in mapping:
                    combo_type_onehot[mapping[ctype]] = 1.0
                base_rank = None
                if combo:
                    if ctype in ('single','pair','triple','four') and combo.get('rank') is not None:
                        base_rank = combo['rank']
                    elif ctype == 'straight':
                        ranks = combo.get('ranks', [])
                        base_rank = ranks[0] if ranks else None
                if base_rank is not None and 1 <= base_rank <= 13:
                    rank_onehot[base_rank-1] = 1.0
            revolution_bit = 1.0 if revo else 0.0
            field_size_norm = field_size/13.0
            feat.extend([revolution_bit] + combo_type_onehot + rank_onehot + [field_size_norm])
            # FieldCards (53)
            field_bits = [0.0]*53
            for c in field:
                try:
                    idx = card_index(c)
                    if 0 <= idx < 53:
                        field_bits[idx] = 1.0
                except Exception:
                    pass
            feat.extend(field_bits)
            # PlayHistory 入力は削除

            # OpponentDiscards (53*(N-1)) プレイヤー別に「その相手が場に出したことのあるカード」フラグ
            # 対象: 自分以外の相手。play_history から自分以外の行動を抽出。
            opp_discards_map = {i: [0.0]*53 for i in opponents}
            try:
                action_hist = list(getattr(g, '_action_history', []) or [])
            except Exception:
                action_hist = []
            for h in action_hist:
                try:
                    pid_act = h.get('pid')
                    act = h.get('action')
                    if pid_act in opp_discards_map and act not in (None, 'pass'):
                        # action はカードリスト (tuple/list) または単体カード
                        cards_iter = act if isinstance(act, (list, tuple)) else [act]
                        for c in cards_iter:
                            try:
                                idx = card_index(c)
                                if 0 <= idx < 53:
                                    opp_discards_map[pid_act][idx] = 1.0
                            except Exception:
                                pass
                except Exception:
                    continue
            # 出力順は opponents の順番で連結
            for i in opponents:
                feat.extend(opp_discards_map[i])

            # PassMatrix (13*(N-1)) ランク別に「その相手がそのランク以上(革命時は以下)に対してパスした」近似情報
            # 方針: pass した直前の場 field_before から基準ランク base_rank を推定し、
            # single/pair/triple/four のみ対象。革命時は大小比較を反転。
            pass_matrix_map = {i: [0.0]*13 for i in opponents}
            def _rank_str_to_int(s: str):
                try:
                    core = s[1:]
                    mp = {'A':1,'J':11,'Q':12,'K':13}
                    return mp.get(core.upper(), int(core))
                except Exception:
                    return None
            for h in action_hist:
                try:
                    if h.get('action') != 'pass':
                        continue
                    pid_pass = h.get('pid')
                    if pid_pass not in pass_matrix_map:
                        continue
                    fb = h.get('field_before') or []
                    if not fb:
                        continue
                    combo_type = h.get('combo_type')
                    if combo_type not in (None,'single','pair','triple','four','joker_single'):
                        continue  # straight などは除外
                    base_rank = None
                    ranks_tmp = []
                    for s in fb:
                        r = _rank_str_to_int(s)
                        if r is not None:
                            ranks_tmp.append(r)
                    if ranks_tmp:
                        base_rank = min(ranks_tmp)
                    if base_rank is None:
                        continue
                    revo_flag = bool(h.get('revo', False))
                    # パス時点で「出せたはずの（勝てる）ランク」集合を近似：non-revo で base_rank より上、revo で下
                    if not revo_flag:
                        target_ranks = [r for r in range(base_rank+1, 14)]
                    else:
                        target_ranks = [r for r in range(1, base_rank)]
                    for r in target_ranks:
                        if 1 <= r <= 13:
                            pass_matrix_map[pid_pass][r-1] = 1.0
                    # Joker は 52 index だが PassMatrix は 13 ランクのみを扱う設計
                except Exception:
                    continue
            for i in opponents:
                feat.extend(pass_matrix_map[i])

            # Belief 分布は入力から削除済みのため、ここでの計算・付加は行わない

            # Turn one-hot
            turn_onehot = [0.0]*num_players
            if 0 <= pid < num_players:
                turn_onehot[pid] = 1.0
            feat.extend(turn_onehot)

            cur_len = len(feat)
            if cur_len != expected_full_dim:
                if cur_len < expected_full_dim:
                    feat.extend([0.0]*(expected_full_dim - cur_len))
                else:
                    del feat[expected_full_dim:]
                if not hasattr(self,'_warned_full_dim_autofix'):
                    print(f"[WARN] adjusted full_input length from {cur_len} to expected {expected_full_dim} (layout: no Belief/PlayHistory + OpponentDiscards + PassMatrix)")
                    self._warned_full_dim_autofix = True
            base['full_input'] = feat
            base['full_input_dim'] = expected_full_dim
            base['full_input_version'] = 6
            return base
        except Exception:
            return {"turn": 0}

    # ---------------- Replay buffer ----------------
    def _store_sample(self, state, legal_actions, pi, value, value_pred: Optional[float] = None):
        """リプレイサンプル1件を保存。共有バッファなら append の参照を返す."""
        # ==============================================================
        # Lossless モード (strict_lossless=True):
        #   一切の量子化 / packbits 圧縮 / ID 化を行わず、元の float / 配列 / legal_actions を保持する。
        #   これにより復元時に情報欠落の可能性がゼロになる代わりにメモリ使用量が増える。
        #   想定用途: 正確な解析 / デバッグ / 再現性重視フェーズ。
        # --------------------------------------------------------------
        if self.config.get('strict_lossless', False):
            sample = {
                "player_id": self.player_id,
                "state": state,  # full_input をそのまま保持
                "legal_actions": legal_actions,
                "pi": list(pi) if pi is not None else None,  # そのまま float 配列
                "value": value,
                "value_pred": value_pred,
                "model_version": getattr(self, 'model_version', 0),
                "feature_version": state.get('full_input_version', 1) if (isinstance(state, dict) and ('full_input' in state or 'full_compact' in state)) else 0,
                "lossless": True,
            }
            # フル特徴量モード時に legacy を格納しないポリシーは維持
            if self.config.get('use_full_features') and sample['feature_version'] == 0:
                return sample
            if self._use_shared and hasattr(self.replay_buffer, 'append'):
                # 非同期挿入が有効なら append_async を使用
                try:
                    if bool(self.config.get('replay_async_enabled', False)) and hasattr(self.replay_buffer, 'append_async'):
                        drop_oldest = bool(self.config.get('replay_async_drop_oldest', True))
                        self.replay_buffer.append_async(sample, drop_oldest=drop_oldest)
                    else:
                        self.replay_buffer.append(sample)
                except Exception:
                    try:
                        self.replay_buffer.append(sample)
                    except Exception:
                        pass
                return sample
            # ローカル (list / deque) 互換処理
            try:
                from collections import deque as _dq
                if isinstance(self.replay_buffer, _dq):
                    self.replay_buffer.append(sample)
                else:
                    if len(self.replay_buffer) >= self.max_buffer_size:
                        self.replay_buffer.pop(0)
                    self.replay_buffer.append(sample)
            except Exception:
                pass
            return sample
        # ==============================================================
        # --- メモリ削減: full_input をコンパクト表現へ圧縮 (packbits + float16) ---
        try:
            # 圧縮方針:
            #   1) enable_compact_full_input が True なら、store_full_input の値に関わらず
            #      まず full_input から full_compact を生成（未生成時のみ）。
            #   2) store_full_input が False の場合は raw full_input を削除し、圧縮表現のみ保持して学習可能にする。
            #      以前は両方削除して feature_version=0 になり学習が停止してしまっていたため修正。
            if (self.config.get('use_full_features') and
                self.config.get('enable_compact_full_input', True) and
                isinstance(state, dict) and 'full_input' in state and 'full_compact' not in state):
                fi = state.get('full_input')
                import numpy as _np
                fi_arr = _np.asarray(fi, dtype=_np.float32)
                total_len = fi_arr.shape[0]
                # Only support v4 (59N + 125). Older layouts (v1-v3) are no longer compressed.
                layout_version = None
                N = None
                if total_len >= 125:
                    cand = (total_len - 125) / 59
                    if abs(cand - int(cand)) < 1e-6 and 2 <= int(cand) <= 10 and 59 * int(cand) + 125 == total_len:
                        layout_version = 4
                        N = int(cand)
                if layout_version is None:
                    # 不一致なら圧縮スキップ
                    pass
                else:
                    binary_indices = []
                    float_indices = []
                    cursor = 0
                    # Self block (共通) 53 bits + pass(bit) + remain(float)
                    binary_indices.extend(range(cursor, cursor + 53))
                    binary_indices.append(cursor + 53)
                    float_indices.append(cursor + 54)
                    cursor += 55
                    # OppSummary (v4 rank4)
                    opp_cnt = N - 1
                    per_opp = 1 + 4
                    for _ in range(opp_cnt):
                        float_indices.append(cursor)
                        binary_indices.extend(range(cursor + 1, cursor + 1 + 4))
                        cursor += per_opp
                    # Field block (22)
                    if cursor + 22 <= total_len:
                        # revolution + combo7 + rank13 + field_size_norm
                        binary_indices.append(cursor); cursor += 1
                        binary_indices.extend(range(cursor, cursor + 7)); cursor += 7
                        binary_indices.extend(range(cursor, cursor + 13)); cursor += 13
                        float_indices.append(cursor); cursor += 1
                        # FieldCards 53 + PlayHistory 53
                        if cursor + 53 <= total_len:
                            binary_indices.extend(range(cursor, cursor + 53))
                            cursor += 53
                        if cursor + 53 <= total_len:
                            binary_indices.extend(range(cursor, cursor + 53))
                            cursor += 53
                        # Belief block (floats) 53*(N-1)
                        belief_len = 53 * (N - 1)
                        if cursor + belief_len <= total_len:
                            float_indices.extend(range(cursor, cursor + belief_len))
                            cursor += belief_len
                        # Turn one-hot N
                        if cursor + N <= total_len:
                            binary_indices.extend(range(cursor, cursor + N))
                            cursor += N
                    if cursor == total_len and binary_indices and float_indices:
                        bin_vals = fi_arr[binary_indices]
                        bin_bits = (bin_vals > 0.5).astype(_np.uint8)
                        packed = _np.packbits(bin_bits).tobytes()
                        float_vals = fi_arr[float_indices].astype(_np.float16)
                        state['full_compact'] = {
                            'packed_bits': packed,
                            'floats': float_vals,
                            'binary_len': int(bin_bits.shape[0]),
                            'num_players': int(N),
                            'format': 'cfv1',
                            'full_input_dim': int(total_len),
                            'layout_version': int(layout_version),
                        }
                        # store_full_input が False なら raw full_input を削除してメモリ節約
                        if not self.config.get('store_full_input', True):
                            try:
                                del state['full_input']
                            except Exception:
                                pass
                    else:
                        # v5 など未対応レイアウト用の汎用フォールバック圧縮
                        try:
                            bin_mask = (fi_arr <= 1e-6) | (fi_arr >= 1.0 - 1e-6)
                            binary_indices = _np.where(bin_mask)[0].tolist()
                            float_indices = _np.where(~bin_mask)[0].tolist()
                            if binary_indices or float_indices:
                                bin_vals = fi_arr[binary_indices] if binary_indices else fi_arr[0:0]
                                bin_bits = (bin_vals > 0.5).astype(_np.uint8) if binary_indices else _np.zeros(0, dtype=_np.uint8)
                                packed = _np.packbits(bin_bits).tobytes() if bin_bits.size > 0 else b''
                                float_vals = fi_arr[float_indices].astype(_np.float16) if float_indices else fi_arr[0:0].astype(_np.float16)
                                state['full_compact'] = {
                                    'packed_bits': packed,
                                    'floats': float_vals,
                                    'binary_len': int(bin_bits.shape[0]),
                                    'num_players': int(self.config.get('num_players', 4)),
                                    'format': 'cfv1',
                                    'full_input_dim': int(total_len),
                                    'layout_version': -1,  # generic
                                }
                                if not self.config.get('store_full_input', True):
                                    try:
                                        del state['full_input']
                                    except Exception:
                                        pass
                        except Exception:
                            pass
        except Exception:
            pass
        # --- 追加メモリ削減: pi 量子化(uint16), value/value_pred を uint8 ---
        import numpy as _np

        # pi 量子化
        pi_arr = _np.asarray(pi, dtype=_np.float32)
        s = float(pi_arr.sum())
        if s <= 0:
            if pi_arr.size > 0:
                pi_arr[:] = 1.0 / pi_arr.size
            s = 1.0
        scale = 65535.0 / s
        pi_q = _np.clip(_np.round(pi_arr * scale), 0, 65535).astype(_np.uint16)
        diff = int(65535 - int(pi_q.sum()))
        if diff != 0 and pi_q.size > 0:
            i = int(_np.argmax(pi_q))
            new_val = int(pi_q[i]) + diff
            if 0 <= new_val <= 65535:
                pi_q[i] = new_val  # type: ignore
        # value / value_pred 量子化
        def _q8(v):
            if v is None:
                return 255  # 特殊値 (未確定)
            try:
                return int(max(0, min(255, round(float(v) * 255))))
            except Exception:
                return 255
        value_u8 = _q8(value)
        value_pred_u8 = _q8(value_pred)
    # 固定ヘッド学習では ACTION_ID を使用しない（legal_ids/actions_format を保存しない）

        sample = {
            "player_id": self.player_id,
            "state": state,
            # 量子化/圧縮表現
            "pi_q": pi_q,
            "pi_format": "u16_norm65535",
            "value_u8": value_u8,
            "value_pred_u8": value_pred_u8,
            "value": value,  # assign_values 更新対象
            "model_version": getattr(self, 'model_version', 0),
            "feature_version": state.get('full_input_version', 1) if (isinstance(state, dict) and ('full_input' in state or 'full_compact' in state)) else 0,
        }
        # ------------------ 重複サンプルフィルタ ------------------
        if self._dup_enabled:
            try:
                sig_type = self.config.get('duplicate_signature_type', 'top_value_len')
                top_idx = int(_np.argmax(pi_q)) if pi_q.size > 0 else -1
                # legal 長は pi_q の長さを近似として用いる（固定ヘッドでは ACTION_ID を使用しない）
                legal_len = int(pi_q.size)
                vq = int(value_pred_u8) if value_pred_u8 is not None else 255
                if sig_type == 'top_value_len':
                    sig = (top_idx, vq, legal_len)
                else:
                    sig = (top_idx, vq, legal_len)
                max_cnt = int(self.config.get('duplicate_signature_max_count', 50) or 50)
                q = self._dup_sig_queue
                counts = self._dup_sig_counts
                if q is not None and counts is not None:
                    c = counts.get(sig, 0) + 1
                    counts[sig] = c
                    q.append(sig)
                    # ウィンドウから溢れた古いシグネチャをデクリメント (deque maxlen 発動時に一括処理できないため周期的に再計算)
                    # 簡易: ウィンドウ長が閾値に達したタイミングで O(n) 再カウント (コスト許容)
                    if len(q) == q.maxlen and (len(q) % 997 == 0):  # 疑似周期 (素数で偏り軽減)
                        new_counts = {}
                        for s_ in q:
                            new_counts[s_] = new_counts.get(s_, 0) + 1
                        counts.clear(); counts.update(new_counts)
                    if c > max_cnt:
                        # スキップ (格納しない)
                        self._dup_skipped += 1
                        sample['in_buffer'] = False
                        # ログ (interval)
                        log_int = int(self.config.get('duplicate_log_interval', 0) or 0)
                        if log_int > 0:
                            total_seen = self._dup_skipped + self._dup_kept
                            if total_seen - self._dup_last_log >= log_int:
                                if self.logger:
                                    self.logger.log_text(f"[dup] skipped={self._dup_skipped} kept={self._dup_kept} ratio={(self._dup_skipped/max(1,total_seen)):.3f}")
                                else:
                                    print(f"[dup] skipped={self._dup_skipped} kept={self._dup_kept} ratio={(self._dup_skipped/max(1,total_seen)):.3f}")
                                self._dup_last_log = total_seen
                        return sample
                    else:
                        self._dup_kept += 1
                        sample['dup_sig'] = sig
                        sample['in_buffer'] = True
            except Exception:
                pass
        # ----------------------------------------------------------
        # (Option) raw pi を保持: 分析/デバッグのため（drop_raw_pi=False のとき）。
        if not self.config.get('drop_raw_pi', True):
            try:
                import numpy as _np
                sample['pi'] = _np.asarray(pi, dtype=_np.float16)  # 半精度で縮小
            except Exception:
                pass
        # legal_actions のバックアップ (ID 復元で十分なら無効化してメモリ節約)
        if self.config.get('enable_legal_actions_backup', False):
            try:
                sample['legal_actions'] = legal_actions
            except Exception:
                pass
        # value_pred は value_pred_u8 に集約するため raw は保持しない
        # フル特徴量モード時に legacy サンプル(=0) を破棄して無駄容量を防ぐ
        if self.config.get('use_full_features') and sample['feature_version'] == 0:
            return sample  # 破棄 (格納しない)。戻り値だけ返す。
        # worker_zero_buffer モード: replay_buffer が None の場合はサンプルを返すのみ
        if self.replay_buffer is None:
            return sample  # Queue 送信は finalize_phase で行う
        if self._use_shared and hasattr(self.replay_buffer, 'append'):
            # 共有リプレイ (ReplayBuffer.append 内でエビクション処理)
            try:
                if bool(self.config.get('replay_async_enabled', False)) and hasattr(self.replay_buffer, 'append_async'):
                    drop_oldest = bool(self.config.get('replay_async_drop_oldest', True))
                    self.replay_buffer.append_async(sample, drop_oldest=drop_oldest)
                else:
                    self.replay_buffer.append(sample)
            except Exception:
                try:
                    self.replay_buffer.append(sample)
                except Exception:
                    pass
            return sample
        # ローカル deque: maxlen による自動 FIFO なのでそのまま append
        if isinstance(self.replay_buffer, deque):
            self.replay_buffer.append(sample)
        else:
            # 後方互換 (万一 list のまま残っているケース)
            if len(self.replay_buffer) >= self.max_buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(sample)
        return sample

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
            prev = rec.get("value")
            if prev is None:
                self.total_value_samples += 1
                if value > 0.5:
                    self.total_positive += 1
            rec["value"] = value
            # 量子化フィールドも更新
            try:
                rec["value_u8"] = int(255 if value is None else max(0, min(255, round(float(value) * 255))))
            except Exception:
                pass
            # 学習/検証スプリットを一度だけ付与
            try:
                if rec.get('split') is None:
                    ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                    import random as _r
                    rec['split'] = 'val' if (_r.random() < ratio) else 'train'
            except Exception:
                pass

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

    def finalize_phase(self, winner_player_id: int, was_active: bool):
        """フェーズ終端処理: 勝者IDに基づき 0/1 ラベル付与 + 予測精度集計."""
        if not was_active or not self._phase_samples:
            self._phase_samples = []
            return
        val = 1.0 if self.player_id == winner_player_id else 0.0
        lost = sum(1 for s in self._phase_samples if isinstance(s, dict) and s.get("in_buffer") is False)
        if lost:
            self.lost_phase_samples += lost
        try:
            last_rec = self._phase_samples[-1]
            pred = last_rec.get("value_pred") if isinstance(last_rec, dict) else None
            if pred is not None:
                self.phase_total += 1
                self.episode_phase_total += 1
                hit = (pred > 0.5) == (val > 0.5)
                if hit:
                    self.phase_correct += 1
                    self.episode_phase_correct += 1
        except Exception:
            pass
        self.assign_values(self._phase_samples, val)
        # worker_zero_buffer モード: 確定サンプルを一時リストに保存
        if self.replay_buffer is None:
            for s in self._phase_samples:
                if isinstance(s, dict) and s.get("value") is not None:
                    self._episode_confirmed_samples.append(s)
        self._phase_samples = []

    def flush_unfinished_phase(self):
        """未確定フェーズを 0 扱いで確定 (エピソード終了/中断時)。"""
        if self._phase_samples:
            lost = sum(1 for s in self._phase_samples if isinstance(s, dict) and s.get("in_buffer") is False)
            if lost:
                self.lost_phase_samples += lost
            try:
                last_rec = self._phase_samples[-1]
                pred = last_rec.get("value_pred") if isinstance(last_rec, dict) else None
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
                    if isinstance(s, dict) and s.get("value") is not None:
                        self._episode_confirmed_samples.append(s)
            self._phase_samples = []

    def finalize_game(self, *_args, **_kwargs):  # 互換維持用 no-op
        """ゲーム終端フック (最終順位報酬を使わないので何もしない)。"""
        # フェーズ一時サンプル破棄
        self._phase_samples = []
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
        """Ensure that the optimizer exists (lazy init with current hyperparams)."""
        if self._optimizer is None:
            try:
                import torch
                params = []
                if self.model is not None and hasattr(self.model, 'parameters'):
                    params = [p for p in self.model.parameters() if getattr(p, 'requires_grad', True)]
                lr = float(self.config.get("lr", getattr(self, 'lr', 1e-4)))
                wd = float(self.config.get("weight_decay", getattr(self, 'weight_decay', 1e-4)))
                self._optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
            except Exception:
                self._optimizer = None

    def save_optimizer(self, path: Optional[str] = None) -> bool:
        """Save optimizer state_dict to file. Returns True on success."""
        try:
            import os
            import torch
            if path is None:
                path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "optimizer_latest.pt")
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
            try:
                rb.clear()
            except Exception:
                pass

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
                        try:
                            bk = path + '.legacy_backup'
                            joblib.dump(data, bk, compress=3)
                        except Exception:
                            pass
                        data = [s for s in data if isinstance(s, dict) and s.get('feature_version',0)==1]
                        print(f"[INFO] purged legacy replay samples={legacy} kept={len(data)}")
            self.replay_buffer = data
        except FileNotFoundError:
            self.replay_buffer = []

    # ---------------- Training ----------------
    def train_step(self, batch_size: int = 64):
        """サンプルの一部で1ステップ学習し各種メトリクスを返す。value=None のものは除外。

        戻り値: dict(loss, policy_loss, value_loss, entropy, 追加統計...)
        代表的失敗理由: torch_not_installed / no_model / no_data / no_valid_samples
        """
        self._logged_inside = False
        try:
            import torch
        except ImportError:
            return {"loss": None, "reason": "torch_not_installed"}
        if self.model is None:
            return {"loss": None, "reason": "no_model"}
        # 共有バッファ: 自プレイヤーの確定サンプルのみ抽出 (学習splitのみ)
        if self._use_shared and hasattr(self.replay_buffer, 'iter_all'):
            my_samples = [s for s in self.replay_buffer.iter_all(owner_pid=self.player_id)
                          if s.get("value") is not None and (s.get('split') != 'val')]
            if not my_samples:
                return {"loss": None, "reason": "no_data"}
            batch_pool = my_samples
        else:  # ローカル
            if not self.replay_buffer:
                return {"loss": None, "reason": "no_data"}
            try:
                batch_pool = [s for s in self.replay_buffer if isinstance(s, dict) and (s.get('value') is not None) and (s.get('split') != 'val')]
            except Exception:
                batch_pool = self.replay_buffer

        # Optimizer 遅延初期化
        if self._optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # --- Prioritized sampling (lightweight, conservative) ---
        uid2weight = None
        prioritized_uids = None
        if (self._use_shared and hasattr(self.replay_buffer, 'sample_prioritized')
                and bool(self.config.get('prioritized_replay', True))):
            try:
                alpha = float(self.config.get('prioritized_replay_alpha', 0.6) or 0.6)
                eps = float(self.config.get('prioritized_replay_eps', 1e-6) or 1e-6)
                fast_full_input_np = None
                if hasattr(self.replay_buffer, 'sample_prioritized_fast'):
                    try:
                        fast_res = self.replay_buffer.sample_prioritized_fast(
                            batch_size, owner_pid=self.player_id, alpha=alpha, eps=eps,
                            return_weights=True, return_full_input=True)
                    except Exception:
                        fast_res = None
                else:
                    fast_res = None
                if isinstance(fast_res, dict) and fast_res.get('samples') is not None:
                    sampled = fast_res['samples']
                    sampled_uids = fast_res.get('uids', [])
                    is_weights = fast_res.get('is_weights', [])
                    fast_full_input_np = fast_res.get('full_input')
                else:
                    sampled, sampled_uids, is_weights = self.replay_buffer.sample_prioritized(
                        batch_size, owner_pid=self.player_id, alpha=alpha, eps=eps, return_weights=True)
                # map uid -> importance weight (normalized to max=1 by buffer)
                uid2weight = {int(u): float(w) for u, w in zip(sampled_uids, is_weights)}
                prioritized_uids = list(sampled_uids)
                batch = sampled
            except Exception:
                batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
                fast_full_input_np = None
        else:
            batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
            fast_full_input_np = None
        # フル特徴量モード時に旧フォーマット(feature_version=0)サンプルを除外
        if self.config.get('use_full_features'):
            filtered = [s for s in batch if s.get('feature_version', 0) >= 1]
            if not filtered:
                return {"loss": None, "reason": "no_full_feature_samples"}
            batch = filtered
        # --- クラス別ミックス (陽性率の最低保証) ---
        try:
            import math as _m
            target_pos_rate = float(self.config.get('value_pos_min_rate', 0.2) or 0.2)
        except Exception:
            target_pos_rate = 0.0
        if target_pos_rate > 0.0 and batch:
            def _is_pos(sample):
                v = sample.get('value', None)
                if v is None:
                    vu = sample.get('value_u8', None)
                    if isinstance(vu, int) and vu != 255:
                        v = vu / 255.0
                try:
                    return float(v) > 0.5
                except Exception:
                    return False
            cur_pos = sum(1 for s in batch if _is_pos(s))
            B = len(batch)
            max_frac = float(self.config.get('value_pos_topup_max_frac', 0.35) or 0.35)
            target_count = int(_m.ceil(min(max_frac, max(0.0, target_pos_rate)) * B))
            need = max(0, target_count - cur_pos)
            if need > 0:
                # 候補プール: 同一オーナー/学習splitの全サンプルから陽性のみ抽出
                try:
                    # batch_pool は先に作成済みの学習候補プール
                    candidates = [s for s in batch_pool if _is_pos(s)]
                except Exception:
                    candidates = []
                # 過学習対策: 直近使用UIDの除外 / 現バッチのUID除外 / 新し過ぎるUID除外
                in_batch_uids = set()
                for s in batch:
                    u = s.get('uid', None)
                    if u is not None:
                        try:
                            in_batch_uids.add(int(u))
                        except Exception:
                            pass
                recent_set = set(list(self._recent_pos_uids)) if isinstance(self._recent_pos_uids, (list, set)) else set(list(getattr(self, '_recent_pos_uids', [])))
                try:
                    max_uid_all = max(int(s.get('uid', -1)) for s in candidates if s.get('uid') is not None) if candidates else -1
                except Exception:
                    max_uid_all = -1
                uid_age_margin = int(self.config.get('value_pos_uid_age_margin', 1000) or 1000)
                filtered_cands = []
                for s in candidates:
                    uid = s.get('uid', None)
                    try:
                        uid_i = int(uid) if uid is not None else None
                    except Exception:
                        uid_i = None
                    if uid_i is not None:
                        if uid_i in in_batch_uids:
                            continue
                        if uid_i in recent_set:
                            continue
                        if (uid_age_margin > 0) and (max_uid_all >= 0) and (uid_i > (max_uid_all - uid_age_margin)):
                            # 新規すぎるサンプルは避ける（多様性確保）
                            continue
                    filtered_cands.append(s)
                # 最低候補数が小さい場合はスキップ（極端な再利用を避ける）
                min_cands = int(self.config.get('value_pos_topup_min_candidates', 200) or 200)
                if len(filtered_cands) >= max(min_cands, need):
                    # ミックス: 一部は一様ランダム、残りは優先度重み付き
                    import random as _r
                    uni_ratio = float(self.config.get('value_pos_topup_uniform_mix', 0.3) or 0.3)
                    take_uni = int(round(need * max(0.0, min(1.0, uni_ratio))))
                    take_pri = max(0, need - take_uni)
                    _r.shuffle(filtered_cands)
                    top_uni = filtered_cands[:take_uni]
                    remain_cands = filtered_cands[take_uni:]
                    # 重み: priority^alpha（無ければ1.0）
                    try:
                        alpha = float(self.config.get('prioritized_replay_alpha', 0.6) or 0.6)
                    except Exception:
                        alpha = 0.6
                    def _prio_w(s):
                        try:
                            p = float(s.get('priority', 1.0) or 1.0)
                        except Exception:
                            p = 1.0
                        try:
                            return max(1e-8, p) ** alpha
                        except Exception:
                            return 1.0
                    weights = [_prio_w(s) for s in remain_cands]
                    # 重み付きサンプル（置換なし）
                    top_pri = []
                    if take_pri > 0 and remain_cands:
                        # 簡易: サンプリング毎に正規化して選択
                        pool = list(remain_cands)
                        w = list(weights)
                        for _ in range(min(take_pri, len(pool))):
                            s_w = sum(w)
                            if s_w <= 0:
                                idx = _r.randrange(len(pool))
                            else:
                                r = _r.random() * s_w
                                acc = 0.0
                                idx = 0
                                for j, ww in enumerate(w):
                                    acc += ww
                                    if acc >= r:
                                        idx = j
                                        break
                            top_pri.append(pool.pop(idx))
                            _ = w.pop(idx)
                    topups = top_uni + top_pri
                    # 負例を置換して陽性率を引き上げる
                    if topups:
                        neg_indices = [i for i, s in enumerate(batch) if not _is_pos(s)]
                        if neg_indices:
                            # replace up to available slots
                            replace_n = min(len(topups), len(neg_indices))
                            # 置換でfast_full_inputは不整合になるので無効化
                            fast_full_input_np = None
                            for k in range(replace_n):
                                idx = neg_indices[k]
                                batch[idx] = topups[k]
                                # IS重みがある場合は新規UIDを1.0で登録（保守的）
                                try:
                                    if uid2weight is not None:
                                        uid_new = topups[k].get('uid')
                                        if uid_new is not None:
                                            uid2weight[int(uid_new)] = 1.0
                                except Exception:
                                    pass
                                # 最近使用UIDとして登録
                                try:
                                    uid_reg = topups[k].get('uid')
                                    if uid_reg is not None and hasattr(self, '_recent_pos_uids') and hasattr(self._recent_pos_uids, 'append'):
                                        self._recent_pos_uids.append(int(uid_reg))
                                except Exception:
                                    pass
        # --- ここまで: クラス別ミックス ---
        # 損失集計用のコンテナ
        policy_losses = []
        value_losses = []
        hand_losses = []
        entropies = []
        valid = 0
        collected_pi = []
        collected_model = []
        collected_v_pred = []
        collected_v_t = []
        collected_hand_pos_rate = []
        used_uids = []
        variable = getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate')

        vectorized_ok = False
        if not variable and hasattr(self.model, 'forward_batch'):
            try:
                import numpy as _np
                states = []
                pi_arrays = []  # list[np.ndarray]
                v_targets_list = []
                is_weights_list = []
                lengths = []
                hand_label_list = []  # list[np.ndarray or None]
                # 事前検証 / 復元フェーズ (最小限の Python ループ)
                for sample in batch:
                    # legal 長さを pi ベースで判断 (legal_actions 復元コスト削減)
                    # legal_actions が必要なのは variable モデルのみなので省略
                    # π 復元
                    if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                        pi_q = sample.get('pi_q')
                        try:
                            if hasattr(pi_q, 'astype'):
                                arr = pi_q.astype(_np.float32, copy=False)
                            else:
                                arr = _np.asarray(list(pi_q), dtype=_np.float32)
                            s_q = float(arr.sum())
                            if s_q > 0:
                                pi_arr = arr / s_q
                            else:
                                if arr.size == 0:
                                    continue
                                pi_arr = _np.ones_like(arr, dtype=_np.float32) / arr.size
                        except Exception:
                            raw = sample.get('pi')
                            if not raw:
                                continue
                            pi_arr = _np.asarray(list(raw), dtype=_np.float32)
                    else:
                        raw = sample.get('pi')
                        if not raw:
                            continue
                        pi_arr = _np.asarray(list(raw), dtype=_np.float32)
                    v_target = sample.get('value')
                    if v_target is None and 'value_u8' in sample:
                        vu = sample.get('value_u8')
                        if isinstance(vu, int) and vu != 255:
                            v_target = vu / 255.0
                    if v_target is None:
                        continue
                    # full feature zero padding skip check
                    if self.config.get('use_full_features') and self.config.get('skip_zero_padded_full_samples', True):
                        st = sample.get('state') or {}
                        if ('full_compact' not in st) and ('full_input' not in st):
                            continue
                    if pi_arr.size == 0:
                        continue
                    states.append(sample['state'])
                    pi_arrays.append(pi_arr)
                    v_targets_list.append(float(v_target))
                    # 手札ラベル（存在すれば収集）
                    try:
                        hl = None
                        st = sample.get('state') or {}
                        if isinstance(st, dict) and ('hand_labels' in st):
                            hl_raw = st.get('hand_labels')
                            if hl_raw is not None:
                                hl = _np.asarray(list(hl_raw), dtype=_np.float32)
                        hand_label_list.append(hl)
                    except Exception:
                        hand_label_list.append(None)
                    # importance-sampling weight for this sample (default 1.0)
                    try:
                        uid = sample.get('uid')
                        iw = float(uid2weight.get(int(uid), 1.0)) if uid2weight is not None else 1.0
                    except Exception:
                        iw = 1.0
                    is_weights_list.append(iw)
                    try:
                        if uid is not None:
                            used_uids.append(int(uid))
                    except Exception:
                        pass
                    lengths.append(int(pi_arr.shape[0]))
                if states:
                    # モデル一括 forward（belief ヘッド出力も取得）
                    import torch
                    try:
                        if fast_full_input_np is not None:
                            import torch as _t
                            xs = _t.from_numpy(fast_full_input_np).to(self.model.device).float()
                        else:
                            xs = torch.stack([self.model._encode_state(s) for s in states], dim=0)
                    except Exception:
                        xs = None
                    if xs is not None and hasattr(self.model, 'forward_with_belief'):
                        policy_logits_batch, value_logits_batch, hand_logits_batch = self.model.forward_with_belief(xs)
                    else:
                        policy_logits_batch, value_logits_batch = self.model.forward_batch(states)
                        hand_logits_batch = None
                    device = policy_logits_batch.device
                    max_len = max(lengths)
                    B = len(states)
                    # Pad π ターゲット
                    pi_pad = _np.zeros((B, max_len), dtype=_np.float32)
                    for i, arr in enumerate(pi_arrays):
                        pi_pad[i, :arr.shape[0]] = arr
                    pi_pad_t = torch.from_numpy(pi_pad).to(device)
                    lengths_t = torch.tensor(lengths, device=device)
                    # logits のパディング処理: 余剰部を -inf 相当でマスク
                    logits_slice = policy_logits_batch[:, :max_len]
                    # 長さ未満部分だけ使用するためマスクを構築
                    arange = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
                    mask = (arange < lengths_t.unsqueeze(1)).float()
                    LARGE_NEG = -1e9
                    masked_logits = logits_slice * mask + (1 - mask) * LARGE_NEG
                    log_probs = torch.log_softmax(masked_logits, dim=1)
                    probs = torch.exp(log_probs) * mask  # パディング部ほぼ0
                    # policy loss (各行で sum)
                    policy_loss_all = - (pi_pad_t * log_probs).sum(dim=1)
                    # value ロジット選択 (自プレイヤー視点)
                    pid = getattr(self, 'player_id', 0)
                    if value_logits_batch.ndim == 2 and pid < value_logits_batch.shape[1]:
                        v_logits = value_logits_batch[:, pid]
                    else:
                        v_logits = value_logits_batch[:, 0]
                    v_targets_t = torch.tensor(v_targets_list, dtype=torch.float32, device=device)
                    is_weights_t = torch.tensor(is_weights_list, dtype=torch.float32, device=device)
                    if 'bce_logits_loss_fn' not in self.__dict__:
                        import torch.nn as _nn
                        try:
                            pw = float(self.pos_weight)
                        except Exception:
                            pw = 1.0
                        pos_w_tensor = None
                        if pw != 1.0:
                            pos_w_tensor = torch.tensor([pw], dtype=torch.float32, device=device)
                        self.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()
                    # Compute per-sample BCE (no reduction) then apply IS weights conservatively to value loss only
                    try:
                        import torch.nn as _nn
                        # create a reduction='none' BCE with same pos_weight behavior
                        pos_w = None
                        try:
                            pos_w = float(self.pos_weight)
                        except Exception:
                            pos_w = 1.0
                        pos_w_tensor = None
                        if pos_w != 1.0:
                            pos_w_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
                        bce_none = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor, reduction='none') if pos_w_tensor is not None else _nn.BCEWithLogitsLoss(reduction='none')
                        value_loss_per = bce_none(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1)).squeeze(1)
                        # apply importance weights and average
                        value_loss_all = (value_loss_per * is_weights_t).mean()
                    except Exception:
                        # fallback to previous mean if anything goes wrong
                        value_loss_all = self.bce_logits_loss_fn(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1))
                    # Entropy
                    entropy_all = - (probs * log_probs).sum(dim=1)
                    # 集約
                    policy_losses.append(policy_loss_all.mean())
                    value_losses.append(value_loss_all)  # already mean
                    entropies.append(entropy_all.mean())
                    valid = len(states)
                    # hand 予測損失（存在時のみ）
                    try:
                        hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
                    except Exception:
                        hand_coef = 0.0
                    if hand_coef > 0.0 and (hand_logits_batch is not None):
                        # 対象サンプルのみ抽出
                        idx_labeled = [i for i, hl in enumerate(hand_label_list) if hl is not None]
                        if idx_labeled:
                            # hand_head 出力次元（後方互換で belief_dim も参照）
                            D = getattr(self.model, 'hand_pred_dim', None)
                            if D is None:
                                D = getattr(self.model, 'belief_dim', None)
                            # テンソル化
                            tgt_list = []
                            for i in idx_labeled:
                                hl = hand_label_list[i]
                                if hl is None:
                                    continue
                                if D is not None and hl.shape[0] != int(D):
                                    # 長さ不一致は切り詰め/パディング
                                    import numpy as _np
                                    if hl.shape[0] < int(D):
                                        pad = _np.zeros(int(D), dtype=_np.float32)
                                        pad[:hl.shape[0]] = hl
                                        hl = pad
                                    else:
                                        hl = hl[:int(D)]
                                tgt_list.append(torch.tensor(hl, dtype=torch.float32, device=device))
                            if tgt_list:
                                tgt = torch.stack(tgt_list, dim=0)
                                pred_logits = hand_logits_batch[idx_labeled]
                                import torch.nn as _nn
                                hloss = _nn.BCEWithLogitsLoss(reduction='mean')(pred_logits, tgt)
                                hand_losses.append(hloss)
                                # 参考: 正例率（監視用）
                                try:
                                    pos_rate_h = float((tgt.mean().detach().cpu().item()))
                                except Exception:
                                    pos_rate_h = None
                                collected_hand_pos_rate.append(pos_rate_h)
                    # メトリクス用個別保存
                    for i in range(len(states)):
                        n = lengths[i]
                        collected_pi.append(pi_pad_t[i, :n].detach())
                        collected_model.append(probs[i, :n].detach())
                        v_prob = torch.sigmoid(v_logits[i])
                        collected_v_pred.append(v_prob.detach())
                        collected_v_t.append(v_targets_t[i].detach())
                    vectorized_ok = True
            except Exception as _vec_e:
                # 一度だけ警告してフォールバック
                if not hasattr(self, '_vec_warned'):
                    print(f"[WARN] vectorized train_step fallback: {_vec_e}")
                    self._vec_warned = True  # type: ignore[attr-defined]
                vectorized_ok = False

        if not vectorized_ok:
            # 従来 per-sample ループ (variable モデル含む)
            for sample in batch:
                legal_actions = sample.get("legal_actions")
                # 可変長モデル使用時のみ、id_v1 からの復元を試みる（固定ヘッドでは不要）
                if variable and legal_actions is None and sample.get('actions_format') == 'id_v1' and 'legal_ids' in sample:
                    try:
                        # ACTION_ID_LIST を使わない方針のため、固定ヘッドでは復元スキップ
                        # 互換: 可変長モデル時のみ別経路での復元を想定（現状未使用）
                        legal_actions = None
                    except Exception:
                        legal_actions = None
                if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                    try:
                        import numpy as _np
                        pi_q = sample['pi_q']
                        pi_arr = pi_q.astype(_np.float32) if hasattr(pi_q, 'astype') else _np.asarray(list(pi_q), dtype=_np.float32)
                        s_q = float(pi_arr.sum())
                        if s_q <= 0:
                            pi_target = [1.0 / len(pi_arr)] * int(len(pi_arr)) if len(pi_arr) > 0 else []
                        else:
                            pi_target = (pi_arr / s_q).tolist()
                    except Exception:
                        pi_target = sample.get('pi')
                else:
                    pi_target = sample.get('pi')
                v_target = sample.get('value')
                if v_target is None and 'value_u8' in sample:
                    vu = sample.get('value_u8')
                    try:
                        if isinstance(vu, int) and vu != 255:
                            v_target = vu / 255.0
                    except Exception:
                        pass
                # 固定ヘッドでは legal_actions は不要。可変長のみ必須とする。
                if (variable and not legal_actions) or (not pi_target) or (v_target is None):
                    continue
                if self.config.get('use_full_features') and self.config.get('skip_zero_padded_full_samples', True):
                    try:
                        st = sample.get('state') or {}
                        if ('full_compact' not in st) and ('full_input' not in st):
                            continue
                    except Exception:
                        pass
                # 可変長: 合法手数、固定ヘッド: πの長さを使用
                n = (len(legal_actions) if variable else len(pi_target))
                if variable:
                    logits_raw, v_pred_raw = self.model.evaluate(sample['state'], legal_actions)
                else:
                    logits_raw, v_out_logits = self.model.forward(sample['state'])
                    if hasattr(v_out_logits, 'shape'):
                        pid = getattr(self, 'player_id', 0)
                        if 0 <= pid < v_out_logits.shape[0]:
                            v_pred_raw = v_out_logits[pid]
                        else:
                            v_pred_raw = v_out_logits[0]
                    else:
                        v_pred_raw = v_out_logits
                if hasattr(logits_raw, 'shape'):
                    logits_t = logits_raw
                    if logits_t.shape[0] < n:
                        pad = torch.zeros(n - logits_t.shape[0], device=logits_t.device)
                        logits_t = torch.cat([logits_t, pad], dim=0)
                    else:
                        logits_t = logits_t[:n]
                else:
                    logits_list = list(logits_raw)
                    if len(logits_list) < n:
                        logits_list += [0.0] * (n - len(logits_list))
                    logits_t = torch.tensor(logits_list[:n], dtype=torch.float32)
                log_probs = logits_t.log_softmax(dim=0)
                probs = log_probs.exp()
                pi_t = torch.tensor(pi_target, dtype=torch.float32, device=log_probs.device)
                if pi_t.shape[0] != log_probs.shape[0]:
                    m = min(pi_t.shape[0], log_probs.shape[0])
                    pi_t = pi_t[:m]
                    log_probs = log_probs[:m]
                    probs = probs[:m]
                policy_loss = -(pi_t * log_probs).sum()
                if isinstance(v_pred_raw, float):
                    v_logit = torch.tensor(v_pred_raw, dtype=torch.float32)
                else:
                    v_logit = v_pred_raw.float()
                v_t = torch.tensor(float(v_target), dtype=torch.float32, device=v_logit.device)
                if 'bce_logits_loss_fn' not in self.__dict__:
                    import torch.nn as _nn
                    try:
                        pw = float(self.pos_weight)
                    except Exception:
                        pw = 1.0
                    pos_w_tensor = None
                    if pw != 1.0:
                        import torch as _t
                        pos_w_tensor = _t.tensor([pw], dtype=_t.float32, device=v_logit.device)
                    self.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()
                # apply importance-sampling weight (if prioritized sampling was used)
                try:
                    uid = sample.get('uid')
                    iw = float(uid2weight.get(int(uid), 1.0)) if uid2weight is not None else 1.0
                except Exception:
                    iw = 1.0
                value_loss = self.bce_logits_loss_fn(v_logit.unsqueeze(0), v_t.unsqueeze(0)) * float(iw)
                v_prob = torch.sigmoid(v_logit)
                entropy = -(probs * log_probs).sum()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                # hand ヘッド（単体; 可能なら）簡素化: 例外は一括捕捉
                hand_coef = 0.0
                try:
                    hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
                except Exception:
                    hand_coef = 0.0
                if hand_coef > 0.0 and hasattr(self.model, 'forward_with_belief'):
                    hand_logits = None
                    try:
                        _, _, hand_logits = self.model.forward_with_belief(sample['state'])
                    except Exception:
                        hand_logits = None
                    if hand_logits is not None:
                        hl = None
                        st = sample.get('state') or {}
                        if isinstance(st, dict):
                            hl = st.get('hand_labels')
                        if hl is not None:
                            try:
                                import torch.nn as _nn, numpy as _np, torch
                                D = getattr(self.model, 'belief_dim', None)
                                hl_arr = _np.asarray(list(hl), dtype=_np.float32)
                                if D is not None and hl_arr.shape[0] != int(D):
                                    if hl_arr.shape[0] < int(D):
                                        pad = _np.zeros(int(D), dtype=_np.float32)
                                        pad[:hl_arr.shape[0]] = hl_arr
                                        hl_arr = pad
                                    else:
                                        hl_arr = hl_arr[:int(D)]
                                tgt = torch.tensor(hl_arr, dtype=torch.float32, device=hand_logits.device)
                                hloss = _nn.BCEWithLogitsLoss(reduction='mean')(hand_logits.float(), tgt)
                                hand_losses.append(hloss)
                                try:
                                    collected_hand_pos_rate.append(float(tgt.mean().detach().cpu().item()))
                                except Exception:
                                    pass
                            except Exception:
                                pass
                
                valid += 1
                collected_pi.append(pi_t.detach())
                collected_model.append(probs.detach())
                collected_v_pred.append(v_prob.detach())
                collected_v_t.append(v_t.detach())
                try:
                    uid = sample.get('uid')
                    if uid is not None:
                        used_uids.append(int(uid))
                except Exception:
                    pass

        if valid == 0:
            return {"loss": None, "reason": "no_valid_samples"}

        policy_loss_mean = torch.stack(policy_losses).mean()
        value_loss_mean = torch.stack(value_losses).mean()
        hand_loss_mean = (torch.stack(hand_losses).mean() if hand_losses else None)
        entropy_mean = torch.stack(entropies).mean()
        total_loss = (self.policy_loss_coef * policy_loss_mean +
                      self.value_loss_coef * value_loss_mean -
                      self.entropy_coef * entropy_mean)
        try:
            hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
        except Exception:
            hand_coef = 0.0
        if hand_coef > 0.0 and hand_loss_mean is not None:
            total_loss = total_loss + hand_coef * hand_loss_mean
        # --- Mixed precision training (GradScaler) ---
        try:
            import torch as _t
        except Exception:
            _t = None

        # decide whether to use AMP: require CUDA device on the model
        _use_amp = False
        try:
            _dev = getattr(self.model, 'device', None)
            _use_amp = bool(getattr(_dev, 'type', None) == 'cuda') and (_t is not None and _t.cuda.is_available())
        except Exception:
            _use_amp = False

        # delayed GradScaler init (store on agent)
        if _use_amp:
            try:
                if not hasattr(self, '_amp_scaler') or self._amp_scaler is None:
                    self._amp_scaler = _t.cuda.amp.GradScaler()
            except Exception:
                # fallback to no AMP
                try:
                    self._amp_scaler = None
                except Exception:
                    pass

        self._optimizer.zero_grad()
        did_update = False
        if _use_amp and getattr(self, '_amp_scaler', None) is not None:
            try:
                # scale the loss, backward, then unscale for clipping
                self._amp_scaler.scale(total_loss).backward()
                try:
                    # unscale before clip
                    self._amp_scaler.unscale_(self._optimizer)
                except Exception:
                    pass
                if self.grad_clip and self.grad_clip > 0:
                    try:
                        _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    except Exception:
                        pass
                try:
                    self._amp_scaler.step(self._optimizer)
                    self._amp_scaler.update()
                    did_update = True
                except Exception:
                    # If step failed (e.g. inf/overflow), update scaler and skip this step
                    try:
                        self._amp_scaler.update()
                    except Exception:
                        pass
                    did_update = False
            except Exception:
                # fallback to FP32 path on unexpected errors
                try:
                    total_loss.backward()
                    if self.grad_clip and self.grad_clip > 0:
                        try:
                            _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        except Exception:
                            pass
                    self._optimizer.step()
                    did_update = True
                except Exception:
                    did_update = False
        else:
            # FP32 update path
            try:
                total_loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    try:
                        _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    except Exception:
                        pass
                self._optimizer.step()
                did_update = True
            except Exception:
                did_update = False
        # Scheduler step (warmup+cosine) — optimizer 実ステップがあった時のみ実行
        try:
            if did_update:
                if self._scheduler is None:
                    self.ensure_scheduler()
                if self._scheduler is not None:
                    # 一部の実装では optimizer._step_count に基づき順序チェックが行われるため念のため確認
                    step_cnt = getattr(self._optimizer, "_step_count", None)
                    if (step_cnt is None) or (int(step_cnt) > 0):
                        # self._update_step を 1 進め、scheduler も 1 ステップ進める
                        self._update_step += 1
                        self._scheduler.step()
        except Exception:
            pass

        # 追加メトリクス計算（共通ユーティリティに委譲, pos_rate はローカルで算出）
        policy_kl = None
        policy_top1 = None
        value_acc = None
        value_brier = None
        pos_rate = None
        if collected_pi:
            try:
                from utils.metrics import compute_policy_value_metrics
                extra = compute_policy_value_metrics(collected_pi, collected_model, collected_v_pred, collected_v_t)
                policy_kl = extra.get('policy_kl')
                policy_top1 = extra.get('policy_top1_match')
                value_acc = extra.get('value_acc')
                value_brier = extra.get('value_brier')
                # pos_rate は学習時のみ必要なためここで算出
                v_label_list = [float(v_lab.item()) for v_lab in collected_v_t]
                if v_label_list:
                    pos_rate = float(sum(1.0 if v > 0.5 else 0.0 for v in v_label_list) / len(v_label_list))
            except Exception as e:
                if not hasattr(self, '_metric_warned'):
                    print(f"[WARN] metric calc failed: {e}")
                    self._metric_warned = True

        metrics = {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss_mean.item()),
            "value_loss": float(value_loss_mean.item()),
            "entropy": float(entropy_mean.item()),
            "samples": valid,
            "policy_kl": policy_kl,
            "policy_top1_match": policy_top1,
            "value_acc": value_acc,
            "value_brier": value_brier,
            "pos_rate": pos_rate,
            "cum_pos_rate": (self.total_positive / self.total_value_samples) if self.total_value_samples > 0 else None,
            "tt_hit_rate": (self.tt_hits / max(1, (self.tt_hits + self.tt_misses))) if (self.tt_hits + self.tt_misses) > 0 else None,
        }
        # 追加: hand ヘッド関連メトリクス
        try:
            if hand_coef > 0.0:
                metrics["hand_pred_loss"] = (float(hand_loss_mean.item()) if hand_loss_mean is not None else None)
                # ラベルの正例率（監視用）
                if collected_hand_pos_rate:
                    # None を除外して平均
                    vals = [v for v in collected_hand_pos_rate if v is not None]
                    metrics["hand_label_pos_rate"] = (sum(vals) / len(vals)) if vals else None
                else:
                    metrics["hand_label_pos_rate"] = None
        except Exception:
            pass
        # pos_rate 警告
        try:
            warn_th = float(self.config.get('pos_rate_warn_threshold', 0.02))
            if pos_rate is not None and pos_rate < warn_th:
                print(f"[WARN] value positive sample rate low ({pos_rate:.2%}) < {warn_th:.2%}")
        except Exception:
            pass

# --- Update priorities for prioritized replay (conservative: abs(pred-target) + eps) ---
        try:
            if bool(self.config.get('prioritized_replay', False)) and hasattr(self.replay_buffer, 'update_priorities') and used_uids:
                uid_to_p = {}
                eps = float(self.config.get('prioritized_replay_eps', 1e-6) or 1e-6)
                for i, uid in enumerate(used_uids):
                    try:
                        pred = collected_v_pred[i]
                        targ = collected_v_t[i]
                        # convert tensors to float safely
                        try:
                            pred_f = float(pred.detach().cpu().item()) if hasattr(pred, 'detach') else float(pred)
                        except Exception:
                            try:
                                pred_f = float(pred.item())
                            except Exception:
                                pred_f = float(pred)
                        try:
                            targ_f = float(targ.detach().cpu().item()) if hasattr(targ, 'detach') else float(targ)
                        except Exception:
                            try:
                                targ_f = float(targ.item())
                            except Exception:
                                targ_f = float(targ)
                        p = abs(pred_f - targ_f) + eps
                        uid_to_p[int(uid)] = float(p)
                    except Exception:
                        continue
                if uid_to_p:
                    try:
                        self.replay_buffer.update_priorities(uid_to_p)
                    except Exception:
                        pass
        except Exception:
            pass
        if self.logger:
            self.logger.log_train(metrics)
            self._logged_inside = True
            # メモリスナップショット (学習プレイヤーのみ想定: config.learning_player_id)
            try:
                lp = int(self.config.get('learning_player_id', 0))
                if self.player_id == lp and hasattr(self.logger, 'log_memory_snapshot'):
                    # 共有リプレイ形式かローカルかでサンプル数を取得
                    sample_count = None
                    try:
                        if self.replay_buffer is None:
                            sample_count = 0  # worker_zero_buffer モードではサンプル数は0
                        elif self._use_shared and hasattr(self.replay_buffer, '__len__'):
                            sample_count = len(self.replay_buffer)
                        elif isinstance(self.replay_buffer, list):
                            sample_count = len(self.replay_buffer)
                    except Exception:
                        pass
                    self.logger.log_memory_snapshot(sample_count=sample_count, force=False)
            except Exception:
                pass
            # events.log へ周期的に TRAIN 指標を一行テキストとして追記 (軽量モニタ用)
            try:
                freq = int(self.config.get('events_log_train_every', 0) or 0)
                if freq > 0:
                    step = getattr(self.logger, 'update_step', None)
                    # log_train 呼び出し後なので update_step は 1 インクリメント済み
                    if step and (step % freq == 0):
                        parts = [
                            f"loss={metrics.get('loss'):.4f}" if metrics.get('loss') is not None else None,
                            f"pl={metrics.get('policy_loss'):.4f}" if metrics.get('policy_loss') is not None else None,
                            f"vl={metrics.get('value_loss'):.4f}" if metrics.get('value_loss') is not None else None,
                            f"hand={metrics.get('hand_pred_loss'):.4f}" if metrics.get('hand_pred_loss') is not None else None,
                            f"ent={metrics.get('entropy'):.3f}" if metrics.get('entropy') is not None else None,
                            f"kl={metrics.get('policy_kl'):.4f}" if metrics.get('policy_kl') is not None else None,
                            f"top1={metrics.get('policy_top1_match'):.3f}" if metrics.get('policy_top1_match') is not None else None,
                            f"v_acc={metrics.get('value_acc'):.3f}" if metrics.get('value_acc') is not None else None,
                            f"v_brier={metrics.get('value_brier'):.4f}" if metrics.get('value_brier') is not None else None,
                            f"pos={metrics.get('pos_rate'):.3f}" if metrics.get('pos_rate') is not None else None,
                            f"hand_pos={metrics.get('hand_label_pos_rate'):.3f}" if metrics.get('hand_label_pos_rate') is not None else None,
                            f"cum_pos={metrics.get('cum_pos_rate'):.3f}" if metrics.get('cum_pos_rate') is not None else None,
                            f"samples={metrics.get('samples')}" if metrics.get('samples') is not None else None,
                        ]
                        line = " ".join(p for p in parts if p is not None)
                        self.logger.log_text(f"[TRAIN] step={step} {line}", also_print=False)
            except Exception:
                pass
        return metrics

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
                try:
                    self.logger.log_text(line, also_print=False)
                except Exception:
                    pass
            else:
                try:
                    import os, datetime
                    log_dir = self.config.get('log_dir', 'logs')
                    os.makedirs(log_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                        f.write(f"[{ts}] {line}\n")
                except Exception:
                    pass
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
        try:
            check = self.train_step(batch_size=batch_size)
            if isinstance(check, dict) and check.get('reason') is not None:
                return {"count": 0, "total_s": 0.0, "mean_s": None, "std_s": None, "min_s": None, "max_s": None, "reason": check.get('reason')}
        except Exception:
            # 例外は無視して計測を試みる
            pass

        # warmup
        for _ in range(max(0, int(warmup or 0))):
            try:
                self.train_step(batch_size=batch_size)
            except Exception:
                pass

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
            return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": f"validate_error: {type(e).__name__}: {e}"}

    def reset_episode(self):
        """エピソード開始時にカウンタ類を初期化."""
        self.move_count = 0
        # 進行に応じた温度スケジュール短縮のため、エピソード数をカウント
        try:
            self.episodes_played += 1
        except Exception:
            self.episodes_played = int(getattr(self, "episodes_played", 0)) + 1
        self.episode_phase_total = 0
        self.episode_phase_correct = 0
        # Belief 増分キャッシュ/直近状態をリセット
        try:
            self._belief_cache = None
            self._last_state = None
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
            except Exception:
                pass
        # Phase サンプルの積極的クリア（メモリ削減オプション）
        if self.config.get('aggressive_phase_clear', False):
            try:
                if hasattr(self, '_phase_samples'):
                    self._phase_samples.clear()
            except Exception:
                pass
        # determinization プールはゲームを跨ぐと署名ミスマッチが増えるためデフォルトで再初期化
        try:
            if self.config.get('reset_det_pool_each_game', True):
                self.shutdown_det_pool()
        except Exception:
            pass
        # per-episode perf counters
        try:
            self._perf_infer_ms_accum = 0.0
            self._perf_infer_calls = 0
        except Exception:
            pass

    # ------------ Determinization pool teardown ------------
    def shutdown_det_pool(self):
        """テスト/終了時に determinization ワーカーを安全に停止する補助メソッド."""
        try:
            if getattr(self, '_det_stop_event', None) is not None:
                self._det_stop_event.set()
            th = getattr(self, '_det_pool_thread', None)
            if th and th.is_alive():
                th.join(timeout=1.0)
        except Exception:
            pass
        # リソース参照を解放
        try:
            self._det_pool = None
            self._det_pool_thread = None
            self._det_pool_lock = None
            self._det_stop_event = None
        except Exception:
            pass


DRLAgent = AlphaZeroAgent


class _AlphaZeroTTView(dict):
    """エージェント内簡易 LRU TT の薄いビュー。

    内部表現: agent._mcts_tt: {key: (result, tick)}
    ここでは dict 互換の最小操作のみを提供し、参照される度に tick を更新。
    容量超過時には最古 tick を削除。
    """
    def __init__(self, agent: AlphaZeroAgent):
        self._agent = agent
    def _mk(self, k):
        return (getattr(self._agent, 'model_version', 0), k)

    def __contains__(self, k):
        try:
            store = self._agent._mcts_tt
            found = self._mk(k) in store
            try:
                if found:
                    self._agent.tt_hits += 1
                else:
                    self._agent.tt_misses += 1
            except Exception:
                pass
            return found
        except RecursionError:
            # 予期せぬ再帰増殖を検知したらTTを無効化して以降は未使用にする
            try:
                self._agent.enable_mcts_tt = False
                self._agent._mcts_tt = {}
            except Exception:
                pass
            return False

    def __getitem__(self, k):
        try:
            store = self._agent._mcts_tt
            key = self._mk(k)
            res, _tick = store[key]
            self._agent._mcts_tt_tick += 1
            store[key] = (res, self._agent._mcts_tt_tick)
            return res
        except RecursionError:
            try:
                self._agent.enable_mcts_tt = False
                self._agent._mcts_tt = {}
            except Exception:
                pass
            raise KeyError(k)

    def __setitem__(self, k, v):
        try:
            agent = self._agent
            store = agent._mcts_tt
            agent._mcts_tt_tick += 1
            store[self._mk(k)] = (v, agent._mcts_tt_tick)
            # 簡易エビクション
            cap = max(1, int(agent.mcts_tt_capacity or 1))
            if len(store) > cap:
                # 最小 tick を 1 件落とす
                oldest_k = min(store.items(), key=lambda kv: kv[1][1])[0]
                store.pop(oldest_k, None)
        except RecursionError:
            try:
                agent = self._agent
                agent.enable_mcts_tt = False
                agent._mcts_tt = {}
            except Exception:
                pass
            # TT を無効化して以降の呼び出しは黙って破棄
            return
