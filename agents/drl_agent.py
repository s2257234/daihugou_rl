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
        # バッチ推論/TT 設定（デフォルトは挙動不変: バッチ=1, キャッシュoff）
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

        # 学習用ハイパーパラメータ
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

        # 統計: 累積フェーズラベル分布
        self.total_value_samples = 0
        self.total_positive = 0
        # フェーズ予測精度集計
        self.phase_total = 0
        self.phase_correct = 0
        self.episode_phase_total = 0
        self.episode_phase_correct = 0
        # リプレイ追い出し検知
        self.lost_phase_samples = 0
        # TT 統計
        self.tt_hits = 0
        self.tt_misses = 0
        # pos weight (クラス不均衡対策) optional
        self.pos_weight = float(self.config.get("value_pos_weight", 1.5))

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
        
        # 学習ターゲット用: policy_target_tau で固定（訪問数の素直な正規化）
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

        # 直近状態の抽出（差分用ヒント）。評価時でも次回の差分計算に必要なため常に作成。
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
                if self.model is None:
                    outs = []
                    for i, e in enumerate(env_list):
                        legal = (legal_list[i] if legal_list is not None else self._get_legal_actions(e))
                        if not legal:
                            outs.append(({}, 0.0))
                        else:
                            p = 1.0 / len(legal)
                            outs.append(({a: p for a in legal}, 0.0))
                    return outs
                # 可変長アクションモデルはバッチ困難 → 個別に処理
                if getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate'):
                    return [self._policy_value(e) for e in env_list]
                # forward_batch があれば使う
                if hasattr(self.model, 'forward_batch'):
                    states = [self._extract_state(e) for e in env_list]
                    # 推論は勾配不要 + CUDA では AMP を利用
                    try:
                        import torch as _t
                        _dev = getattr(self.model, 'device', None)
                        _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
                        with _t.no_grad():
                            with _t.amp.autocast('cuda', enabled=_use_amp):
                                import time as _tperf
                                _t0 = _tperf.time()
                                logits_b, value_vec_b = self.model.forward_batch(states)
                        try:
                            self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
                        except Exception:
                            pass
                    except Exception:
                        import time as _tperf
                        _t0 = _tperf.time()
                        logits_b, value_vec_b = self.model.forward_batch(states)
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
        state = self._extract_state(env)
        legal = self._get_legal_actions(env)
        n = len(legal)
        if n == 0:
            return {}, 0.0
        if self.model is None:
            p = 1.0 / n
            return {a: p for a in legal}, 0.0

        # 可変長アクション対応モデル
        try:
            import torch as _t
            _dev = getattr(self.model, 'device', None)
            _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
            with _t.no_grad():
                if getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate'):
                    # 可変長アクションモデルの逐次評価
                    # 期待: value は各プレイヤーのロジット/確率ベクター
                    # 後方互換: スカラーしか返らない場合は N 人分に複製
                    import time as _tperf
                    _t0 = _tperf.time()
                    logits, value_out = self.model.evaluate(state, legal)
                    try:
                        self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
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
                        logits_t, value_vec_t = self.model.forward(state)  # tensors
                    try:
                        self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
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
            if getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate'):
                import time as _tperf
                _t0 = _tperf.time()
                logits, value_out = self.model.evaluate(state, legal)
                try:
                    self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
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
                logits_t, value_vec_t = self.model.forward(state)
                try:
                    self._perf_infer_ms_accum += ( (_tperf.time() - _t0) * 1000.0 )
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
                    for s in fb:
                        core = s[:-1]
                        try:
                            ranks.append(int(core))
                        except Exception:
                            pass
                    if not ranks:
                        continue
                    base_rank = min(ranks)
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
            for attempt in range(max_retry):
                unknown = list(unknown_seed)
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
          - 「場が空」の場合はパスを合法手から除外する。
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
        """部分観測用特徴量 (full_input v4)。

                レイアウト (rank クラス 4 種: daifugo,fugo,hinmin,daihinmin):
                    Self(55):  自分手札53bit + pass1 + remain_norm1
                    OppSummary( (1 + 4) * (N-1) = 5(N-1) ): 各 opponent の remain_norm1 + rank one-hot(4)
                    Field(22): revolution1 + combo7 + base_rank13 + field_size_norm1
                    FieldCards(53): 現在場に出ている具体カードビット
                    PlayHistory(53): これまで公開された(場 or 自分が保持して見えた)カードフラグ
                    Belief(53*(N-1)): opponent ごとのカード存在確率 (所在不明カードのみ一定値)
                    Turn(N): 手番 one-hot

                        合計次元:
                                                        = Self 55
                                                            + OppSummary 5(N-1)
                                                            + Field 22
                                                            + FieldCards 53
                                                            + PlayHistory 53
                                                            + Belief 53(N-1)
                                                            + Turn N
                                                        = (55 + 22 + 53 + 53) + 5(N-1) + 53(N-1) + N
                                                        = 183 + 58(N-1) + N
                                                        = 183 + 58N - 58 + N
                                                        = 59N + 125
                                                したがって expected_full_dim = 59 * num_players + 125
                                (v3 との差異: FieldCards + PlayHistory の 106 次元追加)
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
            expected_full_dim = 59 * num_players + 125

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
            # PlayHistory (53)
            history_bits = [0.0]*53
            try:
                hist_cards = list(getattr(g, 'play_history', []))
            except Exception:
                hist_cards = []
            merged = set()
            for seq in (getattr(me,'hand',[]), hist_cards, field):
                for c in seq:
                    try:
                        idx = card_index(c)
                        if 0 <= idx < 53:
                            merged.add(idx)
                    except Exception:
                        pass
            for idx in merged:
                history_bits[idx] = 1.0
            feat.extend(history_bits)

            # --- Belief 計算用の履歴インデックス（除去済み・公開済み）---
            # 重要: Belief の「所在不明」からは、過去に場へ出て流れたカード（play_history）も除外する。
            # これにより「既に除去済みカードを相手が持っている確率が >0」という論理矛盾を防ぐ。
            history_idx = set()
            for c in hist_cards:
                try:
                    idx = card_index(c)
                    if 0 <= idx < 53:
                        history_idx.add(idx)
                except Exception:
                    pass

            # Belief distributions
            # 目的: "所在不明" の各カードが 各 opponent の手札にある確率 P(card=k ∈ hand_i) を推定し 53*(N-1) 次元に展開。
            # 改良: パス時の論理的消去法を導入し、各相手 i の PossibilitySet_i を構成。
            #      パスが発生した手番の場の状態(役タイプ/基準ランク/枚数/革命)に基づき、
            #      その時点で出せたはずの全ての“候補カード”を PossibilitySet_i から除外する。
            #      シンプルに、single/pair/triple/four については基準を満たすランク(とJOKER)の全スートカードを除外。
            #      straight については安全のため現段階では除外ロジックを適用しない（過剰除外を避ける）。
            #      最終的に x_i[k] は k∈PossibilitySet_i のとき H_i / |PossibilitySet_i|、それ以外は 0 とする（上限1.0にクリップ）。
            #      |PossibilitySet_i|==0 の場合はフォールバックとして 0 をセットする。
            # ステップ1: 所在不明カード集合 U を構成 (自分の手札 / 現在フィールド / PlayHistory を除外)
            self_idx = set()
            for c in getattr(me,'hand',[]):
                try:
                    idx = card_index(c)
                    if 0 <= idx < 53:
                        self_idx.add(idx)
                except Exception:
                    pass
            field_idx = set()
            for c in field:
                try:
                    idx = card_index(c)
                    if 0 <= idx < 53:
                        field_idx.add(idx)
                except Exception:
                    pass
            # PlayHistory 由来の公開・除去カードも unknown から除外する
            unknown = [i for i in range(53) if i not in self_idx and i not in field_idx and i not in history_idx]
            unknown_set = set(unknown)

            # ステップ2: 各 opponent の手札残数 H_i を取得
            opp_ids = opponents
            H = {i: opp_remains[i] for i in opp_ids}

            # 逆引きヘルパ: index -> (rank, suit, is_joker)
            def index_to_rsi(idx: int):
                if idx == 52:
                    return (None, None, True)
                try:
                    suit_idx = idx // 13
                    rank = (idx % 13) + 1
                    suits = ['S','H','D','C']
                    suit = suits[suit_idx] if 0 <= suit_idx < 4 else 'S'
                    return (rank, suit, False)
                except Exception:
                    return (None, None, False)

            def idxs_of_rank(r: int):
                if r is None:
                    return []
                base = (r - 1)
                return [base + 13 * s for s in range(4)]  # S,H,D,C の順

            # ステップ3: パス履歴に基づく PossibilitySet_i の構築
            # キャッシュがあれば差分適用のみ行う（毎回の全履歴再走査を避ける）
            try:
                history = list(getattr(g, '_action_history', []) or [])
            except Exception:
                history = []

            use_cache = False
            poss_sets = None
            start_idx = 0
            cached = getattr(self, '_belief_cache', None)
            if isinstance(cached, dict) and cached.get('num_players') == num_players:
                c_poss = cached.get('poss_sets')
                c_hist = cached.get('hist_len')
                if isinstance(c_poss, dict) and isinstance(c_hist, int) and 0 <= c_hist <= len(history):
                    # 既存キャッシュをベースに差分だけ適用
                    poss_sets = c_poss  # 参照のまま更新（差分適用）
                    start_idx = c_hist
                    # 対象プレイヤー集合が一致しない場合は再初期化
                    if set(poss_sets.keys()) != set(opp_ids):
                        poss_sets = None
                    else:
                        use_cache = True
            if not use_cache or poss_sets is None:
                # キャッシュなし/不整合時は初期化
                poss_sets = {i: set(unknown_set) for i in opp_ids}
                start_idx = 0

            # 場の基準情報抽出用ユーティリティ
            def _base_rank_from_field_strs(field_strs: List[str]) -> Optional[int]:
                # 例: ['♦10'] 等 → ランク最小/最大は revolution で解釈が変わるが、pair/triple は同ランク
                ranks = []
                for s in field_strs:
                    try:
                        core = s[1:]  # 先頭はスート
                        map_face = {'A':1,'J':11,'Q':12,'K':13}
                        ranks.append(map_face.get(core.upper(), int(core)))
                    except Exception:
                        pass
                if not ranks:
                    return None
                return min(ranks)  # 基準として最小ランクを用いる（compare は revolution で吸収）

            for h in history[start_idx:]:
                try:
                    if h.get('action') != 'pass':
                        continue
                    pid_pass = h.get('pid')
                    if pid_pass is None or pid_pass == pid:
                        # 自分のパスは Belief には不要
                        continue
                    field_before = h.get('field_before') or []
                    if not field_before:
                        # 場が空でのパス（実質起きない）は制約なし
                        continue
                    combo_type = h.get('combo_type')
                    revo_flag = bool(h.get('revo', False))
                    base_rank = _base_rank_from_field_strs(field_before)
                    # 除外候補 index 集合
                    exclude_idxs: set[int] = set()
                    if combo_type in (None, 'single', 'joker_single'):
                        # single 系: 基準を上回れる単カード全て（革命時は逆順）+ Joker
                        if combo_type == 'joker_single':
                            # Joker 同士のみ可能 → Joker を排除
                            exclude_idxs.add(52)
                        else:
                            # ランク 1..13 のうち条件を満たすもの
                            for r in range(1, 14):
                                ok = (r > base_rank) if not revo_flag else (r < base_rank)
                                if ok:
                                    exclude_idxs.update(idxs_of_rank(r))
                            # Joker は常に上位として扱う
                            exclude_idxs.add(52)
                    elif combo_type in ('pair','triple','four'):
                        # 同ランク複数枚: 基準を上回れるランク r の単カード全て（組めるなら出せたはず → 強めの除外）
                        size = len(field_before)
                        # ランク判定
                        for r in range(1, 14):
                            ok_rank = (r > base_rank) if not revo_flag else (r < base_rank)
                            if not ok_rank:
                                continue
                            # r の全スートカード（組合せ要件はここでは近似として無視し、積極的に除外）
                            exclude_idxs.update(idxs_of_rank(r))
                        # Joker も代用として使える可能性が高いので除外
                        exclude_idxs.add(52)
                    elif combo_type == 'straight':
                        # 階段は構成の自由度が高く、過剰除外のリスクが大きいので現段階では適用しない
                        exclude_idxs = set()
                    # 適用: 未知カードのみ対象
                    if pid_pass in poss_sets:
                        poss_sets[pid_pass] -= (exclude_idxs & unknown_set)
                except Exception:
                    continue

            # キャッシュ更新
            try:
                self._belief_cache = {
                    'num_players': num_players,
                    'poss_sets': poss_sets,
                    'hist_len': len(history),
                }
            except Exception:
                pass

            # ステップ4: 53 長ベクトル生成 (非所在 or 自手札/場カードは 0)。
            #           x_i[k] = H_i / |PossibilitySet_i| (k ∈ PossibilitySet_i), それ以外は 0。上限は 1.0 にクリップ。
            for i in opponents:
                probs = [0.0] * 53
                S_i = poss_sets.get(i, set()) & unknown_set
                size_si = len(S_i)
                h_i = int(H.get(i, 0) or 0)
                if size_si > 0 and h_i > 0:
                    p_each = min(1.0, float(h_i) / float(size_si))
                    for idx in S_i:
                        probs[idx] = p_each
                # size==0 または h_i==0 の場合は 0 埋め
                feat.extend(probs)
            # 以前は unknown_count と opponent remains の不一致を一度だけ通知していたが
            # 運用で冗長になったためログ出力を廃止 (計算ロジックはそのまま)。

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
                    print(f"[WARN] adjusted full_input length from {cur_len} to expected {expected_full_dim}")
                    self._warned_full_dim_autofix = True
            base['full_input'] = feat
            base['full_input_dim'] = expected_full_dim
            base['full_input_version'] = 4
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
                self.replay_buffer.append(sample)
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
            # store_full_input=False の場合は featureベクトル自体を保持しない
            if isinstance(state, dict) and not self.config.get('store_full_input', True):
                # full_input / compact の両方削除 (既に圧縮済みでも除去)
                if 'full_input' in state:
                    try: del state['full_input']
                    except Exception: pass
                if 'full_compact' in state:
                    try: del state['full_compact']
                    except Exception: pass
            if (self.config.get('use_full_features') and
                self.config.get('enable_compact_full_input', True) and
                self.config.get('store_full_input', True) and
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
                        try:
                            del state['full_input']
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
            self.replay_buffer.append(sample)
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

        batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
        # フル特徴量モード時に旧フォーマット(feature_version=0)サンプルを除外
        if self.config.get('use_full_features'):
            filtered = [s for s in batch if s.get('feature_version', 0) >= 1]
            if not filtered:
                return {"loss": None, "reason": "no_full_feature_samples"}
            batch = filtered

        policy_losses = []
        value_losses = []
        entropies = []
        valid = 0
        collected_pi = []
        collected_model = []
        collected_v_pred = []
        collected_v_t = []
        variable = getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate')

        vectorized_ok = False
        if not variable and hasattr(self.model, 'forward_batch'):
            try:
                import numpy as _np
                states = []
                pi_arrays = []  # list[np.ndarray]
                v_targets_list = []
                lengths = []
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
                    lengths.append(int(pi_arr.shape[0]))
                if states:
                    # モデル一括 forward
                    policy_logits_batch, value_logits_batch = self.model.forward_batch(states)
                    import torch
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
                    value_loss_all = self.bce_logits_loss_fn(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1))  # mean over batch
                    # Entropy
                    entropy_all = - (probs * log_probs).sum(dim=1)
                    # 集約
                    policy_losses.append(policy_loss_all.mean())
                    value_losses.append(value_loss_all)  # already mean
                    entropies.append(entropy_all.mean())
                    valid = len(states)
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
                value_loss = self.bce_logits_loss_fn(v_logit.unsqueeze(0), v_t.unsqueeze(0))
                v_prob = torch.sigmoid(v_logit)
                entropy = -(probs * log_probs).sum()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                valid += 1
                collected_pi.append(pi_t.detach())
                collected_model.append(probs.detach())
                collected_v_pred.append(v_prob.detach())
                collected_v_t.append(v_t.detach())

        if valid == 0:
            return {"loss": None, "reason": "no_valid_samples"}

        policy_loss_mean = torch.stack(policy_losses).mean()
        value_loss_mean = torch.stack(value_losses).mean()
        entropy_mean = torch.stack(entropies).mean()
        total_loss = (self.policy_loss_coef * policy_loss_mean +
                      self.value_loss_coef * value_loss_mean -
                      self.entropy_coef * entropy_mean)
        self._optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            import torch as _t
            _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        did_update = False
        self._optimizer.step()
        did_update = True
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
        # pos_rate 警告
        try:
            warn_th = float(self.config.get('pos_rate_warn_threshold', 0.02))
            if pos_rate is not None and pos_rate < warn_th:
                print(f"[WARN] value positive sample rate low ({pos_rate:.2%}) < {warn_th:.2%}")
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
                            f"ent={metrics.get('entropy'):.3f}" if metrics.get('entropy') is not None else None,
                            f"kl={metrics.get('policy_kl'):.4f}" if metrics.get('policy_kl') is not None else None,
                            f"top1={metrics.get('policy_top1_match'):.3f}" if metrics.get('policy_top1_match') is not None else None,
                            f"v_acc={metrics.get('value_acc'):.3f}" if metrics.get('value_acc') is not None else None,
                            f"v_brier={metrics.get('value_brier'):.4f}" if metrics.get('value_brier') is not None else None,
                            f"pos={metrics.get('pos_rate'):.3f}" if metrics.get('pos_rate') is not None else None,
                            f"cum_pos={metrics.get('cum_pos_rate'):.3f}" if metrics.get('cum_pos_rate') is not None else None,
                            f"samples={metrics.get('samples')}" if metrics.get('samples') is not None else None,
                        ]
                        line = " ".join(p for p in parts if p is not None)
                        self.logger.log_text(f"[TRAIN] step={step} {line}", also_print=False)
            except Exception:
                pass
        return metrics

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
