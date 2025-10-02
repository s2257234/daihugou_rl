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
        # モデル世代 (Trainer 側で更新される想定)。データ多様性確保用にサンプルへ埋め込む。
        self.model_version = config.get("current_model_version", 0) if isinstance(config, dict) else 0

        # フェーズ中サンプル保持 (フェーズ確定時にラベル付与)
        self._phase_samples: List[Any] = []
        self.env_ref = None  # 直近参照環境
        self.logger = None   # 外部ロガー (TensorBoard 等)
        self._logged_inside = False  # 二重記録防止

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
        self.pos_weight = float(self.config.get("value_pos_weight", 1.0))

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

        root = self._run_mcts(env)
        actions = list(root.children.keys())
        visits = [child.visit_count for child in root.children.values()]

        # 温度決定 (手数/進行に応じたスケジュール)
        cur_temp = self._select_temperature(training=training)
        pi = softmax_temperature_policy(visits, cur_temp)

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
                        "temperature": cur_temp,
                    })
            except Exception:
                pass

        # π に従い行動サンプリング (行動なしなら pass)
        chosen = random.choices(actions, weights=pi, k=1)[0] if actions else "pass"
        action_env = None if chosen == "pass" else chosen
        if isinstance(action_env, tuple):  # tuple を list 化
            action_env = list(action_env)
        action_env = self._validate_action(env, action_env)

        # リプレイサンプル保存 (value=None : 未確定)
        state_repr = self._extract_state(env)
        serialized_legal = [None if a == "pass" else (list(a) if isinstance(a, tuple) else a) for a in actions]
        try:
            # ルート価値を再取得 (MCTS 内で破棄されるため再計算)
            _, value_scalar_for_store = self._policy_value(env)
        except Exception:
            value_scalar_for_store = 0.5
        stored = self._store_sample(state_repr, serialized_legal, pi, None, value_pred=value_scalar_for_store)
        self._phase_samples.append(stored)
        self.move_count += 1
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

    # ---------------- Core (MCTS) ----------------
    def _run_mcts(self, env) -> PUCTNode:
        """環境を軽量コピーし PUCT MCTS を実行してルートノードを返す."""
        env_copy = self._copy_env(env)

        def policy_value_fn(e):  # ノード展開時に prior と value を取得
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
        def policy_value_batch_fn(env_list):
            try:
                if self.model is None:
                    outs = []
                    for e in env_list:
                        legal = self._get_legal_actions(e)
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
                                logits_b, value_vec_b = self.model.forward_batch(states)
                    except Exception:
                        logits_b, value_vec_b = self.model.forward_batch(states)
                    outs = []
                    for i, e in enumerate(env_list):
                        legal = self._get_legal_actions(e)
                        if not legal:
                            outs.append(({}, 0.0))
                            continue
                        n = len(legal)
                        logits_t = logits_b[i]
                        if hasattr(logits_t, 'shape'):
                            import torch as _t
                            if logits_t.shape[0] < n:
                                pad = _t.zeros(n - logits_t.shape[0], device=logits_t.device)
                                logits_t = _t.cat([logits_t, pad], dim=0)
                            logits_t = logits_t[:n]
                            logits = logits_t.tolist()
                        else:
                            logits = list(logits_t)[:n]
                            if len(logits) < n:
                                logits += [0.0] * (n - len(logits))
                        # value
                        v_vec = value_vec_b[i]
                        pid = getattr(self, 'player_id', 0)
                        if hasattr(v_vec, 'shape') and 0 <= pid < v_vec.shape[0]:
                            v_scalar = float(v_vec[pid].item())
                        else:
                            try:
                                v_scalar = float(v_vec[pid])
                            except Exception:
                                v_scalar = float(v_vec[0]) if hasattr(v_vec, '__getitem__') else float(v_vec)
                        # softmax
                        import math as _m
                        mx = max(logits) if logits else 0.0
                        exps = [_m.exp(x - mx) for x in logits]
                        s = sum(exps)
                        probs = [e_ / s for e_ in exps] if s > 0 else [1.0 / n] * n
                        outs.append(({legal[j]: probs[j] for j in range(n)}, v_scalar))
                    return outs
            except Exception:
                pass
            # 失敗時は逐次版へフォールバック
            return [self._policy_value(e) for e in env_list]

        return run_puct_mcts(
            root_env_copy=env_copy,
            num_simulations=self.num_simulations,
            policy_value_fn=policy_value_fn,
            policy_value_batch_fn=policy_value_batch_fn,
            get_legal_actions_fn=legal_fn,
            c_puct=self.puct_c,
            add_dirichlet=True,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            root_player_id=getattr(env.game, "turn", 0),
            batch_eval_size=self.mcts_batch_eval_size,
            transposition_table=TT,
        )

    def _policy_value(self, env):
        """(合法手→事前確率dict, 自プレイヤー視点value) を返す。

        モデル未設定時は一様分布 + value=0。可変長アクション対応モデルなら evaluate() を使用。
        """
        legal = self._get_legal_actions(env)
        if not legal:
            return {}, 0.0
        n = len(legal)
        state = self._extract_state(env)

        # モデルが無ければ一様 prior
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
                    logits, value_scalar = self.model.evaluate(state, legal)
                else:
                    # 固定ヘッドはAMPでバッチ/単発推論
                    with _t.amp.autocast('cuda', enabled=_use_amp):
                        logits_t, value_vec_t = self.model.forward(state)  # tensors
                    if hasattr(logits_t, 'shape') and logits_t.shape[0] < n:  # 念のためパディング
                        pad = _t.zeros(n - logits_t.shape[0], device=getattr(logits_t, 'device', None))
                        logits_t = _t.cat([logits_t, pad], dim=0)
                    logits_t = logits_t[:n]
                    pid = getattr(self, 'player_id', 0)
                    if hasattr(value_vec_t, 'shape') and 0 <= pid < value_vec_t.shape[0]:
                        value_scalar = float(value_vec_t[pid].item())
                    else:
                        value_scalar = float(value_vec_t[0].item())
                    logits = logits_t.tolist() if hasattr(logits_t, 'tolist') else list(logits_t)
        except Exception:
            # フォールバック（従来通り）
            if getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate'):
                logits, value_scalar = self.model.evaluate(state, legal)
            else:
                logits_t, value_vec_t = self.model.forward(state)
                if hasattr(logits_t, 'shape') and logits_t.shape[0] < n:
                    import torch as _t
                    pad = _t.zeros(n - logits_t.shape[0])
                    logits_t = _t.cat([logits_t, pad], dim=0)
                logits_t = logits_t[:n]
                pid = getattr(self, 'player_id', 0)
                try:
                    value_scalar = float(value_vec_t[pid].item())
                except Exception:
                    value_scalar = float(value_vec_t[0].item())
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
        return out, float(value_scalar)

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

    def _get_legal_actions(self, env) -> List[Any]:
        """現在手番プレイヤーの合法手集合を可変長リストで返す (最後に必ず pass を追加)。"""
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
        if "pass" not in seen:  # パスを保証
            acts.append("pass")
        return acts

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

    def _extract_state(self, env):
        """状態特徴を抽出。
        config.use_full_features が True の場合は拡張特徴量ベクトル full_input を生成。
        full_input レイアウト (順序固定):
            For each player i (0..N-1):
              - 53 bits: 各カード所持 (標準52 + Joker1) (1/0)
              - 1 bit : pass フラグ (現ターンでパス状態)
              - 1 scalar: 残り枚数 (正規化 0..1, /53)
            Field block:
              - 1 bit revolution
              - 7 bits combo type one-hot (empty,single,pair,triple,four,straight,joker)
              - 13 bits field base rank one-hot (同ランク系: その rank, 階段: 先頭ランク, joker_single: none all zero, empty: all zero)
              - 1 scalar: field_size / 13
            Turn one-hot (N)
        合計次元 = players * (53+1+1) + (1+7+13+1) + N
        """
        try:
            g = env.game
            pid = g.turn
            rule_checker = getattr(g, "rule_checker", None)
            revo = bool(getattr(rule_checker, "revolution", False)) if rule_checker else False
            use_full = bool(getattr(self, 'config', {}).get('use_full_features', False))
            base = {
                "turn": pid,
            }
            me = g.players[pid]
            base.update({
                "hand_size": len(me.hand),
                "field_size": len(g.current_field),
                "revolution": revo,
            })
            if not use_full:
                return base
            # --- フル特徴生成 ---
            num_players = len(g.players)
            # 統一フォーマット次元: per_player(53bits + pass + remain =55) * N + field(22) + turn_onehot(N) = 56N + 22
            expected_full_dim = 56 * num_players + 22
            # カードインデックス: suit*13 + (rank-1) => 0..51, Joker => 52
            def card_index(card):
                if card.is_joker:
                    return 52
                suit_order = {'♠':0,'♥':1,'♦':2,'♣':3}
                return suit_order.get(card.suit,0)*13 + (card.rank-1)
            per_player_dim = 53 + 1 + 1
            players_block = []
            for i, pl in enumerate(g.players):
                bits = [0.0]*53
                for c in pl.hand:
                    try:
                        idx = card_index(c)
                        if 0 <= idx < 53:
                            bits[idx] = 1.0
                    except Exception:
                        pass
                pass_bit = 1.0 if (i < len(g.passed) and g.passed[i]) else 0.0
                remain_norm = len(pl.hand)/53.0
                players_block.extend(bits + [pass_bit, remain_norm])
            # Field combo type
            field = g.current_field
            combo_type_onehot = [0.0]*7  # empty,single,pair,triple,four,straight,joker_single
            rank_onehot = [0.0]*13
            field_size = len(field)
            if field_size == 0:
                combo_type_onehot[0] = 1.0
                base_rank = None
            else:
                combo = rule_checker.classify_combo(field) if rule_checker else None
                ctype = combo['type'] if combo else None
                mapping = {
                    'single':1,
                    'pair':2,
                    'triple':3,
                    'four':4,
                    'straight':5,
                    'joker_single':6,
                }
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
                    # rank 1..13 -> index 0..12 (A=1 -> 0)
                    rank_onehot[base_rank-1] = 1.0
            revolution_bit = 1.0 if revo else 0.0
            field_size_norm = field_size/13.0
            field_block = [revolution_bit] + combo_type_onehot + rank_onehot + [field_size_norm]
            # turn one-hot
            turn_onehot = [0.0]*num_players
            if 0 <= pid < num_players:
                turn_onehot[pid] = 1.0
            full_vec = players_block + field_block + turn_onehot
            # 次元検証 & 補正 (不足はゼロ埋め / 超過は切り詰め) 常に expected_full_dim に揃える
            cur_len = len(full_vec)
            if cur_len != expected_full_dim:
                if cur_len < expected_full_dim:
                    full_vec = full_vec + [0.0] * (expected_full_dim - cur_len)
                else:
                    full_vec = full_vec[:expected_full_dim]
                if not hasattr(self, '_warned_full_dim_autofix'):
                    print(f"[WARN] adjusted full_input length from {cur_len} to expected {expected_full_dim}")
                    self._warned_full_dim_autofix = True  # type: ignore[attr-defined]
            base['full_input'] = full_vec
            base['full_input_dim'] = expected_full_dim
            # フル特徴量次元一貫性チェック
            try:
                if self.config.get('use_full_features'):
                    cur_dim = expected_full_dim
                    ref = getattr(self, '_full_input_dim_ref', None)
                    if ref is None:
                        self._full_input_dim_ref = cur_dim
                    elif ref != cur_dim:
                        print(f"[WARN] full_input_dim mismatch expected={ref} got={cur_dim}")
            except Exception:
                pass
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
                "feature_version": 1 if (isinstance(state, dict) and ('full_input' in state)) else 0,
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
            if (self.config.get('use_full_features') and
                self.config.get('enable_compact_full_input', True) and
                isinstance(state, dict) and 'full_input' in state and 'full_compact' not in state):
                fi = state.get('full_input')
                import numpy as _np
                fi_arr = _np.asarray(fi, dtype=_np.float32)
                # num_players 推定 (len = 56N + 22)
                total_len = fi_arr.shape[0]
                # 56N + 22 = total_len -> N = (total_len - 22)/56
                N = int((total_len - 22) // 56) if total_len >= 22 else self.config.get('num_players', 4)
                if N > 0 and 56 * N + 22 == total_len:
                    # binary_len = 55N + 21, float count = N + 1
                    binary_indices = []
                    float_indices = []
                    # players block
                    for p in range(N):
                        base = p * 55
                        # 53 card bits
                        binary_indices.extend(range(base, base + 53))
                        # pass bit
                        binary_indices.append(base + 53)
                        # remain_norm
                        float_indices.append(base + 54)
                    field_base = 55 * N
                    # revolution
                    binary_indices.append(field_base)
                    # combo 7 bits
                    binary_indices.extend(range(field_base + 1, field_base + 8))
                    # rank 13 bits
                    binary_indices.extend(range(field_base + 8, field_base + 21))
                    # field_size_norm
                    float_indices.append(field_base + 21)
                    # turn one-hot N bits
                    turn_start = field_base + 22
                    binary_indices.extend(range(turn_start, turn_start + N))
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
                    }
                    # 元のベクトルは削除して常駐メモリ削減
                    try:
                        del state['full_input']
                    except Exception:
                        pass
        except Exception:
            pass
        # --- 追加メモリ削減: pi 量子化(uint16), value/value_pred を uint8、legal_actions を ID 化 ---
        import numpy as _np
        # グローバル action 辞書 (プロセス内共有) を lazy 初期化
        global _ACTION_ID_MAP, _ACTION_ID_LIST
        try:
            _ACTION_ID_MAP  # type: ignore
        except NameError:
            _ACTION_ID_MAP = {}  # type: ignore
            _ACTION_ID_LIST = []  # type: ignore

        def _encode_actions(acts):
            ids = []
            for a in acts:
                # None/pass も区別: 文字列化 (tuple/list は repr)
                if a is None:
                    key = 'PASS'
                else:
                    try:
                        if isinstance(a, (list, tuple)):
                            key = 'T:' + ','.join(map(str, a))
                        else:
                            key = 'S:' + str(a)
                    except Exception:
                        key = 'S:ERR'
                if key not in _ACTION_ID_MAP:  # type: ignore
                    _ACTION_ID_MAP[key] = len(_ACTION_ID_LIST)  # type: ignore
                    _ACTION_ID_LIST.append(a)  # type: ignore
                ids.append(_ACTION_ID_MAP[key])  # type: ignore
            return _np.asarray(ids, dtype=_np.int32)

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
        # legal actions を ID 配列化
        acts_ids = _encode_actions(legal_actions or [])

        sample = {
            "player_id": self.player_id,
            "state": state,
            # 量子化/圧縮表現
            "pi_q": pi_q,
            "pi_format": "u16_norm65535",
            "legal_ids": acts_ids,
            "actions_format": "id_v1",
            "value_u8": value_u8,
            "value_pred_u8": value_pred_u8,
            "value": value,  # assign_values 更新対象
            "model_version": getattr(self, 'model_version', 0),
            "feature_version": 1 if (isinstance(state, dict) and ('full_input' in state or 'full_compact' in state)) else 0,
        }
        # (Option) raw pi を保持: 分析/デバッグのため。drop_raw_pi=False かつ pi 長さが legal_ids と一致する場合のみ。
        if not self.config.get('drop_raw_pi', True):
            try:
                import numpy as _np
                if len(pi) == len(acts_ids):
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
            self._phase_samples = []

    def finalize_game(self, *_args, **_kwargs):  # 互換維持用 no-op
        """ゲーム終端フック (最終順位報酬を使わないので何もしない)。"""
        self._phase_samples = []

    # ---------------- Persistence ----------------
    def save_replay(self, path: Optional[str] = None):
        """リプレイバッファを joblib で保存 (圧縮3)。"""
        path = path or self.config.get("replay_path", "replay_buffer.joblib")
        try:
            joblib.dump(self.replay_buffer, path, compress=3)
        except Exception:
            joblib.dump(self.replay_buffer, path)

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
        # 共有バッファ: 自プレイヤーの確定サンプルのみ抽出
        if self._use_shared and hasattr(self.replay_buffer, 'iter_all'):
            my_samples = [s for s in self.replay_buffer.iter_all(owner_pid=self.player_id) if s.get("value") is not None]
            if not my_samples:
                return {"loss": None, "reason": "no_data"}
            batch_pool = my_samples
        else:  # ローカル
            if not self.replay_buffer:
                return {"loss": None, "reason": "no_data"}
            batch_pool = self.replay_buffer

        # Optimizer 遅延初期化
        if self._optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
        # フル特徴量モード時に旧フォーマット(feature_version=0)サンプルを除外
        if self.config.get('use_full_features'):
            filtered = [s for s in batch if s.get('feature_version', 0) == 1]
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

        for sample in batch:
            # --- 復元: legal_actions / pi ---
            legal_actions = sample.get("legal_actions")
            if legal_actions is None and sample.get('actions_format') == 'id_v1' and 'legal_ids' in sample:
                # ID から元アクションへ (必要時のみ)。学習上は長さ一致だけで良いならダミー化も可能。
                try:
                    global _ACTION_ID_LIST
                    ids = sample['legal_ids']
                    if hasattr(ids, 'tolist'):
                        ids_list = ids.tolist()
                    else:
                        ids_list = list(ids)
                    legal_actions = []
                    for i in ids_list:
                        try:
                            legal_actions.append(_ACTION_ID_LIST[i])  # type: ignore
                        except Exception:
                            legal_actions.append('pass')
                except Exception:
                    legal_actions = None
            # π 復元 (量子化優先)
            if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                try:
                    import numpy as _np
                    pi_q = sample['pi_q']
                    if hasattr(pi_q, 'astype'):
                        pi_arr = pi_q.astype(_np.float32)
                    else:
                        pi_arr = _np.asarray(list(pi_q), dtype=_np.float32)
                    s_q = float(pi_arr.sum())
                    if s_q <= 0:
                        pi_target = [1.0 / len(pi_arr)] * int(len(pi_arr)) if len(pi_arr) > 0 else []
                    else:
                        pi_target = (pi_arr / s_q).tolist()
                except Exception:
                    pi_target = sample.get('pi')
            else:
                pi_target = sample.get("pi")
            v_target = sample.get("value")
            if v_target is None and 'value_u8' in sample:
                vu = sample.get('value_u8')
                try:
                    if isinstance(vu, int) and vu != 255:
                        v_target = vu / 255.0
                except Exception:
                    pass
            if not legal_actions or not pi_target or v_target is None:
                continue  # 無効サンプルスキップ
            # フル特徴量モデルで zero padded サンプルを除外 (config 制御)
            if self.config.get('use_full_features') and self.config.get('skip_zero_padded_full_samples', True):
                try:
                    st = sample.get('state') or {}
                    # compact / full_input が一切無い場合 (models.PolicyValueNet で警告したケース)
                    if ('full_compact' not in st) and ('full_input' not in st):
                        continue
                except Exception:
                    pass
            n = len(legal_actions)
            # Forward
            if variable:
                logits_raw, v_pred_raw = self.model.evaluate(sample["state"], legal_actions)
            else:
                logits_raw, v_out = self.model.forward(sample["state"])  # policy_logits, value_vec
                if hasattr(v_out, 'shape'):
                    pid = getattr(self, 'player_id', 0)
                    if 0 <= pid < v_out.shape[0]:
                        v_pred_raw = v_out[pid]
                    else:
                        v_pred_raw = v_out[0]
                else:
                    v_pred_raw = v_out

            # Logits -> tensor & サイズ調整
            if hasattr(logits_raw, 'shape'):
                logits_t = logits_raw
                if logits_t.shape[0] < n:  # 念のためパディング
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
            if pi_t.shape[0] != log_probs.shape[0]:  # 念のため揃える
                m = min(pi_t.shape[0], log_probs.shape[0])
                pi_t = pi_t[:m]
                log_probs = log_probs[:m]
                probs = probs[:m]
            policy_loss = -(pi_t * log_probs).sum()

            # Value loss (BCE) 手動展開 (安定化のため clamp)
            if isinstance(v_pred_raw, float):
                v_pred_t = torch.tensor(v_pred_raw, dtype=torch.float32)
            else:
                v_pred_t = v_pred_raw.float()
            v_t = torch.tensor(float(v_target), dtype=torch.float32, device=v_pred_t.device)
            eps = 1e-7
            v_clamped = v_pred_t.clamp(eps, 1 - eps)
            # クラス不均衡対策 (pos_weight) 適用
            pos_w = self.pos_weight if v_t.item() > 0.5 else 1.0
            value_loss = - (pos_w * v_t * v_clamped.log() + (1 - v_t) * (1 - v_clamped).log())
            entropy = -(probs * log_probs).sum()

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            valid += 1

            # 解析用に各分布と value を保存
            collected_pi.append(pi_t.detach())
            collected_model.append(probs.detach())
            collected_v_pred.append(v_clamped.detach())
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
        self._optimizer.step()

        # 追加メトリクス計算
        policy_kl = None
        policy_top1 = None
        value_acc = None
        value_brier = None
        pos_rate = None
        if collected_pi:
            try:
                import torch as _t
                kl_list = []
                top1_list = []
                v_hit_list = []
                brier_list = []
                v_label_list = []
                for pi_t, model_p, v_pred_c, v_lab in zip(collected_pi, collected_model, collected_v_pred, collected_v_t):
                    kl = (pi_t * (pi_t.add(1e-12).log() - model_p.add(1e-12).log())).sum().item()
                    kl_list.append(kl)
                    if pi_t.numel() > 0 and model_p.numel() > 0:
                        top1_list.append(1.0 if int(pi_t.argmax()) == int(model_p.argmax()) else 0.0)
                    v_hit_list.append(1.0 if (float(v_pred_c) > 0.5) == (float(v_lab) > 0.5) else 0.0)
                    brier_list.append(float((v_pred_c - v_lab).pow(2).item()))
                    v_label_list.append(float(v_lab.item()))
                if kl_list:
                    policy_kl = float(sum(kl_list) / len(kl_list))
                if top1_list:
                    policy_top1 = float(sum(top1_list) / len(top1_list))
                if v_hit_list:
                    value_acc = float(sum(v_hit_list) / len(v_hit_list))
                if brier_list:
                    value_brier = float(sum(brier_list) / len(brier_list))
                if v_label_list:
                    pos_rate = float(sum(1.0 if v>0.5 else 0.0 for v in v_label_list) / len(v_label_list))
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
                        if self._use_shared and hasattr(self.replay_buffer, '__len__'):
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

    def __getitem__(self, k):
        store = self._agent._mcts_tt
        key = self._mk(k)
        res, _tick = store[key]
        self._agent._mcts_tt_tick += 1
        store[key] = (res, self._agent._mcts_tt_tick)
        return res

    def __setitem__(self, k, v):
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
