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
        # リプレイバッファ (共有 or ローカル list)。共有時は trainer 注入想定。
        self.replay_buffer: Any = []
        self.max_buffer_size = self.config.get("buffer_size", 50000)
        self._use_shared = self.config.get("use_shared_replay", False)

        # MCTS 関連パラメータ
        self.num_simulations = self.config.get("num_simulations", 64)
        self.puct_c = self.config.get("puct_c", 1.4)
        self.dirichlet_alpha = self.config.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = self.config.get("dirichlet_epsilon", 0.25)

        # 温度スケジュール (序盤探索重視 → 後半確定的)
        self.temperature = self.config.get("temperature", 1.0)
        self.temperature_low = self.config.get("temperature_low", 0.05)
        self.temperature_decay_moves = self.config.get("temperature_decay_moves", 20)
        self.move_count = 0  # エピソード内手数カウンタ

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

    # ---------------- Public API ----------------
    def set_model(self, model):
        """後から学習済みモデルを差し替える."""
        self.model = model

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

        # 温度決定 (序盤: high / 後半: low / 評価: near-greedy)
        if training:
            cur_temp = self.temperature if self.move_count < self.temperature_decay_moves else self.temperature_low
        else:
            cur_temp = 1e-6
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

    # ---------------- Core (MCTS) ----------------
    def _run_mcts(self, env) -> PUCTNode:
        """環境を軽量コピーし PUCT MCTS を実行してルートノードを返す."""
        env_copy = self._copy_env(env)

        def policy_value_fn(e):  # ノード展開時に prior と value を取得
            return self._policy_value(e)

        def legal_fn(e):  # 合法手生成
            return self._get_legal_actions(e)

        return run_puct_mcts(
            root_env_copy=env_copy,
            num_simulations=self.num_simulations,
            policy_value_fn=policy_value_fn,
            get_legal_actions_fn=legal_fn,
            c_puct=self.puct_c,
            add_dirichlet=True,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            root_player_id=getattr(env.game, "turn", 0),
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
        if getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate'):
            logits, value_scalar = self.model.evaluate(state, legal)
        else:
            logits_t, value_vec_t = self.model.forward(state)  # tensors
            if logits_t.shape[0] < n:  # 念のためパディング
                import torch as _t
                pad = _t.zeros(n - logits_t.shape[0])
                logits_t = _t.cat([logits_t, pad], dim=0)
            logits_t = logits_t[:n]
            pid = getattr(self, 'player_id', 0)
            if 0 <= pid < value_vec_t.shape[0]:
                value_scalar = float(value_vec_t[pid].item())
            else:
                value_scalar = float(value_vec_t[0].item())
            logits = logits_t.tolist()

        # softmax 正規化
        import math as _m
        mx = max(logits) if logits else 0.0
        exps = [_m.exp(x - mx) for x in logits]
        s = sum(exps)
        probs = [e / s for e in exps] if s > 0 else [1.0 / n] * n
        return {legal[i]: probs[i] for i in range(n)}, float(value_scalar)

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
        """学習用の簡易状態特徴を dict で抽出 (手札枚数 / 場枚数 / 手番ID / 革命フラグ)。"""
        try:
            g = env.game
            pid = g.turn
            me = g.players[pid]
            rule_checker = getattr(g, "rule_checker", None)
            revo = bool(getattr(rule_checker, "revolution", False)) if rule_checker else False
            return {
                "hand_size": len(me.hand),
                "field_size": len(g.current_field),
                "turn": pid,
                "revolution": revo,
            }
        except Exception:
            return {"turn": 0}

    # ---------------- Replay buffer ----------------
    def _store_sample(self, state, legal_actions, pi, value, value_pred: Optional[float] = None):
        """リプレイサンプル1件を保存。共有バッファなら append の参照を返す."""
        sample = {
            "player_id": self.player_id,
            "state": state,
            "legal_actions": legal_actions,
            "pi": pi,
            "value": value,
            "value_pred": value_pred,
            "model_version": getattr(self, 'model_version', 0),
        }
        if self._use_shared and hasattr(self.replay_buffer, 'append'):
            self.replay_buffer.append(sample)
            return sample
        if len(self.replay_buffer) >= self.max_buffer_size:  # 古いものをFIFOで削除
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
            self.replay_buffer = joblib.load(path)
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
            legal_actions = sample.get("legal_actions")
            pi_target = sample.get("pi")
            v_target = sample.get("value")
            if not legal_actions or not pi_target or v_target is None:
                continue  # 無効サンプルスキップ
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
            value_loss = -(v_t * v_clamped.log() + (1 - v_t) * (1 - v_clamped).log())
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
        }
        if self.logger:
            self.logger.log_train(metrics)
            self._logged_inside = True
        return metrics

    def reset_episode(self):
        """エピソード開始時にカウンタ類を初期化."""
        self.move_count = 0
        self.episode_phase_total = 0
        self.episode_phase_correct = 0


DRLAgent = AlphaZeroAgent
