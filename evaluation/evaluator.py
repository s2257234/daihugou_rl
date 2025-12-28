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
import multiprocessing as mp
import time

from agents.models import PolicyValueNet
from agents.drl_agent import AlphaZeroAgent
from agents.config import ALPHA_ZERO_CONFIG
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from game.environment import DaifugoSimpleEnv
from evaluation.rating import RatingManager, EloConfig

# ======================================================
# 並列評価用ワーカー (Windows 対応: トップレベル関数)
# ======================================================
_EV_CFG: Dict[str, Any] | None = None
_EV_DEVICE: str | None = None
_EV_CKPT: str | None = None
_EV_MODEL = None
_EV_PAST_MODELS: List[Any] | None = None
_EV_PAST_CKPTS: List[str] | None = None
_EV_BASELINE_MIX: List[str] | None = None
_EV_SEAT_ROTATION: bool = True
_EV_VALUE_THRESHOLD: float = 0.5
_EV_WRITE_ZERO_ON_MISSING: bool = False

def _parallel_eval_init(checkpoint_path: str,
                        device: str,
                        base_cfg: Dict[str, Any],
                        baseline_mix: List[str],
                        det_mode_eval: str | None,
                        past_checkpoints: List[str] | None,
                        seed: int | None,
                        value_threshold: float = 0.5,
                        write_zero_on_missing: bool = False):
    # グローバルを書き換え
    global _EV_CFG, _EV_DEVICE, _EV_CKPT, _EV_MODEL, _EV_PAST_MODELS, _EV_PAST_CKPTS, _EV_BASELINE_MIX, _EV_SEAT_ROTATION
    import random as _rnd
    if seed is not None:
        _rnd.seed(seed + os.getpid())
    _EV_DEVICE = device
    _EV_CKPT = checkpoint_path
    # 評価時の安定設定を適用した config を構築
    cfg = dict(base_cfg)
    try:
        cfg["inference_dirichlet"] = False
        cfg["dirichlet_epsilon"] = 0.0
        cfg["temperature"] = 0.0
        cfg["opening_random_enable"] = False
    except Exception:
        pass
    if det_mode_eval is not None:
        cfg["determinization_mode_eval"] = det_mode_eval
    _EV_CFG = cfg
    _EV_BASELINE_MIX = list(baseline_mix or ["rule", "random", "random"])
    _EV_PAST_CKPTS = list(past_checkpoints or [])
    _EV_SEAT_ROTATION = True  # 座席回転の有無は呼び出し側でエピソード順に渡すため常に有効扱い
    _EV_VALUE_THRESHOLD = float(value_threshold)
    _EV_WRITE_ZERO_ON_MISSING = bool(write_zero_on_missing)
    # モデル読み込み
    from agents.models import PolicyValueNet as _PVN
    try:
        _EV_MODEL = _PVN.load(checkpoint_path, map_location=device)
        try:
            _EV_MODEL.to(device)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            _EV_MODEL.eval()
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] eval worker failed to load main model: {e}")
        _EV_MODEL = None
    # 過去モデル読み込み（最大3つまで使用）
    _EV_PAST_MODELS = []
    if _EV_PAST_CKPTS:
        from agents.models import PolicyValueNet as _PVN2
        for p in _EV_PAST_CKPTS[:3]:
            try:
                m = _PVN2.load(p, map_location=device)
                try:
                    m.to(device)  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    m.eval()
                except Exception:
                    pass
                _EV_PAST_MODELS.append((p, m))
            except Exception as _e:
                print(f"[WARN] eval worker failed to load past model '{p}': {_e}")

def _parallel_eval_one(ep_index: int) -> Dict[str, Any]:
    """1エピソードを実行し、順位とラベルを返す。"""
    from agents.drl_agent import AlphaZeroAgent as _AZ
    from agents.rule_based_agent import RuleBasedAgent as _RB
    from agents.random_agent import RandomAgent as _RA
    from game.environment import DaifugoSimpleEnv as _Env
    # グローバル参照
    cfg = dict(_EV_CFG or {})
    device = _EV_DEVICE or "cpu"
    # エージェント構築（ベース順序）
    agents: List[Any] = []
    labels: List[str] = []
    # 評価対象
    ag_eval = _AZ(player_id=0, model=_EV_MODEL, config=cfg)
    agents.append(ag_eval)
    labels.append("AlphaZeroAgent")
    # 過去モデル最大3
    for p, m in (_EV_PAST_MODELS or [])[:3]:
        pid = len(agents)
        ag = _AZ(player_id=pid, model=m, config=cfg)
        agents.append(ag)
        labels.append(f"AlphaZeroAgent@{os.path.basename(p)}")
    # 不足分は baseline_m mix で補完
    needed = 4 - len(agents)
    mix = list(_EV_BASELINE_MIX or [])
    while len(mix) < needed:
        mix.append("random")
    class_counts: Dict[str, int] = {}
    for spec in mix[:needed]:
        pid = len(agents)
        if spec == "rule":
            ag = _RB(player_id=pid)
            cname = "RuleBasedAgent"
        else:
            ag = _RA(player_id=pid)
            cname = "RandomAgent"
        agents.append(ag)
        class_counts[cname] = class_counts.get(cname, 0) + 1
        labels.append(f"{cname}#{class_counts[cname]}")
    agents = agents[:4]
    labels = labels[:4]
    # 座席回転
    r = ep_index % 4
    if r != 0:
        agents = agents[r:] + agents[:r]
        labels = labels[r:] + labels[:r]
    # seat id 割当
    for i, ag in enumerate(agents):
        try:
            ag.player_id = i
        except Exception:
            pass
    # 対戦
    env = _Env(num_players=4, agent_classes=None)
    env.agents = agents
    if hasattr(ag_eval, 'set_env_ref'):
        try:
            ag_eval.set_env_ref(env)
        except Exception:
            pass
    env.reset()
    step_limit = 1000
    steps = 0
    eval_preds: List[float] = []
    while not getattr(env.game, 'done', False):
        if steps >= step_limit:
            break
        current_player_id = env.game.turn
        agent = env.agents[current_player_id]
        if isinstance(agent, _AZ):
            # collect value prediction for the evaluated model instance
            try:
                # only collect preds from the main evaluated agent object (ag_eval)
                if agent is ag_eval:
                    _, v = agent._policy_value(env)
                    if isinstance(v, dict):
                        v_prob = float(v.get(getattr(agent, 'player_id', 0), 0.0))
                    elif isinstance(v, (list, tuple)):
                        pid = int(getattr(agent, 'player_id', 0) or 0)
                        try:
                            v_prob = float(v[pid])
                        except Exception:
                            v_prob = float(v[0]) if v else 0.0
                    else:
                        v_prob = float(v)
                    eval_preds.append(v_prob if v_prob is not None else 0.0)
            except Exception:
                try:
                    eval_preds.append(0.5)
                except Exception:
                    pass
            action = agent.select_action(env, training=False)
        else:
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
        remaining = [i for i in range(4) if i not in rankings]
        rankings += remaining
    # compute recall for evaluated model (ag_eval)
    recall = None
    try:
        try:
            eval_seat = agents.index(ag_eval)
        except Exception:
            eval_seat = 0
        won = (len(rankings) > 0 and rankings[0] == eval_seat)
        if won:
            tp = sum(1 for p in eval_preds if p > 0.5)
            fn = sum(1 for p in eval_preds if p <= 0.5)
            denom = tp + fn
            recall = (tp / denom) if denom > 0 else None
        else:
            recall = None
    except Exception:
        recall = None
    # ラベル順（席番号順）を返し、親で Elo を更新できるようにする
    return {"ep": ep_index, "rankings": rankings, "labels": labels, "steps": steps, "recall": recall}


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
        seat_rotation: bool = True,
        # 評価安定化のため既定で fixed_once を適用（推論/実運用は config 側デフォルトの "stochastic" を維持）
        determinization_mode_override: str | None = "fixed_once",
        past_checkpoints: List[str] | None = None,
        # 閾値: value 予測を陽性と見なすカットオフ
        value_threshold: float = 0.5,
        # 指標が計算できないときに 0.0 を出力するか (False -> 空欄)
        write_zero_on_missing_metrics: bool = False,
    ):
        if seed is not None:
            random.seed(seed)
        self.checkpoint_path = checkpoint_path
        self.device = device or self._auto_device()
        self.num_simulations = num_simulations or ALPHA_ZERO_CONFIG.get("num_simulations", 32)
        self.elo = RatingManager(save_dir=elo_dir, config=EloConfig())
        # baseline_mix 例: ["rule", "random", "random"] -> 学習エージェント + 3 baseline
        self.baseline_mix = baseline_mix or ["rule", "random", "random"]
        # 過去モデルのチェックポイント群（最大3枠まで採用）
        self.past_checkpoints = list(past_checkpoints or [])
        self.config = dict(ALPHA_ZERO_CONFIG)
        self.config["num_simulations"] = self.num_simulations
        self.config["device"] = self.device
        # 評価ではメモリ削減を優先: MCTS TT と並列デタミニゼーションプールを無効化
        try:
            self.config["enable_mcts_tt"] = False
            self.config["enable_parallel_determinization"] = False
        except Exception:
            pass
        # 評価フェーズでは探索ノイズOFF・温度0・序盤ランダム無効化を徹底
        # （AlphaZeroAgent.select_action(training=False) でも低温/ノイズ無効になるが、明示的に設定）
        try:
            self.config["inference_dirichlet"] = False  # 推論時のルートDirichletを無効化
            self.config["dirichlet_epsilon"] = 0.0      # 念のため係数も0に
            self.config["temperature"] = 0.0            # 温度0（_select_temperatureで非学習時は1e-6だが整合のため）
            self.config["opening_random_enable"] = False
        except Exception:
            pass
        # 評価時の determinization モードを上書き（デフォルトで fixed_once）。
        if determinization_mode_override is not None:
            self.config["determinization_mode_eval"] = determinization_mode_override
        # メトリクス CSV 設定
        self.metrics_csv = metrics_csv or os.path.join(elo_dir, "eval_metrics.csv")
        os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
        if not os.path.exists(self.metrics_csv):
            try:
                with open(self.metrics_csv, "w", encoding="utf-8") as f:
                    f.write("episode,win_rate,avg_rank,rating_p0,raw_rank,steps,recall,precision,auc\n")
            except Exception:
                pass
        # 各エージェント別の順位分布/勝率ログ（縦持ち, 累積）
        self.agent_winrates_csv = os.path.join(elo_dir, "agent_winrates.csv")
        try:
            os.makedirs(os.path.dirname(self.agent_winrates_csv), exist_ok=True)
        except Exception:
            pass
        header = (
            "episode,agent,win_rate,avg_rank,"
            "rank1_rate,rank2_rate,rank3_rate,rank4_rate,"
            "rank1,rank2,rank3,rank4,games\n"
        )
        need_rewrite = False
        if os.path.exists(self.agent_winrates_csv):
            try:
                with open(self.agent_winrates_csv, "r", encoding="utf-8") as f:
                    first = f.readline()
                if "rank1_rate" not in first:
                    need_rewrite = True
            except Exception:
                need_rewrite = True
        else:
            need_rewrite = True
        if need_rewrite:
            try:
                with open(self.agent_winrates_csv, "w", encoding="utf-8") as f:
                    f.write(header)
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
        # モデル読み込み (full_feature_dim 必須化に対応)
        # 優先: 既存 ckpt から load (full_feature_dim を内部推定)
        if os.path.exists(self.checkpoint_path):
            try:
                self.model = PolicyValueNet.load(self.checkpoint_path, map_location=self.device)
                # map_location で device に乗らないケース (cpu→cuda) を補正
                try:
                    self.model.to(self.device)  # type: ignore[arg-type]
                except Exception:
                    pass
                # 推論モードへ切替（Dropout/BN を停止）
                try:
                    self.model.eval()
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] checkpoint load failed ({e}) -> fallback new model")
                full_dim = 56 * self.config["num_players"] + 22
                self.model = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device=self.device,
                    full_feature_dim=full_dim,
                )
        else:
            print(f"[WARN] checkpoint not found: {self.checkpoint_path}. Using random initialized model.")
            full_dim = 56 * self.config["num_players"] + 22
            self.model = PolicyValueNet(
                max_policy_size=self.config["max_policy_size"],
                hidden_size=self.config["hidden_size"],
                num_players=self.config["num_players"],
                device=self.device,
                full_feature_dim=full_dim,
            )
        # 念のため新規作成時も eval() を明示
        try:
            self.model.eval()
        except Exception:
            pass
        # 評価用 AlphaZeroAgent（ベース座席IDは 0 だが、エピソードごとに回転可能）
        self.eval_agent = AlphaZeroAgent(player_id=0, model=self.model, config=self.config)
        self.seat_rotation = bool(seat_rotation)
        # ベースラインを含む基準順序（回転の基点）
        self.players_base = self._build_agents()
        # 安定した固有ラベルを各エージェント実体に割り当て（Elo が重複名を許容しないため）
        self._agent_label_map: Dict[int, str] = {}
        class_counts: Dict[str, int] = {}
        # 評価対象モデルの固定ラベル（旧席名P0との衝突を避ける）
        self.eval_label = "AlphaZeroAgent"
        for ag in self.players_base:
            if ag is self.eval_agent:
                self._agent_label_map[id(ag)] = self.eval_label
            else:
                # AlphaZero の過去モデルにはチェックポイント名を付与
                if isinstance(ag, AlphaZeroAgent):
                    ck = getattr(ag, "checkpoint_name", None)
                    if ck:
                        self._agent_label_map[id(ag)] = f"AlphaZeroAgent@{ck}"
                        continue
                cname = type(ag).__name__
                class_counts[cname] = class_counts.get(cname, 0) + 1
                self._agent_label_map[id(ag)] = f"{cname}#{class_counts[cname]}"
        # 現在の使用リスト（初期状態）
        self.players = list(self.players_base)
        # 初期座席IDを正規化
        self._assign_seat_ids(self.players)
        # 表示用の初期ラベル（実行時はエピソードごとに再構築）
        self.agent_labels = self._labels_for_players(self.players)

        # value -> positive の閾値と出力オプション
        self.value_threshold = float(value_threshold)
        self.write_zero_on_missing_metrics = bool(write_zero_on_missing_metrics)

    def _auto_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _release_heavy_refs_for_parallel(self):
        """並列実行時に親プロセス側の不要な巨大参照（モデル/エージェント）を解放して総メモリを抑える。"""
        # players, players_base, eval_agent を解放
        try:
            self.players = []
        except Exception:
            pass
        try:
            self.players_base = []
        except Exception:
            pass
        try:
            self.eval_agent = None
        except Exception:
            pass
        # モデルは CPU へ移してから参照を外し、CUDA ならキャッシュも解放
        try:
            import torch  # type: ignore
            if getattr(self, "model", None) is not None:
                try:
                    self.model.to("cpu")  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.model = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            try:
                self.model = None
            except Exception:
                pass

    def _make_baseline_agent(self, spec: str, player_id: int):
        if spec == "rule":
            return RuleBasedAgent(player_id=player_id)
        return RandomAgent(player_id=player_id)

    def _make_alpha_zero_eval_agent_from_ckpt(self, player_id: int, ckpt_path: str):
        """評価用の AlphaZeroAgent を過去チェックポイントから生成（ノイズOFF/温度0/eval固定）。"""
        cfg = dict(self.config)
        try:
            model = PolicyValueNet.load(ckpt_path, map_location=self.device)
            try:
                model.to(self.device)  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                model.eval()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] failed to load past checkpoint '{ckpt_path}': {e}")
            model = None
        ag = AlphaZeroAgent(player_id=player_id, model=model, config=cfg)
        # ラベル用にベース名を保持
        try:
            import os as _os
            setattr(ag, "checkpoint_name", _os.path.basename(ckpt_path))
        except Exception:
            pass
        return ag

    def _build_agents(self) -> List[Any]:
        agents: List[Any] = [self.eval_agent]
        # まず過去モデルを最大3枠まで採用
        added = 0
        for ckpt in self.past_checkpoints:
            if added >= 3:
                break
            pid = len(agents)
            agents.append(self._make_alpha_zero_eval_agent_from_ckpt(player_id=pid, ckpt_path=ckpt))
            added += 1
        # 不足分は baseline_mix で補完
        needed = 4 - len(agents)
        if needed > 0:
            mix = list(self.baseline_mix)
            while len(mix) < needed:
                mix.append("random")
            for spec in mix[:needed]:
                pid = len(agents)
                agents.append(self._make_baseline_agent(spec, player_id=pid))
        # 安全のため4人に制限
        return agents[:4]

    def _assign_seat_ids(self, players: List[Any]):
        """座席順に player_id を再割り当てする（環境・モデルの整合性のため）。"""
        for i, ag in enumerate(players):
            try:
                ag.player_id = i
            except Exception:
                pass

    def _rotate_players_for_episode(self, episode_index: int) -> List[Any]:
        """エピソードごとに座席を回転。seat_rotation=False ならそのまま返す。"""
        if not self.seat_rotation:
            return list(self.players_base)
        r = episode_index % 4
        base = list(self.players_base)
        if r == 0:
            return base
        return base[r:] + base[:r]

    def _labels_for_players(self, players: List[Any]) -> List[str]:
        labels: List[str] = []
        for ag in players:
            # 過去モデルにはファイル名を付加
            if isinstance(ag, AlphaZeroAgent) and (ag is not self.eval_agent):
                ck = getattr(ag, "checkpoint_name", None)
                if ck:
                    labels.append(f"AlphaZeroAgent@{ck}")
                    continue
            labels.append(self._agent_label_map.get(id(ag), type(ag).__name__))
        return labels

    def play_one_game(self) -> Dict[str, Any]:
        env = DaifugoSimpleEnv(num_players=4, agent_classes=None)
        env.agents = self.players  # あらかじめ構築した順序 (P0=評価対象)
        if hasattr(self.eval_agent, 'set_env_ref'):
            self.eval_agent.set_env_ref(env)
        env.reset()
        # 進行
        step_limit = 1000
        steps = 0
        prev_rankings: List[int] = list(getattr(env.game, 'rankings', []))
        # collect per-move value predictions for the eval agent
        eval_preds: List[float] = []
        while not getattr(env.game, 'done', False):
            if steps >= step_limit:
                print(f"[WARN] step limit reached ({step_limit}) forcing termination")
                break
            current_player_id = env.game.turn
            agent = env.agents[current_player_id]
            if isinstance(agent, AlphaZeroAgent):
                # get a direct value prediction for this agent (probability of 'winning'/phase)
                try:
                    _, v = agent._policy_value(env)
                    # extract scalar for this agent's seat if possible
                    v_prob = None
                    if isinstance(v, dict):
                        v_prob = float(v.get(getattr(agent, 'player_id', 0), 0.0))
                    elif isinstance(v, (list, tuple)):
                        pid = int(getattr(agent, 'player_id', 0) or 0)
                        try:
                            v_prob = float(v[pid])
                        except Exception:
                            v_prob = float(v[0]) if v else 0.0
                    else:
                        v_prob = float(v)
                    eval_preds.append(v_prob if v_prob is not None else 0.0)
                except Exception:
                    # fallback: unknown -> 0.5 (neutral)
                    try:
                        eval_preds.append(0.5)
                    except Exception:
                        pass
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
        # build labels: for now we treat the episode-level outcome as the true label
        # i.e., each collected sample in this episode is labeled positive if eval agent won the episode
        try:
            p0_seat = self.players.index(self.eval_agent)
        except Exception:
            p0_seat = 0
        won = (len(rankings) > 0 and rankings[0] == p0_seat)
        true_label = 1 if won else 0
        labels = [true_label] * len(eval_preds)

        # compute precision/recall
        precision = None
        recall = None
        auc = None
        try:
            thresh = float(getattr(self, 'value_threshold', 0.5))
            preds_pos = [1 if p > thresh else 0 for p in eval_preds]
            tp = sum(1 for p, t in zip(preds_pos, labels) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(preds_pos, labels) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(preds_pos, labels) if p == 0 and t == 1)
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            # AUC: only if both classes present
            if labels and (any(l == 1 for l in labels) and any(l == 0 for l in labels)):
                try:
                    # simple ROC AUC implementation
                    pairs = sorted(list(zip(eval_preds, labels)), key=lambda x: x[0], reverse=True)
                    P = sum(1 for _, l in pairs if l == 1)
                    N = sum(1 for _, l in pairs if l == 0)
                    if P > 0 and N > 0:
                        tp_cum = 0
                        fp_cum = 0
                        prev_tpr = 0.0
                        prev_fpr = 0.0
                        auc_acc = 0.0
                        for score, lab in pairs:
                            if lab == 1:
                                tp_cum += 1
                            else:
                                fp_cum += 1
                            tpr = tp_cum / P
                            fpr = fp_cum / N
                            auc_acc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                            prev_tpr = tpr
                            prev_fpr = fpr
                        auc = float(auc_acc)
                except Exception:
                    auc = None
        except Exception:
            precision = recall = auc = None

        # If configured, write zeros instead of empty for missing metrics
        if getattr(self, 'write_zero_on_missing_metrics', False):
            if precision is None:
                precision = 0.0
            if recall is None:
                recall = 0.0
            if auc is None:
                auc = 0.0

        return {"rankings": rankings, "steps": steps, "recall": recall, "precision": precision, "auc": auc}

    def run(self, num_episodes: int = 10, workers: int | None = None):
        # P0（評価対象: eval_agent）専用の勝率/順位集計はエピソードごとに座席が変わっても
        # エージェント実体に基づいて評価する
        p0_wins = 0
        p0_rank_sum = 0
        # 各エージェント別の集計（ラベル集約）はエピソードごとに現在の座席ラベルを使用
        wins_by_label: Dict[str, int] = {}
        games_by_label: Dict[str, int] = {}
        rank_sum_by_label: Dict[str, int] = {}
        rank_counts_by_label: Dict[str, Dict[int, int]] = {}
        # 並列実行の有無を判定
        workers = int(workers or 1)
        if workers <= 1:
            for ep in range(num_episodes):
                # エピソード用の座席順を決定し反映
                self.players = self._rotate_players_for_episode(ep)
                self._assign_seat_ids(self.players)
                cur_labels = self._labels_for_players(self.players)
                # 初回に辞書を初期化
                if ep == 0:
                    wins_by_label = {lb: 0 for lb in cur_labels}
                    games_by_label = {lb: 0 for lb in cur_labels}
                    rank_sum_by_label = {lb: 0 for lb in cur_labels}
                    rank_counts_by_label = {lb: {1: 0, 2: 0, 3: 0, 4: 0} for lb in cur_labels}
                # play_one_game now returns a dict with rankings, steps, recall
                res = self.play_one_game()
                rankings = res.get("rankings", [])
                steps = res.get("steps")
                recall = res.get("recall")
                # Elo 更新: エピソード内の現在座席に対応する固有ラベルで更新（重複不可）
                ranking_names = [cur_labels[pid] for pid in rankings]
                self.elo.update_from_rankings(ranking_names)
                # 集計 (評価対象=eval_agent 実体)。現在の座席を特定
                try:
                    p0_seat = self.players.index(self.eval_agent)
                except ValueError:
                    p0_seat = 0
                if rankings[0] == p0_seat:
                    p0_wins += 1
                p0_rank_sum += (rankings.index(p0_seat) + 1)
                avg_rank = p0_rank_sum / (ep + 1)
                win_rate = p0_wins / (ep + 1)
                # 評価対象モデルの Elo は eval_label で取得
                r = self.elo.get_rating(self.eval_label)
                # CSV 追記 (recall 列を追加)
                try:
                    recall_str = '' if recall is None else f"{recall:.6f}"
                    with open(self.metrics_csv, "a", encoding="utf-8") as f:
                        f.write(f"{ep+1},{win_rate:.6f},{avg_rank:.6f},{r:.2f},{'|'.join(map(str, rankings))},{'' if steps is None else steps},{recall_str}\n")
                except Exception:
                    pass
                # 各エージェント別の勝率/平均順位を更新（追記は評価終了後に1回のみ）
                try:
                    # rankings[0] は優勝プレイヤーID（0ベース席番号）
                    winner_pid = rankings[0] if len(rankings) > 0 else None
                    for seat, lb in enumerate(cur_labels):
                        games_by_label[lb] += 1
                        try:
                            rk = rankings.index(seat) + 1
                        except Exception:
                            rk = 4
                        rank_sum_by_label[lb] += rk
                        # 順位カウント更新
                        if rk in (1, 2, 3, 4):
                            rank_counts_by_label[lb][rk] = rank_counts_by_label[lb].get(rk, 0) + 1
                        else:
                            rank_counts_by_label[lb][4] = rank_counts_by_label[lb].get(4, 0) + 1
                        if winner_pid is not None and winner_pid == seat:
                            wins_by_label[lb] += 1
                        g = max(1, games_by_label[lb])
                        cur_wr = wins_by_label[lb] / g
                        cur_ar = rank_sum_by_label[lb] / g
                        r1 = rank_counts_by_label[lb].get(1, 0)
                        r2 = rank_counts_by_label[lb].get(2, 0)
                        r3 = rank_counts_by_label[lb].get(3, 0)
                        r4 = rank_counts_by_label[lb].get(4, 0)
                        r1_rate = r1 / g
                        r2_rate = r2 / g
                        r3_rate = r3 / g
                        r4_rate = r4 / g
                except Exception:
                    pass
                # TensorBoard 出力
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("eval/win_rate", win_rate, ep + 1)
                    self.tb_writer.add_scalar("eval/avg_rank", avg_rank, ep + 1)
                    self.tb_writer.add_scalar("eval/rating_p0", r, ep + 1)
                if (ep + 1) % 5 == 0 or ep == num_episodes - 1:
                    print(f"[EVAL] ep={ep+1}/{num_episodes} win_rate={win_rate:.2%} avg_rank={avg_rank:.2f} rating(P0)={r:.1f}")
        else:
            # 並列版: 各ワーカーでモデルを一度だけロードし、各エピソードを分散
            # 親プロセスでの巨大参照は不要なので解放する
            try:
                self._release_heavy_refs_for_parallel()
            except Exception:
                pass
            ctx = mp.get_context("spawn")
            start_ts = time.time()
            # 進捗用
            done = 0
            next_log_at = 0
            log_stride = max(1, num_episodes // 10)  # 10% ごとに表示（少数なら毎件）
            # 結果バッファ（エピソード順で後から集計）
            results_buf: List[Dict[str, Any] | None] = [None] * num_episodes
            with ctx.Pool(processes=workers, initializer=_parallel_eval_init, initargs=(
                self.checkpoint_path,
                self.device,
                self.config,
                self.baseline_mix,
                self.config.get("determinization_mode_eval"),
                self.past_checkpoints,
                None,
            )) as pool:
                for res in pool.imap_unordered(_parallel_eval_one, list(range(num_episodes)), chunksize=1):
                    try:
                        ep_idx = int(res.get("ep", done))
                    except Exception:
                        ep_idx = done
                    if 0 <= ep_idx < num_episodes:
                        results_buf[ep_idx] = res
                    else:
                        # 範囲外は最後に付け足す
                        try:
                            results_buf[results_buf.index(None)] = res
                        except Exception:
                            pass
                    done += 1
                    # 進捗ログ（10%ごと）
                    if done >= next_log_at or done == num_episodes:
                        # 簡易ETA
                        elapsed = time.time() - start_ts
                        eta_str = ""
                        try:
                            rate = elapsed / max(1, done)
                            remain = max(0.0, (num_episodes - done) * rate)
                            eta_str = f" eta~{int(remain)}s"
                        except Exception:
                            pass
                        print(f"[EVAL] progress {done}/{num_episodes}{eta_str}")
                        next_log_at = done + log_stride
                pool.close()
                pool.join()
            # 欠損があれば落ちていないものを詰める
            results = [r for r in results_buf if r is not None]
            # ep順に並べ替え（ep 欄がないものは末尾へ）
            try:
                results.sort(key=lambda x: x.get("ep", 10**9))
            except Exception:
                pass
            # ラベル集合を初期化
            if results:
                all_labels = results[0].get("labels", [])
                wins_by_label = {lb: 0 for lb in all_labels}
                games_by_label = {lb: 0 for lb in all_labels}
                rank_sum_by_label = {lb: 0 for lb in all_labels}
                rank_counts_by_label = {lb: {1: 0, 2: 0, 3: 0, 4: 0} for lb in all_labels}
            # 集計と Elo 更新
            for idx, res in enumerate(results, start=1):
                rankings = res.get("rankings", [0,1,2,3])
                cur_labels = res.get("labels", [])
                if idx == 1 and (not wins_by_label):
                    # 念のため初期化
                    wins_by_label = {lb: 0 for lb in cur_labels}
                    games_by_label = {lb: 0 for lb in cur_labels}
                    rank_sum_by_label = {lb: 0 for lb in cur_labels}
                    rank_counts_by_label = {lb: {1: 0, 2: 0, 3: 0, 4: 0} for lb in cur_labels}
                # Elo 更新
                ranking_names = [cur_labels[pid] for pid in rankings]
                self.elo.update_from_rankings(ranking_names)
                # P0実体= "AlphaZeroAgent" の席を見つける
                try:
                    p0_seat = cur_labels.index(self.eval_label)
                except ValueError:
                    p0_seat = 0
                if rankings[0] == p0_seat:
                    p0_wins += 1
                p0_rank_sum += (rankings.index(p0_seat) + 1)
                avg_rank = p0_rank_sum / idx
                win_rate = p0_wins / idx
                r = self.elo.get_rating(self.eval_label)
                # CSV 追記
                try:
                    with open(self.metrics_csv, "a", encoding="utf-8") as f:
                        recall = res.get('recall')
                        recall_str = '' if recall is None else f"{recall:.6f}"
                        f.write(f"{idx},{win_rate:.6f},{avg_rank:.6f},{r:.2f},{'|'.join(map(str, rankings))},{recall_str}\n")
                except Exception:
                    pass
                # 各エージェント別の勝率・順位
                try:
                    winner_pid = rankings[0] if len(rankings) > 0 else None
                    for seat, lb in enumerate(cur_labels):
                        games_by_label[lb] = games_by_label.get(lb, 0) + 1
                        try:
                            rk = rankings.index(seat) + 1
                        except Exception:
                            rk = 4
                        rank_sum_by_label[lb] = rank_sum_by_label.get(lb, 0) + rk
                        if lb not in rank_counts_by_label:
                            rank_counts_by_label[lb] = {1: 0, 2: 0, 3: 0, 4: 0}
                        if rk in (1,2,3,4):
                            rank_counts_by_label[lb][rk] = rank_counts_by_label[lb].get(rk, 0) + 1
                        else:
                            rank_counts_by_label[lb][4] = rank_counts_by_label[lb].get(4, 0) + 1
                        if winner_pid is not None and winner_pid == seat:
                            wins_by_label[lb] = wins_by_label.get(lb, 0) + 1
                except Exception:
                    pass
                if (idx % 5 == 0) or (idx == num_episodes):
                    print(f"[EVAL] ep={idx}/{num_episodes} win_rate={win_rate:.2%} avg_rank={avg_rank:.2f} rating(P0)={r:.1f}")
            # 並列集計はここまで（逐次ブロックの残骸を削除）
        # 評価終了後に勝率ログを1回だけ出力
        try:
            # 最終エピソード番号を episode 列に使う
            final_ep = num_episodes
            with open(self.agent_winrates_csv, "a", encoding="utf-8") as f:
                for lb in wins_by_label.keys():
                    g = max(1, games_by_label.get(lb, 0))
                    cur_wr = wins_by_label.get(lb, 0) / g
                    cur_ar = (rank_sum_by_label.get(lb, 0) / g) if g > 0 else 0.0
                    rc = rank_counts_by_label.get(lb, {})
                    r1 = rc.get(1, 0)
                    r2 = rc.get(2, 0)
                    r3 = rc.get(3, 0)
                    r4 = rc.get(4, 0)
                    r1_rate = r1 / g
                    r2_rate = r2 / g
                    r3_rate = r3 / g
                    r4_rate = r4 / g
                    f.write(
                        f"{final_ep},{lb},{cur_wr:.6f},{cur_ar:.6f},"
                        f"{r1_rate:.6f},{r2_rate:.6f},{r3_rate:.6f},{r4_rate:.6f},"
                        f"{r1},{r2},{r3},{r4},{g}\n"
                    )
        except Exception:
            pass
        # 最終リーダーボード
        board = self.elo.get_leaderboard()
        # 現行評価の参加者ラベルに限定して表示（旧席名 P0〜P3 を除去）
        allowed = set(self._agent_label_map.values())
        filtered = [(n, r) for (n, r) in board if n in allowed]
        rows = filtered if filtered else board
        print("[EVAL] Leaderboard (participants):")
        for name, rating in rows:
            print(f"  {name}: {rating:.1f}")
        # レーティング履歴を評価終了時に一括で非同期追記
        try:
            if hasattr(self.elo, 'flush_history_async'):
                self.elo.flush_history_async()
        except Exception:
            pass
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        # 戻り値: 今回の評価参加者ラベル一覧（表示/外部利用向け）
        try:
            return self.get_participant_labels()
        except Exception:
            return []

    def print_player_types(self):
        """現在のプレイヤー順序と型を表示 (デバッグ用)。"""
        for i, p in enumerate(getattr(self, 'players', [])):
            print(f"P{i}: {type(p).__name__}")

    # ---- 公開ユーティリティ ----
    def get_participant_labels(self) -> List[str]:
        """今回の評価で使っているエージェントの一意ラベル一覧（表示用）。"""
        try:
            return self._labels_for_players(self.players_base)
        except Exception:
            return [type(ag).__name__ for ag in self.players_base]

__all__ = ["Evaluator"]
