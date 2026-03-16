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
from agents.ucb_mcts_agent import UCBMCTSAgent
from agents.replay_buffer import canonicalize_state
from game.environment import DaifugoSimpleEnv
from evaluation.rating import RatingManager, EloConfig
import atexit

_EVAL_FALLBACK_LOGGED = set()


def _log_eval_fallback_once(key: str, msg: str, exc: Exception | None = None) -> None:
    if key in _EVAL_FALLBACK_LOGGED:
        return
    _EVAL_FALLBACK_LOGGED.add(key)
    try:
        if exc is not None:
            print(f"{msg} ({type(exc).__name__}: {exc})")
        else:
            print(msg)
    except Exception:
        pass

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
_EV_NUM_SIMULATIONS: int = 400

def _parallel_eval_init(checkpoint_path: str,
                        device: str,
                        base_cfg: Dict[str, Any],
                        baseline_mix: List[str],
                        det_mode_eval: str | None,
                        past_checkpoints: List[str] | None,
                        seed: int | None,
                        num_simulations: int = 400,
                        value_threshold: float = 0.5,
                        write_zero_on_missing: bool = False):
    # グローバルを書き換え
    global _EV_CFG, _EV_DEVICE, _EV_CKPT, _EV_MODEL, _EV_PAST_MODELS, _EV_PAST_CKPTS, _EV_BASELINE_MIX, _EV_SEAT_ROTATION, _EV_NUM_SIMULATIONS
    import random as _rnd
    if seed is not None:
        _rnd.seed(seed + os.getpid())
    _EV_DEVICE = device
    _EV_CKPT = checkpoint_path
    _EV_NUM_SIMULATIONS = int(num_simulations)
    # 評価時の安定設定を適用した config を構築
    cfg = dict(base_cfg)
    try:
        cfg["inference_dirichlet"] = False
        cfg["dirichlet_epsilon"] = 0.0
        cfg["temperature"] = 0.0
        cfg["opening_random_enable"] = False
        # バッチ推論サイズが設定されていない場合、デフォルト値（32）を設定
        if "mcts_batch_eval_size" not in cfg or cfg.get("mcts_batch_eval_size", 1) == 1:
            # ALPHA_ZERO_CONFIGから取得、またはデフォルト値32を使用
            from agents.config import ALPHA_ZERO_CONFIG as _AZC
            default_batch_size = _AZC.get("mcts_batch_eval_size", 32)
            cfg["mcts_batch_eval_size"] = default_batch_size
        # MCTS TTを有効化（評価時の高速化）
        cfg["enable_mcts_tt"] = True
        cfg["enable_parallel_determinization"] = True
    except Exception as e:
        _log_eval_fallback_once(
            "eval_init_config_apply_failed",
            "[eval-fallback] failed to apply inference-stable config; using base config",
            e,
        )
    if det_mode_eval is not None:
        cfg["determinization_mode_eval"] = det_mode_eval
    _EV_CFG = cfg
    _EV_BASELINE_MIX = list(baseline_mix or ["ucb_mcts", "random", "rule"])
    _EV_PAST_CKPTS = list(past_checkpoints or [])
    # num_simulations は cfg から取得（もし base_cfg に含まれていない場合はデフォルト200）
    _EV_NUM_SIMULATIONS = int(cfg.get("num_simulations", 200))
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
        _log_eval_fallback_once(
            "eval_init_main_model_load_failed",
            "[eval-fallback] failed to load main model; using None",
            e,
        )
        print(f"[WARN] eval worker failed to load main model: {e}")
        _EV_MODEL = None
    # ワーカー終了時に CUDA IPC ハンドルを回収する登録
    try:
        def _worker_ipc_cleanup():
            try:
                import torch
                if getattr(torch, 'cuda', None) is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            except Exception:
                pass

        atexit.register(_worker_ipc_cleanup)
    except Exception:
        pass
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
                _log_eval_fallback_once(
                    f"eval_init_past_model_load_failed:{p}",
                    "[eval-fallback] failed to load past model; skipping",
                    _e,
                )
                print(f"[WARN] eval worker failed to load past model '{p}': {_e}")

def _parallel_eval_one(ep_index: int) -> Dict[str, Any]:
    """1エピソードを実行し、順位とラベルを返す。"""
    from agents.drl_agent import AlphaZeroAgent as _AZ
    from agents.rule_based_agent import RuleBasedAgent as _RB
    from agents.random_agent import RandomAgent as _RA
    from agents.ucb_mcts_agent import UCBMCTSAgent as _UCB
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
    needed = 4 - len(agents)  # 必ず3になる
    
    # 過去モデルが指定されている場合は使用
    if _EV_PAST_MODELS:
        import os as _os
        for i, (ckpt_path, past_model) in enumerate(_EV_PAST_MODELS[:needed]):
            pid = len(agents)
            try:
                ag = _AZ(player_id=pid, model=past_model, config=cfg)
                try:
                    setattr(ag, "is_past_model", True)
                except Exception:
                    pass
                # ラベル用にベース名を保持
                try:
                    setattr(ag, "checkpoint_name", _os.path.basename(ckpt_path))
                except Exception:
                    pass
                agents.append(ag)
                labels.append(f"AlphaZeroAgent@{_os.path.basename(ckpt_path)}")
            except Exception as e:
                print(f"[WARN] Failed to create agent from past checkpoint '{ckpt_path}': {e}")
    
    # 過去モデルが不足する場合はbaselineエージェントで補完
    remaining = 4 - len(agents)
    if remaining > 0:
        mix = list(_EV_BASELINE_MIX or ["ucb_mcts", "random", "rule"])
        while len(mix) < remaining:
            mix.append("random")
        class_counts: Dict[str, int] = {}
        num_sims = _EV_NUM_SIMULATIONS
        for spec in mix[:remaining]:
            pid = len(agents)
            spec_lower = (spec or "").lower()
            if spec_lower == "rule":
                ag = _RB(player_id=pid)
                cname = "RuleBasedAgent"
            elif spec_lower in ("mcts", "ucb_mcts", "ucbmcts"):
                try:
                    ag = _UCB(player_id=pid, num_simulations=num_sims)
                    cname = "UCBMCTSAgent"
                except TypeError:
                    # num_simulations引数がサポートされていない場合は引数なしで再試行
                    try:
                        ag = _UCB(player_id=pid)
                        cname = "UCBMCTSAgent"
                    except Exception as e:
                        # UCBMCTSAgentの生成に失敗した場合は警告を出力してRandomAgentにフォールバック
                        _log_eval_fallback_once(
                            f"eval_ucb_init_failed:{pid}",
                            "[eval-fallback] failed to create UCBMCTSAgent; fallback to RandomAgent",
                            e,
                        )
                        print(f"[WARN] Failed to create UCBMCTSAgent for player {pid}: {e}, falling back to RandomAgent")
                        ag = _RA(player_id=pid)
                        cname = "RandomAgent"
                except Exception as e:
                    # 予期しないエラーの場合も警告を出力
                    _log_eval_fallback_once(
                        f"eval_ucb_init_exception:{pid}",
                        "[eval-fallback] unexpected UCBMCTSAgent init error; fallback to RandomAgent",
                        e,
                    )
                    print(f"[WARN] Failed to create UCBMCTSAgent for player {pid}: {e}, falling back to RandomAgent")
                    ag = _RA(player_id=pid)
                    cname = "RandomAgent"
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
    # すべての AlphaZeroAgent に env_ref を渡す（並列デタミニゼーション有効化のため）
    for _ag in env.agents:
        if hasattr(_ag, 'set_env_ref'):
            try:
                _ag.set_env_ref(env)
            except Exception:
                pass
    env.reset()
    # デタミニゼーションプールを事前に起動し、warmup待機
    import time as _tpool
    for _ag in env.agents:
        if hasattr(_ag, '_maybe_start_det_pool') and hasattr(_ag, 'config'):
            try:
                if _ag.config.get('enable_parallel_determinization') and _ag.config.get('enable_determinization'):
                    _ag._maybe_start_det_pool(env)
                    # プールが十分に満たされるまで待機（最大2秒）
                    _wait_start = _tpool.time()
                    while (_tpool.time() - _wait_start) < 2.0:
                        with getattr(_ag, '_det_pool_lock', None) or _tpool:
                            if hasattr(_ag, '_det_pool') and _ag._det_pool and len(_ag._det_pool) >= 5:
                                break
                        _tpool.sleep(0.01)
            except Exception:
                pass
    step_limit = 1000
    steps = 0
    eval_preds: List[float] = []
    # 手札予測データ収集用
    hand_pred_data_list: List[Dict[str, Any]] = []
    # 初回ステップでエージェント使用ログ出力
    _agent_usage_logged = False
    # 時間計測用
    import time
    timings = {
        'alpha_zero_select': 0.0,
        'ucb_mcts_select': 0.0,
        'other_agent_select': 0.0,
        'env_step': 0.0,
        'hand_pred_collection': 0.0,
        'total': 0.0
    }
    az_action_count = 0
    ucb_action_count = 0
    other_action_count = 0
    while not getattr(env.game, 'done', False):
        if steps >= step_limit:
            break
        current_player_id = env.game.turn
        agent = env.agents[current_player_id]
        # 初回のみエージェント使用ログ出力
        if not _agent_usage_logged and steps == 0:
            agent_type = type(agent).__name__
            agent_label = labels[current_player_id] if current_player_id < len(labels) else "Unknown"
            print(f"[EVAL] Episode {ep_index}: First action by Player {current_player_id} using {agent_label} (type: {agent_type})")
            _agent_usage_logged = True
        
        step_start = time.time()
        
        if isinstance(agent, _AZ):
            # 手札予測評価のため、推論結果を取得
            az_start = time.time()
            import torch
            with torch.inference_mode():
                # Optional debug: compare raw vs canonicalized state per-seat and model top-policies
                try:
                    if os.environ.get('EVAL_DEBUG_CANON'):
                        import numpy as _np
                        st_dbg = agent._extract_state(env)
                        st_can_dbg = canonicalize_state(agent, st_dbg)
                        print(f"[DEBUG-EVAL] ep={ep_index} seat={getattr(agent,'player_id',None)} agent={type(agent).__name__} state_keys={list(st_dbg.keys()) if isinstance(st_dbg,dict) else type(st_dbg)} canon_self={st_can_dbg.get('self_player_id')}")
                        try:
                            for k in ('full_input','hand_labels'):
                                if isinstance(st_dbg, dict) and k in st_dbg and k in st_can_dbg:
                                    a0 = _np.asarray(st_dbg[k]).ravel()[:8]
                                    a1 = _np.asarray(st_can_dbg[k]).ravel()[:8]
                                    diff = (_np.abs(a0 - a1) > 1e-6).any()
                                    print(f"[DEBUG-EVAL] seat={getattr(agent,'player_id',None)} field={k} differs={diff} before={a0.tolist()} after={a1.tolist()}")
                        except Exception:
                            pass
                        # model top-k (if available)
                        if getattr(agent, 'model', None) is not None and hasattr(agent.model, 'forward_with_belief'):
                            try:
                                out = agent.model.forward_with_belief(st_can_dbg)
                                # try to extract policy logits
                                pol = None
                                if isinstance(out, tuple) or isinstance(out, list):
                                    pol = out[0]
                                elif isinstance(out, dict) and 'policy_logits' in out:
                                    pol = out['policy_logits']
                                elif isinstance(out, dict) and 'pi' in out:
                                    pol = out['pi']
                                if pol is not None:
                                    pol_arr = _np.asarray(pol).ravel()
                                    idx = _np.argsort(-pol_arr)[:5].tolist()
                                    vals = pol_arr[idx].tolist()
                                    print(f"[DEBUG-EVAL] seat={getattr(agent,'player_id',None)} top_pi_idx={idx} top_pi_vals={vals}")
                            except Exception as _e:
                                print(f"[DEBUG-EVAL] model forward/top-pi failed: {_e}")
                except Exception:
                    pass
                # forward_with_beliefで推論結果を取得
                try:
                    # ensure inference-time state is canonical (self==0 viewpoint)
                    state_dict = agent._extract_state(env)
                    # If this agent is a past checkpoint, skip canonicalization to preserve its original view
                    if not getattr(agent, 'is_past_model', False):
                        state_dict = canonicalize_state(agent, state_dict)
                    if agent.model is not None and hasattr(agent.model, 'forward_with_belief'):
                        policy_logits, value_logit, hand_logits = agent.model.forward_with_belief(state_dict)
                    else:
                        hand_logits = None
                    # hand_logitsから手札予測データを収集（計算はメインプロセスで行う）
                    if hand_logits is not None and 'hand_labels' in state_dict:
                        hand_labels = state_dict['hand_labels']
                        mask_unknown = state_dict.get('mask_unknown', None)
                        # 相手の手札枚数を取得
                        opponent_hand_sizes = {}
                        for i in range(4):
                            if i != current_player_id:
                                try:
                                    opponent_hand_sizes[i] = len(env.game.players[i].hand)
                                except Exception:
                                    opponent_hand_sizes[i] = 0
                        # データを収集（Evaluatorメソッドをここで呼べないため）
                        import numpy as np
                        hand_logits_np = hand_logits.cpu().numpy() if isinstance(hand_logits, torch.Tensor) else np.array(hand_logits)
                        hand_pred_data_list.append({
                            'step': steps,
                            'hand_logits': hand_logits_np.tolist(),
                            'hand_labels': hand_labels,
                            'mask_unknown': mask_unknown,
                            'opponent_hand_sizes': opponent_hand_sizes
                        })
                except Exception as e:
                    print(f"[WARN] Hand prediction data collection failed: {e}")

                action = agent.select_action(env, training=False)
            timings['alpha_zero_select'] += time.time() - az_start
            az_action_count += 1
        elif isinstance(agent, _UCB):
            # UCBMCTSAgent: 環境オブジェクトを渡す
            ucb_start = time.time()
            import torch
            with torch.inference_mode():
                action = agent.select_action(env, legal_actions=None)
            timings['ucb_mcts_select'] += time.time() - ucb_start
            ucb_action_count += 1
        else:
            other_start = time.time()
            current_player = env.game.players[current_player_id]
            hand = current_player.hand
            field = env.game.current_field[:]
            legal_actions = env._generate_legal_actions(hand, field)
            obs_simple = {'hand': hand, 'field': field}
            action = agent.select_action(obs_simple, legal_actions=legal_actions)
            timings['other_agent_select'] += time.time() - other_start
            other_action_count += 1
        
        step_time = time.time() - step_start
        timings['total'] += step_time
        
        env_start = time.time()
        try:
            env.step(external_action=action)
        except TypeError:
            env.step(action)
        timings['env_step'] += time.time() - env_start
        steps += 1
    rankings: List[int] = list(getattr(env.game, 'rankings', []))
    if len(rankings) != 4:
        remaining = [i for i in range(4) if i not in rankings]
        rankings += remaining
    
    # 時間計測結果をログ出力（ゲーム終了時）
    if steps > 0:
        avg_step_time = timings['total'] / steps
        print(f"[PERF] Episode {ep_index} timing summary:")
        print(f"  Total time: {timings['total']:.2f}s ({steps} steps, avg {avg_step_time:.3f}s/step)")
        if az_action_count > 0:
            avg_az_time = timings['alpha_zero_select'] / az_action_count
            print(f"  AlphaZeroAgent: {timings['alpha_zero_select']:.2f}s ({az_action_count} actions, avg {avg_az_time:.3f}s/action)")
        if ucb_action_count > 0:
            avg_ucb_time = timings['ucb_mcts_select'] / ucb_action_count
            print(f"  UCBMCTSAgent: {timings['ucb_mcts_select']:.2f}s ({ucb_action_count} actions, avg {avg_ucb_time:.3f}s/action)")
        if other_action_count > 0:
            avg_other_time = timings['other_agent_select'] / other_action_count
            print(f"  Other agents: {timings['other_agent_select']:.2f}s ({other_action_count} actions, avg {avg_other_time:.3f}s/action)")
        print(f"  Env step: {timings['env_step']:.2f}s")
    
    # ラベル順（席番号順）を返し、親で Elo を更新できるようにする
    return {
        "ep": ep_index, 
        "rankings": rankings, 
        "labels": labels, 
        "steps": steps,
        "hand_pred_data": hand_pred_data_list,
        "timings": timings
    }


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
        determinization_mode_override: str | None = "stochastic",
        past_checkpoints: List[str] | None = None,
        # 閾値: value 予測を陽性と見なすカットオフ
        value_threshold: float = 0.5,
        # 指標が計算できないときに 0.0 を出力するか (False -> 空欄)
        write_zero_on_missing_metrics: bool = False,
        # 圧倒的無駄遣いの禁止（Dominated Moves）フィルタリング
        filter_dominated_moves: bool = False,
    ):
        if seed is not None:
            random.seed(seed)
        self.checkpoint_path = checkpoint_path
        self.device = device or self._auto_device()
        self.elo_dir = elo_dir
        self.elo = RatingManager(save_dir=elo_dir, config=EloConfig())
        # baseline_mix 例: ["ucb_mcts", "random", "rule"] -> 学習エージェント + 3 baseline
        # 既定でUCBMCTSAgent、ランダム、ルールベースを含める
        self.baseline_mix = baseline_mix or ["ucb_mcts", "random", "rule"]
        # 過去モデルのチェックポイント群（最大3枠まで採用）
        self.past_checkpoints = list(past_checkpoints or [])
        self.config = dict(ALPHA_ZERO_CONFIG)
        # num_simulationsはconfigの値を使用（上書きしない）
        self.config["device"] = self.device
        # num_simulationsはconfigから取得（後方互換性のためself.num_simulationsも設定）
        self.num_simulations = self.config.get("num_simulations", 400)
        # グローバル変数も更新（並列処理用）
        global _EV_NUM_SIMULATIONS
        _EV_NUM_SIMULATIONS = self.num_simulations
        # 評価でも並列デタミニゼーションプールを有効化（速度改善・ログ抑制）
        # バッチ推論サイズを明示的に設定（評価時の高速化）
        if isinstance(self.config, dict):
            self.config["enable_mcts_tt"] = True
            self.config["enable_parallel_determinization"] = True
            # バッチ推論サイズが設定されていない場合、デフォルト値（32）を設定
            if "mcts_batch_eval_size" not in self.config or self.config.get("mcts_batch_eval_size", 1) == 1:
                # ALPHA_ZERO_CONFIGから取得、またはデフォルト値32を使用
                default_batch_size = ALPHA_ZERO_CONFIG.get("mcts_batch_eval_size", 32)
                self.config["mcts_batch_eval_size"] = default_batch_size
                print(f"[EVAL] mcts_batch_eval_size set to {default_batch_size} for evaluation")
            else:
                print(f"[EVAL] mcts_batch_eval_size: {self.config.get('mcts_batch_eval_size')}")
        else:
            try:
                setattr(self.config, "enable_mcts_tt", True)
                setattr(self.config, "enable_parallel_determinization", True)
                # バッチ推論サイズも設定を試みる
                if not hasattr(self.config, "mcts_batch_eval_size") or getattr(self.config, "mcts_batch_eval_size", 1) == 1:
                    default_batch_size = ALPHA_ZERO_CONFIG.get("mcts_batch_eval_size", 32)
                    setattr(self.config, "mcts_batch_eval_size", default_batch_size)
            except (AttributeError, TypeError) as e:
                _log_eval_fallback_once(
                    "eval_set_config_attr_failed",
                    "[eval-fallback] could not set MCTS config attributes; using defaults",
                    e,
                )
                import warnings
                warnings.warn(f"Could not set MCTS config (type: {type(self.config)}), using defaults")
        # 評価フェーズでは探索ノイズOFF・温度0・序盤ランダム無効化を徹底
        # （AlphaZeroAgent.select_action(training=False) でも低温/ノイズ無効になるが、明示的に設定）
        # config が辞書型でない場合や読み取り専用の場合は KeyError/TypeError が発生する可能性がある
        if isinstance(self.config, dict):
            self.config["inference_dirichlet"] = False  # 推論時のルートDirichletを無効化
            self.config["dirichlet_epsilon"] = 0.0      # 念のため係数も0に
            self.config["temperature"] = 0.0            # 温度0（_select_temperatureで非学習時は1e-6だが整合のため）
            self.config["opening_random_enable"] = False
        else:
            # config が辞書型でない場合は setattr を試行（読み取り専用プロパティの可能性）
            try:
                setattr(self.config, "inference_dirichlet", False)
                setattr(self.config, "dirichlet_epsilon", 0.0)
                setattr(self.config, "temperature", 0.0)
                setattr(self.config, "opening_random_enable", False)
            except (AttributeError, TypeError) as e:
                _log_eval_fallback_once(
                    "eval_set_eval_config_failed",
                    "[eval-fallback] could not set evaluation config; using defaults",
                    e,
                )
                # 設定できない場合は警告を出力して続行
                import warnings
                warnings.warn(f"Could not set evaluation config (type: {type(self.config)}), using defaults")
        # 評価時の determinization モードを上書き（デフォルトで fixed_once）。
        if determinization_mode_override is not None:
            self.config["determinization_mode_eval"] = determinization_mode_override
        # メトリクス CSV 設定
        self.metrics_csv = metrics_csv or os.path.join(elo_dir, "eval_metrics.csv")
        os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
        if not os.path.exists(self.metrics_csv):
            try:
                with open(self.metrics_csv, "w", encoding="utf-8") as f:
                    f.write("episode,win_rate,avg_rank,rating_p0,raw_rank,steps\n")
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
                _log_eval_fallback_once(
                    "eval_checkpoint_load_failed",
                    "[eval-fallback] checkpoint load failed; fallback to new model",
                    e,
                )
                print(f"[WARN] checkpoint load failed ({e}) -> fallback new model")
                full_dim = 56 * self.config["num_players"] + 22
                self.model = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device=self.device,
                    full_feature_dim=full_dim,
                    context_out_dim=self.config.get("hidden_size", 128),  # hidden_sizeに合わせる（過学習防止）
                )
        else:
            _log_eval_fallback_once(
                "eval_checkpoint_missing",
                "[eval-fallback] checkpoint not found; using random initialized model",
            )
            print(f"[WARN] checkpoint not found: {self.checkpoint_path}. Using random initialized model.")
            full_dim = 56 * self.config["num_players"] + 22
            self.model = PolicyValueNet(
                max_policy_size=self.config["max_policy_size"],
                hidden_size=self.config["hidden_size"],
                num_players=self.config["num_players"],
                device=self.device,
                full_feature_dim=full_dim,
                context_out_dim=self.config.get("hidden_size", 128),  # hidden_sizeに合わせる（過学習防止）
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
        # 指定文字列に応じてベースラインエージェントを生成
        s = (spec or "").lower()
        if s in ("mcts", "ucb_mcts", "ucbmcts"):
            try:
                # UCBMCTSAgent のコンストラクタはオプション引数を受け取る可能性があるため柔軟に対応
                # UCBMCTSAgentは独自の探索回数200回を使用
                return UCBMCTSAgent(player_id=player_id, num_simulations=200)
            except TypeError:
                try:
                    return UCBMCTSAgent(player_id=player_id)
                except Exception as e:
                    _log_eval_fallback_once(
                        f"eval_make_baseline_ucb_failed:{player_id}",
                        "[eval-fallback] UCBMCTSAgent init failed; fallback to RandomAgent",
                        e,
                    )
                    return RandomAgent(player_id=player_id)
            except Exception as e:
                _log_eval_fallback_once(
                    f"eval_make_baseline_ucb_exception:{player_id}",
                    "[eval-fallback] UCBMCTSAgent unexpected error; fallback to RandomAgent",
                    e,
                )
                return RandomAgent(player_id=player_id)
        if s == "rule":
            return RuleBasedAgent(player_id=player_id)
        # default: random
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
            _log_eval_fallback_once(
                f"eval_past_ckpt_load_failed:{ckpt_path}",
                "[eval-fallback] failed to load past checkpoint; using None model",
                e,
            )
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
        needed = 4 - len(agents)  # 必ず3になる
        
        # 過去モデルが指定されている場合は使用
        if self.past_checkpoints:
            for i, ckpt_path in enumerate(self.past_checkpoints[:needed]):
                pid = len(agents)
                ag = self._make_alpha_zero_eval_agent_from_ckpt(pid, ckpt_path)
                agents.append(ag)
        
        # 過去モデルが不足する場合はbaselineエージェントで補完
        remaining = 4 - len(agents)
        if remaining > 0:
            mix = list(self.baseline_mix)
            while len(mix) < remaining:
                mix.append("random")
            for spec in mix[:remaining]:
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

    def _calculate_hand_prediction_accuracy(
        self,
        hand_logits: Any,  # torch.Tensor or np.ndarray
        hand_labels: Any,  # np.ndarray
        opponent_hand_sizes: Dict[int, int],  # 座席ID -> 残り手札枚数
        mask_unknown: Any,  # np.ndarray
        num_players: int = 4,
        apply_constraint: bool = True,   # デフォルトで制約なし（学習時と同条件で公平な比較）
    ) -> List[Dict[str, Any]]:
        """
        各相手プレイヤーごとにTop-HandSize Accuracyを計算
        
        hand_logitsの構造:
        - [0:53]: 相手プレイヤー0（自分の次のプレイヤー）の53次元
        - [53:106]: 相手プレイヤー1の53次元
        - [106:159]: 相手プレイヤー2の53次元（4人対戦の場合）
        
        Args:
            apply_constraint: True の場合、カード枚数制約を適用して予測確率を正規化
        
        Returns:
            List of dicts, each containing:
            {
                'opponent_seat_id': int,  # 座席ID（0-2）
                'opponent_hand_size': int,  # 残り手札枚数
                'ai_accuracy': float,       # Top-HandSize Accuracy
                'baseline_accuracy': float, # ランダムベースライン精度（理論値）
                'relative_improvement': float
            }
        """
        import numpy as np
        import torch
        
        results = []
        
        try:
            # hand_logitsをnumpy配列に変換
            if isinstance(hand_logits, torch.Tensor):
                hand_logits_np = hand_logits.detach().cpu().numpy()
            else:
                hand_logits_np = np.asarray(hand_logits, dtype=np.float32)
            
            # sigmoidで確率に変換
            hand_probs = 1.0 / (1.0 + np.exp(-hand_logits_np))
            
            # カード枚数制約を適用（各カードについて全相手の予測確率の合計が4を超えないように正規化）
            if apply_constraint:
                try:
                    from agents.validation import apply_hand_prediction_constraint
                    hand_probs = apply_hand_prediction_constraint(hand_probs, num_opponents=num_players-1)
                except Exception as e:
                    print(f"[WARN] Failed to apply hand prediction constraint: {e}")
            
            # hand_labelsとmask_unknownをnumpy配列に変換
            if isinstance(hand_labels, torch.Tensor):
                hand_labels_np = hand_labels.detach().cpu().numpy()
            else:
                hand_labels_np = np.asarray(hand_labels, dtype=np.float32)
            
            if isinstance(mask_unknown, torch.Tensor):
                mask_unknown_np = mask_unknown.detach().cpu().numpy()
            elif mask_unknown is not None:
                mask_unknown_np = np.asarray(mask_unknown, dtype=np.float32)
            else:
                mask_unknown_np = None
            
            # mask_unknown_npが0次元配列（スカラー）の場合はスキップ
            if mask_unknown_np is not None and mask_unknown_np.ndim == 0:
                mask_unknown_np = None
            
            # 各相手プレイヤーごとに処理
            num_opponents = num_players - 1
            for opp_idx in range(num_opponents):
                offset = opp_idx * 53
                opp_probs = hand_probs[offset:offset+53]
                opp_labels = hand_labels_np[offset:offset+53]
                
                # mask_unknownが有効な場合は使用、そうでなければ全て未知（全1）として扱う
                if mask_unknown_np is not None and mask_unknown_np.ndim > 0:
                    opp_mask = mask_unknown_np[offset:offset+53]
                else:
                    # mask_unknownが無効な場合は全て未知として扱う
                    opp_mask = np.ones(53, dtype=np.float32)
                
                # 座席IDを取得（eval_agentの次のプレイヤーから順に0, 1, 2）
                # 実際の座席IDは、eval_agentの座席ID + opp_idx + 1 (mod 4)
                try:
                    eval_seat = self.players.index(self.eval_agent)
                    opponent_seat_id = (eval_seat + opp_idx + 1) % num_players
                except Exception:
                    opponent_seat_id = opp_idx
                
                # 相手の残り手札枚数を取得
                opponent_hand_size = opponent_hand_sizes.get(opponent_seat_id, 0)
                
                # 手札0枚の場合は評価対象外
                if opponent_hand_size <= 0:
                    continue
                
                # 未知カードのインデックスを取得
                unknown_indices = np.where(opp_mask > 0.5)[0]
                if len(unknown_indices) == 0:
                    continue
                
                # 未知カード数
                N = len(unknown_indices)
                k = opponent_hand_size
                
                if k > N:
                    # 手札枚数が未知カード数を超える場合はスキップ（通常は発生しない）
                    continue
                
                # Top-HandSize Accuracy: 未知カードの中で予測確率上位k枚を選択
                unknown_probs = opp_probs[unknown_indices]
                top_k_indices_in_unknown = np.argsort(unknown_probs)[-k:][::-1]  # 降順
                top_k_card_indices = unknown_indices[top_k_indices_in_unknown]
                
                # 選択したk枚が実際に相手が持っているか確認
                correct_count = np.sum(opp_labels[top_k_card_indices] > 0.5)
                ai_accuracy = float(correct_count / k) if k > 0 else 0.0
                
                # ベースライン精度の計算
                if apply_constraint:
                    # 制約適用時: この相手について、制約を考慮せずランダムに選んだ場合の期待精度
                    # （他の相手の予測を考慮しないナイーブなランダム選択）
                    # 制約により予測確率が正規化されているため、単純なk/Nとの比較は不公平
                    # ここでは「制約なしのランダム選択」として k/N を使用（保守的な下限）
                    baseline_accuracy = float(k / N) if N > 0 else 0.0
                else:
                    # 制約なし: 単純な k / N
                    baseline_accuracy = float(k / N) if N > 0 else 0.0
                
                # 相対改善度
                relative_improvement = ai_accuracy - baseline_accuracy
                
                results.append({
                    'opponent_seat_id': opponent_seat_id,
                    'opponent_hand_size': opponent_hand_size,
                    'ai_accuracy': ai_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'relative_improvement': relative_improvement
                })
        except Exception as e:
            # エラーが発生した場合は空のリストを返す
            print(f"[WARN] _calculate_hand_prediction_accuracy failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return results

    def play_one_game(self) -> Dict[str, Any]:
        env = DaifugoSimpleEnv(num_players=4, agent_classes=None)
        env.agents = self.players  # あらかじめ構築した順序 (P0=評価対象)
        # すべての AlphaZeroAgent に env_ref を渡す（並列デタミニゼーション有効化のため）
        for _ag in env.agents:
            if hasattr(_ag, 'set_env_ref'):
                try:
                    _ag.set_env_ref(env)
                except Exception:
                    pass
        # エージェント使用ログ出力
        labels = self._labels_for_players(self.players)
        print(f"[EVAL] Game started: Agents assigned:")
        for i, (ag, label) in enumerate(zip(self.players, labels)):
            agent_type = type(ag).__name__
            agent_id = getattr(ag, 'player_id', i)
            print(f"  Player {agent_id}: {label} (type: {agent_type})")
        env.reset()
        # デタミニゼーションプールを事前に起動し、warmup待機
        import time as _tpool
        for _ag in self.players:
            if hasattr(_ag, '_maybe_start_det_pool') and hasattr(_ag, 'config'):
                try:
                    if _ag.config.get('enable_parallel_determinization') and _ag.config.get('enable_determinization'):
                        _ag._maybe_start_det_pool(env)
                        # プールが十分に満たされるまで待機（最大2秒）
                        _wait_start = _tpool.time()
                        while (_tpool.time() - _wait_start) < 2.0:
                            with getattr(_ag, '_det_pool_lock', None) or _tpool:
                                if hasattr(_ag, '_det_pool') and _ag._det_pool and len(_ag._det_pool) >= 5:
                                    break
                            _tpool.sleep(0.01)
                except Exception:
                    pass
        # 進行
        step_limit = 1000
        steps = 0
        prev_rankings: List[int] = list(getattr(env.game, 'rankings', []))
        # collect per-move value predictions for the eval agent
        eval_preds: List[float] = []
        # collect hand prediction data for the eval agent
        hand_pred_data_list: List[Dict[str, Any]] = []
        # 初回ステップでエージェント使用ログ出力
        _agent_usage_logged = False
        # 時間計測用
        import time
        timings = {
            'alpha_zero_select': 0.0,
            'ucb_mcts_select': 0.0,
            'other_agent_select': 0.0,
            'env_step': 0.0,
            'hand_pred_collection': 0.0,
            'total': 0.0
        }
        az_action_count = 0
        ucb_action_count = 0
        other_action_count = 0
        while not getattr(env.game, 'done', False):
            if steps >= step_limit:
                print(f"[WARN] step limit reached ({step_limit}) forcing termination")
                break
            current_player_id = env.game.turn
            agent = env.agents[current_player_id]
            # 初回のみエージェント使用ログ出力
            if not _agent_usage_logged and steps == 0:
                agent_type = type(agent).__name__
                agent_label = labels[current_player_id] if current_player_id < len(labels) else "Unknown"
                print(f"[EVAL] First action by Player {current_player_id} using {agent_label} (type: {agent_type})")
                _agent_usage_logged = True
            
            step_start = time.time()
            
            if isinstance(agent, AlphaZeroAgent):
                # 手札予測評価のため、推論結果を取得
                az_start = time.time()
                import torch
                with torch.inference_mode():
                    # Optional debug (see EVAL_DEBUG_CANON env var)
                    try:
                        if os.environ.get('EVAL_DEBUG_CANON'):
                            import numpy as _np
                            s_raw = agent._extract_state(env)
                            s_can = canonicalize_state(agent, s_raw)
                            print(f"[DEBUG-EVAL] seat={getattr(agent,'player_id',None)} keys={list(s_raw.keys()) if isinstance(s_raw,dict) else type(s_raw)} canon_self={s_can.get('self_player_id')}")
                            try:
                                for k in ('full_input','hand_labels'):
                                    if isinstance(s_raw, dict) and k in s_raw and k in s_can:
                                        a0 = _np.asarray(s_raw[k]).ravel()[:8]
                                        a1 = _np.asarray(s_can[k]).ravel()[:8]
                                        diff = (_np.abs(a0 - a1) > 1e-6).any()
                                        print(f"[DEBUG-EVAL] seat={getattr(agent,'player_id',None)} field={k} differs={diff} before={a0.tolist()} after={a1.tolist()}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # forward_with_beliefで推論結果を取得
                    try:
                        # ensure inference-time state is canonical
                        state_dict = agent._extract_state(env)
                        # For past models we want to skip canonicalization (preserve original viewpoint)
                        if not getattr(agent, 'is_past_model', False):
                            state_dict = canonicalize_state(agent, state_dict)
                        if agent.model is not None and hasattr(agent.model, 'forward_with_belief'):
                            policy_logits, value_logit, hand_logits = agent.model.forward_with_belief(state_dict)
                        else:
                            hand_logits = None
                        # hand_logitsから手札予測精度を計算
                        if hand_logits is not None and 'hand_labels' in state_dict:
                            hand_labels = state_dict['hand_labels']
                            mask_unknown = state_dict.get('mask_unknown', None)
                            # 相手の手札枚数を取得
                            opponent_hand_sizes = {}
                            for i in range(4):
                                if i != current_player_id:
                                    try:
                                        opponent_hand_sizes[i] = len(env.game.players[i].hand)
                                    except Exception:
                                        opponent_hand_sizes[i] = 0
                            # 精度計算
                            hand_pred_results = self._calculate_hand_prediction_accuracy(
                                hand_logits, hand_labels, opponent_hand_sizes, mask_unknown, num_players=4
                            )
                            # ターン番号を追加して収集
                            for result in hand_pred_results:
                                result['turn'] = steps
                                hand_pred_data_list.append(result)
                    except Exception as e:
                        print(f"[WARN] Hand prediction evaluation failed: {e}")
                    
                    action = agent.select_action(env, training=False)
                timings['alpha_zero_select'] += time.time() - az_start
                az_action_count += 1
            elif isinstance(agent, UCBMCTSAgent):
                # UCBMCTSAgent: 環境オブジェクトを渡す
                ucb_start = time.time()
                import torch
                with torch.inference_mode():
                    action = agent.select_action(env, legal_actions=None)
                timings['ucb_mcts_select'] += time.time() - ucb_start
                ucb_action_count += 1
            else:
                # baseline: シンプル観測
                other_start = time.time()
                current_player = env.game.players[current_player_id]
                hand = current_player.hand
                field = env.game.current_field[:]
                legal_actions = env._generate_legal_actions(hand, field)
                obs_simple = {'hand': hand, 'field': field}
                action = agent.select_action(obs_simple, legal_actions=legal_actions)
                timings['other_agent_select'] += time.time() - other_start
                other_action_count += 1
            
            step_time = time.time() - step_start
            timings['total'] += step_time
            
            env_start = time.time()
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            timings['env_step'] += time.time() - env_start
            steps += 1
        rankings: List[int] = list(getattr(env.game, 'rankings', []))
        if len(rankings) != 4:
            # 強制終了時など順位未確定は残りをランダム末尾扱い
            remaining = [i for i in range(4) if i not in rankings]
            rankings += remaining
        
        # 時間計測結果をログ出力（ゲーム終了時）
        if steps > 0:
            avg_step_time = timings['total'] / steps
            print(f"[PERF] Game timing summary:")
            print(f"  Total time: {timings['total']:.2f}s ({steps} steps, avg {avg_step_time:.3f}s/step)")
            if az_action_count > 0:
                avg_az_time = timings['alpha_zero_select'] / az_action_count
                print(f"  AlphaZeroAgent: {timings['alpha_zero_select']:.2f}s ({az_action_count} actions, avg {avg_az_time:.3f}s/action)")
            if ucb_action_count > 0:
                avg_ucb_time = timings['ucb_mcts_select'] / ucb_action_count
                print(f"  UCBMCTSAgent: {timings['ucb_mcts_select']:.2f}s ({ucb_action_count} actions, avg {avg_ucb_time:.3f}s/action)")
            if other_action_count > 0:
                avg_other_time = timings['other_agent_select'] / other_action_count
                print(f"  Other agents: {timings['other_agent_select']:.2f}s ({other_action_count} actions, avg {avg_other_time:.3f}s/action)")
            print(f"  Env step: {timings['env_step']:.2f}s")

        return {
            "rankings": rankings, 
            "steps": steps,
            "hand_pred_data": hand_pred_data_list
        }

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
        # 手札予測データの集計用
        all_hand_pred_data: List[Dict[str, Any]] = []
        
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
                # play_one_game now returns a dict with rankings, steps, hand_pred_data
                res = self.play_one_game()
                rankings = res.get("rankings", [])
                steps = res.get("steps")
                hand_pred_data = res.get("hand_pred_data", [])
                
                # 手札予測データを収集（game_idを追加）
                for data in hand_pred_data:
                    data['game_id'] = ep
                    all_hand_pred_data.append(data)
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
                # CSV 追記
                try:
                    with open(self.metrics_csv, "a", encoding="utf-8") as f:
                        f.write(f"{ep+1},{win_rate:.6f},{avg_rank:.6f},{r:.2f},{'|'.join(map(str, rankings))},{'' if steps is None else steps}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to write to eval_metrics.csv: {e}")
                    import traceback
                    traceback.print_exc()
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
                self.num_simulations,
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
            print(f"[DEBUG] Collected {len(results)} results from {num_episodes} episodes")
            # ep順に並べ替え（ep 欄がないものは末尾へ）
            try:
                results.sort(key=lambda x: x.get("ep", 10**9))
            except Exception:
                pass
            # ラベル集合を初期化
            if results:
                print(f"[DEBUG] Processing {len(results)} results for metrics")
                all_labels = results[0].get("labels", [])
                wins_by_label = {lb: 0 for lb in all_labels}
                games_by_label = {lb: 0 for lb in all_labels}
                rank_sum_by_label = {lb: 0 for lb in all_labels}
                rank_counts_by_label = {lb: {1: 0, 2: 0, 3: 0, 4: 0} for lb in all_labels}
            else:
                print(f"[ERROR] No results collected from parallel execution!")
            # 集計と Elo 更新
            for idx, res in enumerate(results, start=1):
                rankings = res.get("rankings", [0,1,2,3])
                cur_labels = res.get("labels", [])
                hand_pred_data = res.get("hand_pred_data", [])
                steps = res.get("steps", None)
                
                # 手札予測データを処理して精度計算結果に変換
                for data in hand_pred_data:
                    try:
                        hand_logits = data.get('hand_logits', [])
                        hand_labels = data.get('hand_labels', [])
                        mask_unknown = data.get('mask_unknown', None)
                        opponent_hand_sizes = data.get('opponent_hand_sizes', {})
                        step = data.get('step', 0)
                        game_id = res.get("ep", idx - 1)
                        
                        # 精度計算
                        hand_pred_results = self._calculate_hand_prediction_accuracy(
                            hand_logits, hand_labels, opponent_hand_sizes, mask_unknown, num_players=4
                        )
                        # game_idとturnを追加して収集
                        for result in hand_pred_results:
                            result['game_id'] = game_id
                            result['turn'] = step
                            all_hand_pred_data.append(result)
                    except Exception as e:
                        print(f"[WARN] Failed to calculate hand prediction accuracy for ep {res.get('ep', idx-1)}: {e}")
                
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
                        f.write(f"{idx},{win_rate:.6f},{avg_rank:.6f},{r:.2f},{'|'.join(map(str, rankings))},{'' if steps is None else steps}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to write to eval_metrics.csv (parallel): {e}")
                    import traceback
                    traceback.print_exc()
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
        # 親プロセス側でも子プロセス終了後に CUDA IPC ハンドルを回収
        try:
            import torch
            if getattr(torch, 'cuda', None) is not None and torch.cuda.is_available():
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

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
        
        # 手札予測データのCSV出力と集計
        if all_hand_pred_data:
            try:
                import csv
                hand_pred_csv_path = os.path.join(self.elo_dir, "hand_prediction_evaluation.csv")
                os.makedirs(os.path.dirname(hand_pred_csv_path), exist_ok=True)
                
                # CSVヘッダー
                fieldnames = ['game_id', 'turn', 'opponent_seat_id', 'opponent_hand_size', 
                             'ai_accuracy', 'baseline_accuracy', 'relative_improvement']
                
                # CSVに書き込み
                with open(hand_pred_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for data in all_hand_pred_data:
                        writer.writerow({
                            'game_id': data.get('game_id', 0),
                            'turn': data.get('turn', 0),
                            'opponent_seat_id': data.get('opponent_seat_id', 0),
                            'opponent_hand_size': data.get('opponent_hand_size', 0),
                            'ai_accuracy': f"{data.get('ai_accuracy', 0.0):.6f}",
                            'baseline_accuracy': f"{data.get('baseline_accuracy', 0.0):.6f}",
                            'relative_improvement': f"{data.get('relative_improvement', 0.0):.6f}"
                        })
                
                # 集計結果を計算
                import numpy as np
                relative_improvements = [d.get('relative_improvement', 0.0) for d in all_hand_pred_data]
                avg_relative_improvement = np.mean(relative_improvements) if relative_improvements else 0.0
                
                # 残り手札枚数ごとの平均精度を計算
                hand_size_stats: Dict[int, Dict[str, List[float]]] = {}
                for data in all_hand_pred_data:
                    hand_size = data.get('opponent_hand_size', 0)
                    if hand_size > 0:
                        if hand_size not in hand_size_stats:
                            hand_size_stats[hand_size] = {'ai': [], 'baseline': []}
                        hand_size_stats[hand_size]['ai'].append(data.get('ai_accuracy', 0.0))
                        hand_size_stats[hand_size]['baseline'].append(data.get('baseline_accuracy', 0.0))
                
                # 集計結果を出力
                print(f"\n[EVAL] Hand Prediction Evaluation Summary:")
                print(f"  Total samples: {len(all_hand_pred_data)}")
                print(f"  Average relative improvement: {avg_relative_improvement:.6f}")
                print(f"  CSV saved to: {hand_pred_csv_path}")
                
                if hand_size_stats:
                    print(f"\n  Accuracy by hand size:")
                    for hand_size in sorted(hand_size_stats.keys(), reverse=True):  # 13→1の順
                        ai_accs = hand_size_stats[hand_size]['ai']
                        baseline_accs = hand_size_stats[hand_size]['baseline']
                        avg_ai = np.mean(ai_accs) if ai_accs else 0.0
                        avg_baseline = np.mean(baseline_accs) if baseline_accs else 0.0
                        count = len(ai_accs)
                        print(f"    Hand size {hand_size:2d}: AI={avg_ai:.4f}, Baseline={avg_baseline:.4f}, Count={count}")
                
                # グラフ描画を実行
                try:
                    from evaluation.plot_hand_pred import plot_hand_prediction_accuracy
                    plot_path = plot_hand_prediction_accuracy(hand_pred_csv_path, self.elo_dir)
                    if plot_path:
                        print(f"  Plot saved to: {plot_path}")
                except Exception as e:
                    print(f"  [WARN] Failed to plot hand prediction accuracy: {e}")
            except Exception as e:
                print(f"[WARN] Failed to save hand prediction evaluation data: {e}")
                import traceback
                traceback.print_exc()
        
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
