"""Pre-update evaluation gate.

Pit a candidate model vs the current/baseline model before promoting the
candidate. Uses duplicate deals with seat swap to reduce variance, and
disables exploration noise for pure strength comparison.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp


def _override_eval_config(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    # Inference-only behavior: no opening randomness, no dirichlet at root
    if cfg.get("eval_gate_disable_opening_random", True):
        cfg["opening_random_enable"] = False
    if cfg.get("eval_gate_disable_dirichlet", True):
        cfg["inference_dirichlet"] = False
    # Temperature handling is already greedy when training=False in agent
    return cfg


def _play_game_with_mapping(
    candidate_model,
    baseline_model,
    base_cfg: Dict[str, Any],
    seat_map: Tuple[str, str, str, str],  # e.g., ("cand","base","base","base")
) -> Dict[str, Any]:
    from agents.drl_agent import AlphaZeroAgent
    from game.environment import DaifugoSimpleEnv

    cfg = _override_eval_config(base_cfg)
    num_players = int(cfg.get("num_players", 4))
    assert num_players == 4, "This evaluator currently assumes 4 players"

    def _make_agent(kind: str, pid: int):
        if kind == "cand":
            return AlphaZeroAgent(player_id=pid, model=candidate_model, config=cfg)
        else:
            return AlphaZeroAgent(player_id=pid, model=baseline_model, config=cfg)

    agents: List[Any] = []
    for i, tag in enumerate(seat_map):
        agents.append(_make_agent("cand" if tag == "cand" else "base", i))

    env = DaifugoSimpleEnv(num_players=4, agent_classes=None)
    env.agents = agents
    for ag in agents:
        if hasattr(ag, 'set_env_ref'):
            ag.set_env_ref(env)
    env.reset()

    # Play until done
    steps = 0
    max_steps = int(cfg.get("max_episode_steps", 1000))
    while not getattr(env.game, 'done', False):
        if steps >= max_steps:
            break
        pid = env.game.turn
        ag = env.agents[pid]
        try:
            act = ag.select_action(env, training=False)
        except TypeError:
            act = ag.select_action(env, training=False)
        try:
            env.step(external_action=act)
        except TypeError:
            env.step(act)
        steps += 1

    rankings: List[int] = list(getattr(env.game, 'rankings', []))
    if len(rankings) != 4:
        # fill remaining by order if forced stop
        remaining = [i for i in range(4) if i not in rankings]
        rankings += remaining
    return {"rankings": rankings, "steps": steps}


def evaluate_candidate(
    candidate_model,
    baseline_model,
    base_cfg: Dict[str, Any],
    games: int = 20,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Run duplicate-deal, seat-swapped evaluation.

    Strategy:
      - Use 10 seed values; for each seed, play two games:
        Game A: seats = (cand, base, base, base)
        Game B: seats = (base, cand, base, base)
      - For each game, compare candidate's rank with the opponent at the
        swapped seat (A: cand@0 vs base@1; B: cand@1 vs base@0).
      - Win if candidate's final rank is strictly better (smaller).

    Returns dict with wins, total, win_rate and per-game details.
    """
    total = max(2, int(games))
    if total % 2 != 0:
        total += 1  # ensure even number for pairs
    pairs = total // 2

    # Seeding for duplicate deals
    if seed is not None:
        try:
            import numpy as _np
            random.seed(seed)
            _np.random.seed(seed % (2**32 - 1))
        except Exception:
            random.seed(seed)

    details = []
    wins = 0
    for i in range(pairs):
        base_seed = (seed or 0) + i if seed is not None else random.randint(0, 10**9)
        # Game A
        random.seed(base_seed)
        try:
            import numpy as _np
            _np.random.seed(base_seed % (2**32 - 1))
        except Exception:
            pass
        res_a = _play_game_with_mapping(candidate_model, baseline_model, base_cfg, ("cand","base","base","base"))
        # Game B (same deal)
        random.seed(base_seed)
        try:
            import numpy as _np
            _np.random.seed(base_seed % (2**32 - 1))
        except Exception:
            pass
        res_b = _play_game_with_mapping(candidate_model, baseline_model, base_cfg, ("base","cand","base","base"))

        # Determine wins per game by seat counterpart
        rk_a = res_a["rankings"]
        rk_b = res_b["rankings"]
        # In A: compare cand@0 vs base@1
        cand_rank_a = rk_a.index(0)
        base_rank_a = rk_a.index(1)
        win_a = 1 if cand_rank_a < base_rank_a else 0
        # In B: compare cand@1 vs base@0
        cand_rank_b = rk_b.index(1)
        base_rank_b = rk_b.index(0)
        win_b = 1 if cand_rank_b < base_rank_b else 0
        wins += (win_a + win_b)
        details.append({
            "pair_index": i,
            "seed": base_seed,
            "gameA_rankings": rk_a,
            "gameB_rankings": rk_b,
            "gameA_win": bool(win_a),
            "gameB_win": bool(win_b),
        })

    win_rate = wins / float(total)
    return {"wins": wins, "games": total, "win_rate": win_rate, "details": details}


# -------------------- Async Gate Helpers --------------------
def gate_worker_eval_proc(
    candidate_path: str,
    baseline_path: str,
    cfg: Dict[str, Any],
    games: int,
    seed: Optional[int],
    device: str,
    out_queue,
):
    """Spawned process entry for async evaluation gate.
    Loads candidate & baseline models then calls evaluate_candidate.
    Puts result or error dict on out_queue.
    """
    try:
        from agents.models import PolicyValueNet as _PVN
        cand_model = _PVN.load(candidate_path, map_location=device)
        base_model = _PVN.load(baseline_path, map_location=device)
        if device:
            cand_model.to(device)  # type: ignore[arg-type]
            base_model.to(device)  # type: ignore[arg-type]
        res = evaluate_candidate(cand_model, base_model, cfg, games=games, seed=seed)
        out_queue.put(res)
    except Exception as e:
        try:
            out_queue.put({"error": str(e)})
        except Exception:
            pass


def start_async_gate(
    candidate_path: str,
    baseline_path: str,
    cfg: Dict[str, Any],
    *,
    games: int,
    seed: Optional[int] = None,
    device: str = "cpu",
    ctx: Optional[mp.context.BaseContext] = None,
):
    """Start async evaluation gate in a separate process.
    Returns (proc, queue).
    """
    if ctx is None:
        try:
            ctx = mp.get_context("spawn")
        except Exception:
            ctx = mp
    q = ctx.Queue()
    p = ctx.Process(
        target=gate_worker_eval_proc,
        args=(candidate_path, baseline_path, dict(cfg), int(games), seed, device, q),
        daemon=True,
    )
    p.start()
    return p, q


def poll_async_gate(q, *, block: bool = False, timeout: float = 0.0):
    """Poll result from async gate queue.
    Returns result dict or None if not ready.
    """
    try:
        if block:
            return q.get(timeout=timeout)
        else:
            return q.get_nowait()
    except Exception:
        return None


def decide_sync(
    candidate_model,
    baseline_model,
    cfg: Dict[str, Any],
    *,
    games: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Run sync evaluation and produce decision using cfg threshold.
    Returns: { 'result': <eval dict>, 'pass': bool, 'threshold': float }
    """
    g = int(cfg.get("eval_gate_games", 20) or 20) if games is None else int(games)
    thr = float(cfg.get("eval_gate_threshold", 0.6) or 0.6)
    res = evaluate_candidate(candidate_model, baseline_model, cfg, games=g, seed=seed)
    return {"result": res, "pass": bool(res.get("win_rate", 0.0) >= thr), "threshold": thr}


def finalize_async_gate_if_ready(
    proc,
    q,
    *,
    cand_path: Optional[str],
    base_path: Optional[str],
    cfg: Dict[str, Any],
    logger: Optional[Any],
    keep_prev_model: bool,
    snapshot_cb: Optional[callable] = None,
) -> Optional[Dict[str, Any]]:
    """If async gate has produced a result, finalize promotion/rejection.

    - Non-blocking: returns None if result not yet available.
    - On completion: moves or deletes candidate file and logs decision.
    - Returns a dict: { 'result': <eval dict or {'win_rate':..}>, 'pass': bool, 'threshold': float }
    """
    if proc is None:
        return None
    # Fetch result non-blocking
    if proc.is_alive():
        res = poll_async_gate(q)
        if res is None:
            return None
    else:
        res = poll_async_gate(q)
    # Cleanup process (best-effort)
    try:
        if not proc.is_alive():
            proc.join(timeout=1)
    except Exception:
        pass
    # No result case
    if res is None:
        if logger:
            try:
                logger.log_text("[WARN] eval-gate async finished without result")
            except Exception:
                pass
        return {"result": None, "pass": False, "threshold": float(cfg.get("eval_gate_threshold", 0.6) or 0.6)}
    # Error case
    if isinstance(res, dict) and "error" in res:
        if logger:
            try:
                logger.log_text(f"[WARN] eval-gate async failed: {res['error']}")
            except Exception:
                pass
        return {"result": None, "pass": False, "threshold": float(cfg.get("eval_gate_threshold", 0.6) or 0.6)}
    # Decision
    thr = float(cfg.get("eval_gate_threshold", 0.6) or 0.6)
    win_rate = float(res.get("win_rate", 0.0)) if isinstance(res, dict) else 0.0
    gated_pass = (win_rate >= thr)
    # Log summary
    try:
        decision = 'promote' if gated_pass else 'reject'
        wins = None
        total = None
        try:
            wins = int(res.get('wins')) if isinstance(res, dict) and res.get('wins') is not None else None
            total = int(res.get('games')) if isinstance(res, dict) and res.get('games') is not None else None
        except Exception:
            wins, total = None, None
        import os as _os
        cand_name = _os.path.basename(cand_path) if cand_path else None
        base_name = _os.path.basename(base_path) if base_path else None
        msg = f"[gate] done async win_rate={win_rate:.2%}"
        if wins is not None and total is not None:
            msg += f" ({wins}/{total})"
        msg += f" decision={decision} thr={thr:.0%}"
        if cand_name and base_name:
            msg += f" cand={cand_name} base={base_name}"
        if logger:
            logger.log_text(msg)
        else:
            print(msg)
    except Exception:
        pass
    # Promote or discard
    try:
        if gated_pass and cand_path is not None:
            latest_path = cfg.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
            os.replace(cand_path, latest_path)
            if keep_prev_model and snapshot_cb is not None:
                try:
                    snapshot_cb()
                except Exception:
                    pass
        else:
            if cand_path is not None and os.path.exists(cand_path):
                try:
                    os.remove(cand_path)
                except Exception:
                    pass
    except Exception:
        pass
    return {"result": res if isinstance(res, dict) else {"win_rate": win_rate}, "pass": gated_pass, "threshold": thr}


