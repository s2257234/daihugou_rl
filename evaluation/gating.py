"""Pre-update evaluation gate.

Pit a candidate model vs the current/baseline model before promoting the
candidate. Uses duplicate deals with seat swap to reduce variance, and
disables exploration noise for pure strength comparison.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple


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
