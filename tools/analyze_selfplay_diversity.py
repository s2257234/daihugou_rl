"""Analyze self-play diversity and log hands / actions / policy / value.

Usage example:
    python tools/analyze_selfplay_diversity.py \
        --model checkpoints/policy_value_latest.pt \
        --episodes 5 \
        --out logs/selfplay_diversity.log

This script:
- Loads env + agents via agents.factory.create_env_and_agents
- Runs a few self-play episodes
- Prints for each episode:
    - Initial hands (per player)
    - First N steps: who played what, resulting field, policy, value
- Computes simple diversity stats for initial hands and field states
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
from collections import Counter

# add project root
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from agents.factory import create_env_and_agents  # type: ignore


def _safe_mkdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _stringify_hand(hand):
    try:
        return [str(c) for c in hand]
    except Exception:
        return [repr(c) for c in hand]


def run_episode_with_details(bundle, episode_id: int, seed: int | None = None, max_steps: int = 2000):
    """Run one self-play episode and capture detailed per-step info.

    This is similar to tools.selfplay_validation.run_episode, but keeps everything in-memory
    so we can analyze it programmatically.
    """
    env = bundle.env
    agents = bundle.agents  # noqa: F841 (not used directly, but kept for symmetry)

    # seeds
    if seed is not None:
        import random

        random.seed(seed)
        try:
            import numpy as _np

            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _t

            _t.manual_seed(seed)
            if _t.cuda.is_available():
                _t.cuda.manual_seed_all(seed)
        except Exception:
            pass

    obs = env.reset()

    # initial hands
    init_hands = []
    try:
        for p in env.game.players:
            init_hands.append(_stringify_hand(p.hand))
    except Exception:
        init_hands = []

    steps = []
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        try:
            obs, reward, done, info = env.step(return_info=True)
            player_id = info.get("player_id") if isinstance(info, dict) else None
            played = info.get("played_cards") if isinstance(info, dict) else None
            field_after = info.get("field_after_play") if isinstance(info, dict) else None
            reset_reason = info.get("reset_reason") if isinstance(info, dict) else None
            policy = info.get("policy") if isinstance(info, dict) else None
            value = info.get("value") if isinstance(info, dict) else None
        except TypeError:
            # older env.step signature
            obs, reward, done = env.step()
            info = {}
            player_id = None
            played = None
            field_after = None
            reset_reason = None
            policy = None
            value = None

        step_record = {
            "step": step_count,
            "player_id": player_id,
            "played": [str(c) for c in played]
            if isinstance(played, (list, tuple))
            else (str(played) if played is not None else None),
            "field_after": field_after,
            "reset_reason": reset_reason,
            "policy": policy,
            "value": value,
        }
        steps.append(step_record)

    result = {
        "episode_id": episode_id,
        "seed": seed,
        "init_hands": init_hands,
        "steps": steps,
        "ended": done,
        "steps_count": step_count,
    }

    # best-effort: final winner
    try:
        rankings = getattr(env.game, "rankings", None)
        if rankings:
            result["winner"] = rankings[0]
        else:
            result["winner"] = None
    except Exception:
        result["winner"] = None

    return result


def _hands_signature(init_hands):
    """Create a simple hashable signature for initial hands of all players."""
    try:
        per_player = ["|".join(map(str, h)) for h in init_hands]
        return " || ".join(per_player)
    except Exception:
        return str(init_hands)


def _field_sequence_signature(steps, max_len: int = 50):
    """Signature of early field states to detect repeated situations."""
    sigs = []
    for s in steps[:max_len]:
        fa = s.get("field_after")
        try:
            sigs.append(json.dumps(fa, sort_keys=True, ensure_ascii=False))
        except Exception:
            sigs.append(repr(fa))
    return " || ".join(sigs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to model checkpoint to load")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", type=str, default="logs/selfplay_diversity.log")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--show_steps", type=int, default=30, help="How many steps per episode to print in detail")
    args = ap.parse_args()

    _safe_mkdir(args.out)

    # load config
    try:
        from agents.config import ALPHA_ZERO_CONFIG as CONFIG  # type: ignore

        config = dict(CONFIG)
    except Exception:
        config = {}

    if args.device:
        config["device"] = args.device

    print(f"[INFO] Creating env and agents with model {args.model}")
    bundle = create_env_and_agents(config, context="selfplay_validation", model_path=args.model, resolved_device=args.device)
    print(f"[INFO] Model loaded from checkpoint? {bundle.loaded_from_checkpoint}, device={bundle.device}")

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"[INFO] Running episode {ep} (seed={seed})")
        res = run_episode_with_details(bundle, ep, seed=seed, max_steps=args.max_steps)
        results.append(res)

    # diversity analysis
    hand_sigs = [_hands_signature(r["init_hands"]) for r in results]
    unique_hand_sigs = set(hand_sigs)
    print("\n===== Initial hands diversity =====")
    print(f"episodes: {len(results)}; unique initial-hand patterns: {len(unique_hand_sigs)}")
    repeated_hand_patterns = []
    if len(unique_hand_sigs) < len(results):
        counts = Counter(hand_sigs)
        print("[WARN] Some episodes share identical initial hands. Top patterns:")
        for sig, cnt in counts.most_common(5):
            print(f"  count={cnt}: {sig}")
            repeated_hand_patterns.append({"pattern": sig, "count": cnt})

    field_sigs = [_field_sequence_signature(r["steps"]) for r in results]
    unique_field_sigs = set(field_sigs)
    print("\n===== Early field-state sequence diversity (first steps) =====")
    print(f"episodes: {len(results)}; unique early-field patterns: {len(unique_field_sigs)}")
    repeated_field_patterns = []
    if len(unique_field_sigs) < len(results):
        counts = Counter(field_sigs)
        print("[WARN] Some episodes share very similar early field sequences. Top patterns:")
        for sig, cnt in counts.most_common(5):
            print(f"  count={cnt}; first 1-2 states: {sig.split(' || ')[:2]}")
            repeated_field_patterns.append({"pattern_excerpt": sig.split(" || ")[:2], "count": cnt})

    diversity_summary = {
        "episodes": len(results),
        "unique_initial_hand_patterns": len(unique_hand_sigs),
        "unique_early_field_patterns": len(unique_field_sigs),
    }
    if repeated_hand_patterns:
        diversity_summary["repeated_initial_hand_patterns_top"] = repeated_hand_patterns
    if repeated_field_patterns:
        diversity_summary["repeated_early_field_patterns_top"] = repeated_field_patterns

    # write detailed log for manual inspection
    with open(args.out, "w", encoding="utf-8") as f:
        header = {
            "timestamp": time.time(),
            "model": args.model,
            "episodes": args.episodes,
            "seed": args.seed,
        }
        f.write(json.dumps({"header": header}, ensure_ascii=False) + "\n")

        # write diversity summary also into the log file
        f.write(json.dumps({"diversity_summary": diversity_summary}, ensure_ascii=False) + "\n")

        for res in results:
            ep = res["episode_id"]
            seed = res["seed"]
            f.write(
                json.dumps(
                    {
                        "episode_summary": {
                            "episode": ep,
                            "seed": seed,
                            "ended": res["ended"],
                            "steps_count": res["steps_count"],
                            "winner": res.get("winner"),
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.write(json.dumps({"initial_hands": res["init_hands"]}, ensure_ascii=False) + "\n")
            for s in res["steps"]:
                f.write(json.dumps({"step": s}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"episode_end": {"episode": ep, "ended": res["ended"]}}, ensure_ascii=False) + "\n")

    # also pretty-print first N steps per episode to stdout for quick visual check
    print("\n===== Per-episode detailed preview (truncated) =====")
    for res in results:
        ep = res["episode_id"]
        print(f"\n--- Episode {ep} (seed={res['seed']}) ---")
        print("Initial hands:")
        for pid, h in enumerate(res["init_hands"]):
            print(f"  Player {pid}: {h}")
        print(f"Steps (first {args.show_steps}):")
        for s in res["steps"][: args.show_steps]:
            pid = s.get("player_id")
            played = s.get("played")
            field_after = s.get("field_after")
            policy = s.get("policy")
            value = s.get("value")
            print(f"  step={s['step']} pid={pid} played={played} field_after={field_after}")
            print(f"    value={value} policy={policy}")

    print(f"\n[INFO] Detailed self-play diversity log written to {args.out}")


if __name__ == "__main__":
    main()
