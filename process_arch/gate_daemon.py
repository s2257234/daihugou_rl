from __future__ import annotations
"""Gate evaluator daemon: promotes candidate model to best model if win rate threshold met.

Usage:
  python -m process_arch.gate_daemon

Monitors candidate_model_dir for newest candidate file. Evaluates vs best_model_path
using evaluation.gating.evaluate_candidate and promotes on threshold.
Writes JSON result to gate_result_path and touches model_refresh_flag_path when promoting.
"""
import os
import sys
import time
import json
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from evaluation.gating import evaluate_candidate, _override_eval_config


def newest_file(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pt')]
    if not files:
        return None
    files.sort(key=lambda p: (os.path.getmtime(p), p))
    return files[-1]


def load_model(path: str):
    from agents.models import PolicyValueNet
    return PolicyValueNet.load(path, map_location='cpu')


def run_gate():
    cfg = dict(ALPHA_ZERO_CONFIG)
    poll = float(cfg.get('gate_poll_interval_sec', 60.0) or 60.0)
    cand_dir = cfg.get('candidate_model_dir', 'checkpoints/candidates')
    best_path = cfg.get('best_model_path', cfg.get('checkpoint_path'))
    result_path = cfg.get('gate_result_path', 'gate_results/latest_gate.json')
    threshold = float(cfg.get('eval_gate_threshold', 0.55) or 0.55)
    refresh_flag = cfg.get('model_refresh_flag_path', 'checkpoints/_refresh.flag')
    seed = cfg.get('eval_gate_seed', None)
    games = int(cfg.get('eval_gate_games', 40) or 40)
    def _log(msg: str):
        try:
            import datetime
            log_dir = cfg.get('log_dir','logs')
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] [GATE] {msg}\n")
        except Exception:
            pass
    _log(f"watching '{cand_dir}' threshold={threshold} games={games}")
    last_promoted_mtime = None
    last_evaluated_path = None
    last_evaluated_mtime = 0.0
    while True:
        cand_path = newest_file(cand_dir)
        if cand_path:
            try:
                mtime = os.path.getmtime(cand_path)
            except Exception:
                mtime = 0.0
        else:
            mtime = 0.0

        # Evaluate only if this candidate is new (path/mtime changed) or a newer one exists
        should_eval = False
        if cand_path:
            if last_evaluated_path is None:
                should_eval = True
            elif os.path.basename(cand_path) != os.path.basename(last_evaluated_path):
                should_eval = True
            elif mtime > last_evaluated_mtime:
                should_eval = True

        if cand_path and should_eval:
            try:
                cand_model = load_model(cand_path)
                base_model = load_model(best_path) if os.path.exists(best_path) else cand_model
                eval_cfg = _override_eval_config(cfg)
                res = evaluate_candidate(cand_model, base_model, eval_cfg, games=games, seed=seed)
                win_rate = res.get('win_rate', 0.0)
                res_out = {
                    'candidate': os.path.basename(cand_path),
                    'best_model': os.path.basename(best_path) if os.path.exists(best_path) else None,
                    'win_rate': win_rate,
                    'wins': res.get('wins'),
                    'games': res.get('games'),
                    'promoted': False,
                }
                if win_rate >= threshold:
                    # promote
                    try:
                        import shutil
                        shutil.copy2(cand_path, best_path)
                        # touch refresh flag
                        with open(refresh_flag, 'w', encoding='utf-8') as f:
                            f.write('promoted')
                        res_out['promoted'] = True
                        try:
                            last_promoted_mtime = os.path.getmtime(cand_path)
                        except Exception:
                            last_promoted_mtime = mtime
                        _log(f"promoted {os.path.basename(cand_path)} -> {os.path.basename(best_path)} win_rate={win_rate:.3f}")
                    except Exception as e:
                        _log(f"promotion failed: {e}")
                else:
                    _log(f"candidate {os.path.basename(cand_path)} win_rate={win_rate:.3f} not promoted")
                # write result json
                try:
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(res_out, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    _log(f"result write failed: {e}")
                # mark evaluated
                last_evaluated_path = cand_path
                last_evaluated_mtime = mtime
            except Exception as e:
                _log(f"evaluation failed: {e}")
        time.sleep(poll)


if __name__ == '__main__':
    run_gate()
