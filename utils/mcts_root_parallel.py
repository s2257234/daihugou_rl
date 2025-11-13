"""Root-parallel MCTS harness with a central (master) batched inference server.

Usage: python tools\mcts_root_parallel.py

This script spawns N worker processes. Each worker runs run_puct_mcts but its
policy_value_fn delegates leaf evaluation requests to the master via a
multiprocessing.Queue. The master batches pending requests and evaluates them
using a provided inference function (default: uniform policy + zero value).

Notes / limitations:
- This is a prototype to demonstrate architecture. By default the master uses a
  simple uniform policy/value. You can provide a callable import path to a
  real inference function via --inference-fn (module:callable) which must accept
  (env_list, legal_list) and return a list of (policy_dict, value) tuples.
- Pickling: worker sends the full env object to master. If your env is
  not picklable, consider implementing a feature-extractor on the worker that
  sends features instead and a compatible inference_fn on the master.
"""

import argparse
import multiprocessing as mp
import pickle
import time
import traceback
import logging
import uuid
from collections import defaultdict

from agents.mcts import run_puct_mcts
from agents.drl_agent import AlphaZeroAgent as _AZ
from game.environment import DaifugoSimpleEnv


def default_inference(env_list, legal_list=None):
    """Default inference: uniform policy over legal moves, zero value.

    Accepts either:
      - env objects (with get_legal_actions()), or
      - dict payloads { 'features': [...], 'legal_actions': [...], ... }
    """
    outs = []
    legs = legal_list or [None] * len(env_list)
    for env, leg in zip(env_list, legs):
        # Prefer provided legal list; if missing, derive from payload/env best-effort
        if leg is None or leg == []:
            if isinstance(env, dict):
                leg = env.get('legal_actions') or []
            else:
                try:
                    leg = env.get_legal_actions() if env is not None else []
                except Exception:
                    leg = []
        if not leg:
            outs.append(({'pass': 1.0}, 0.0))
        else:
            try:
                p = {a: 1.0 / len(leg) for a in leg}
            except Exception:
                # Fallback: stringify non-hashables
                p = {str(a): 1.0 / len(leg) for a in leg}
            outs.append((p, 0.0))
    return outs


def _worker_main(worker_id, root_env, args, request_q, response_q, result_q):
    """Worker process entrypoint."""
    try:
        # Lazy lightweight feature extractor via AlphaZeroAgent._extract_state
        # Note: model is not required for state extraction
        _az = _AZ(player_id=0, model=None, config={"use_full_features": True})

        def policy_value_fn(env):
            # Generate a short-lived UUID for this request to avoid collisions
            req_id = uuid.uuid4().hex
            # Build compact, picklable payload: {features, legal_actions}
            try:
                st = _az._extract_state(env)
                # prefer raw full_input array; convert to list for pickling
                feats = None
                if isinstance(st, dict):
                    if 'full_input' in st and st['full_input'] is not None:
                        try:
                            feats = list(st['full_input'])
                        except Exception:
                            feats = None
                    # if compact only, pass compact dict as-is (still picklable if basic types)
                    if feats is None and 'full_compact' in st:
                        feats = {"format": st['full_compact'].get('format'),
                                 "binary_len": st['full_compact'].get('binary_len'),
                                 "packed_bits": st['full_compact'].get('packed_bits'),
                                 "floats": st['full_compact'].get('floats')}
                legal = []
                try:
                    legal = env.get_legal_actions() or []
                except Exception:
                    legal = []
                payload = {"features": feats, "legal_actions": legal, "feature_version": st.get('feature_version', 1) if isinstance(st, dict) else 1}
            except Exception:
                # Fallback: send only legal list
                try:
                    legal = env.get_legal_actions() or []
                except Exception:
                    legal = []
                payload = {"features": None, "legal_actions": legal}
            # send to master and wait on our dedicated response_q
            request_q.put((worker_id, req_id, payload))
            # block until master posts response for this req id
            while True:
                try:
                    r_id, res = response_q.get(timeout=1.0)
                    if r_id == req_id:
                        return res
                    # otherwise ignore (unlikely) and keep waiting
                except Exception:
                    time.sleep(0.001)

        def get_legal_actions_fn(env):
            try:
                return env.get_legal_actions()
            except Exception:
                return []

        # Run MCTS using delegated policy_value_fn (batched on master)
        root = run_puct_mcts(root_env, num_simulations=args.sims_per_worker,
                             policy_value_fn=policy_value_fn,
                             get_legal_actions_fn=get_legal_actions_fn,
                             batch_eval_size=1,  # worker only delegates single evals to master
                             enable_legal_cache=args.enable_legal_cache,
                             legal_cache_max_size=args.legal_cache_size,
                             enable_virtual_loss=args.enable_virtual_loss,
                             virtual_loss_count=args.virtual_loss_count,
                             virtual_loss_value=args.virtual_loss_value)

        # Extract root children visit counts
        out = {}
        try:
            for a, ch in root.children.items():
                # normalize action keys to picklable types
                try:
                    key = tuple(a) if isinstance(a, list) else a
                except Exception:
                    key = a
                out[key] = getattr(ch, 'visit_count', 0)
        except Exception:
            pass
        result_q.put((worker_id, out))
    except Exception:
        traceback.print_exc()
        result_q.put((worker_id, {}))


def master_inference_loop(request_q, response_queues, inference_fn, batch_size, timeout_sec, stop_event, logger=None):
    """Collect requests and batch-infer until stop_event is set.

    request_q: mp.Queue of (worker_id, req_id, payload)
    response_queues: list of mp.Queue per worker where master posts (req_id, out)
    """
    pending = []  # list of (worker_id, req_id, payload)
    last_flush = time.time()
    while not stop_event.is_set() or not request_q.empty() or pending:
        try:
            # If no pending items, block until one arrives (or timeout)
            if not pending:
                try:
                    item = request_q.get(timeout=timeout_sec)
                    pending.append(item)
                    # silenced per user request: no [MASTER] logs for queueing
                except Exception:
                    # timeout or empty; loop to check stop_event
                    pass

            # Drain up to batch_size quickly (non-blocking)
            try:
                while len(pending) < batch_size:
                    try:
                        item = request_q.get_nowait()
                        pending.append(item)
                        # silenced per user request: no [MASTER] logs for queueing
                    except Exception:
                        break
            except Exception:
                pass

            now = time.time()
            if pending and (len(pending) >= batch_size or (now - last_flush) >= timeout_sec):
                # Build batch
                req_worker_ids, req_ids, payloads = zip(*pending)
                envs = []
                legals = []
                for p in payloads:
                    # support three payload shapes:
                    #  - None -> placeholder
                    #  - bytes/bytearray -> pickled env (legacy)
                    #  - dict (feature/state) -> pass through to inference_fn
                    if p is None:
                        envs.append(None)
                        legals.append([])
                        continue
                    # bytes -> legacy pickled env
                    if isinstance(p, (bytes, bytearray)):
                        try:
                            env = pickle.loads(p)
                            envs.append(env)
                            try:
                                legals.append(env.get_legal_actions() or [])
                            except Exception:
                                legals.append([])
                            continue
                        except Exception:
                            envs.append(None)
                            legals.append([])
                            continue
                    # dict-like -> assume feature/state dict
                    if isinstance(p, dict) or hasattr(p, 'get'):
                        envs.append(p)
                        # prefer legal list inside dict payload if present
                        try:
                            legals.append(p.get('legal_actions') or [])
                        except Exception:
                            legals.append([])
                        continue
                    # fallback
                    try:
                        env = pickle.loads(p)
                        envs.append(env)
                        try:
                            legals.append(env.get_legal_actions() or [])
                        except Exception:
                            legals.append([])
                    except Exception:
                        envs.append(None)
                        legals.append([])
                # Run inference
                try:
                    outs = inference_fn(envs, legals)
                    if not isinstance(outs, list) or len(outs) != len(envs):
                        # fallback mapping
                        outs = [({'pass': 1.0}, 0.0) for _ in envs]
                except Exception:
                    outs = [({'pass': 1.0}, 0.0) for _ in envs]
                # silenced per user request: no [MASTER] logs for batch inference

                # Populate responses to respective worker queues (non-blocking with retries)
                for wid, rid, out in zip(req_worker_ids, req_ids, outs):
                    success = False
                    for attempt in range(3):
                        try:
                            # try non-blocking put to avoid deadlocks; some mp.Queue implementations accept block arg
                            try:
                                response_queues[wid].put((rid, out), block=False)
                            except TypeError:
                                # some implementations don't accept block kwarg; fallback to default put with timeout
                                response_queues[wid].put((rid, out))
                            success = True
                            break
                        except Exception:
                            time.sleep(0.001)
                    if not success:
                        # silenced per user request: no [MASTER] warn logs
                        pass
                    else:
                        # silenced per user request: no [MASTER] responded logs
                        pass
                pending = []
                last_flush = time.time()
            else:
                # small sleep to avoid busy-loop when idle
                time.sleep(0.001)
        except Exception:
            traceback.print_exc()
            time.sleep(0.01)


def parse_import_path(path_str):
    """Import a callable given module:callable string."""
    if not path_str:
        return None
    try:
        modname, fnname = path_str.split(':')
        mod = __import__(modname, fromlist=[fnname])
        return getattr(mod, fnname)
    except Exception:
        return None


def aggregate_results(result_items):
    total = defaultdict(int)
    for wid, mapping in result_items:
        for a, v in mapping.items():
            total[a] += int(v or 0)
    return dict(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sims-per-worker', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16, help='master batch size for inference')
    parser.add_argument('--batch-timeout', type=float, default=0.02, help='max wait seconds before flushing batch')
    parser.add_argument('--enable-legal-cache', action='store_true')
    parser.add_argument('--legal-cache-size', type=int, default=1024)
    parser.add_argument('--enable-virtual-loss', action='store_true')
    parser.add_argument('--virtual-loss-count', type=int, default=1)
    parser.add_argument('--virtual-loss-value', type=float, default=-1.0)
    parser.add_argument('--inference-fn', type=str, default='',
                        help='Optional import path module:callable to use as inference function')
    args = parser.parse_args()

    # Build a fresh root env
    root_env = DaifugoSimpleEnv()

    # Use plain multiprocessing Queues: one request queue and per-worker response queues
    # Bound the queues to avoid unbounded memory growth in heavy-load scenarios
    request_q = mp.Queue(maxsize=1000)
    response_queues = [mp.Queue(maxsize=16) for _ in range(args.workers)]
    result_q = mp.Queue()
    stop_event = mp.Event()

    # Resolve inference function
    inf_fn = parse_import_path(args.inference_fn) if args.inference_fn else None
    if inf_fn is None:
        inf_fn = default_inference

    # Start master inference thread (in main process) as a separate process to avoid blocking
    infer_proc = mp.Process(target=master_inference_loop, args=(request_q, response_queues, inf_fn, args.batch_size, args.batch_timeout, stop_event))
    infer_proc.start()

    # Spawn workers
    workers = []
    for wid in range(args.workers):
        # Each worker gets its own copy of root env (must be picklable)
        try:
            root_copy = pickle.loads(pickle.dumps(root_env))
        except Exception:
            # If serialization fails, create a new env instance
            root_copy = DaifugoSimpleEnv()
        p = mp.Process(target=_worker_main, args=(wid, root_copy, args, request_q, response_queues[wid], result_q))
        p.start()
        workers.append(p)

    # Collect results
    results = []
    for _ in workers:
        wid, mapping = result_q.get()
        results.append((wid, mapping))

    # Signal inference loop to stop
    stop_event.set()
    time.sleep(0.05)
    infer_proc.join(timeout=1.0)
    for p in workers:
        p.join(timeout=0.5)

    agg = aggregate_results(results)
    print('Aggregated root visit counts:')
    for a, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {a}: {v}")


if __name__ == '__main__':
    main()
