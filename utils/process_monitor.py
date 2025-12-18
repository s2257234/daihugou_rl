from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


class ProcessMonitor:
    """
    Process/memory monitoring and graceful worker restart controller.

    Responsibilities moved out of Trainer to keep SRP:
    - Memory-based marking of workers for restart (including emergency total RSS)
    - Graceful restart flow: flush -> wait -> terminate -> respawn

    Behavior mirrors the original inline implementation.
    """

    def __init__(
        self,
        restart_cfg: Dict[str, Any],
        *,
        logger: Optional[Any] = None,
        get_gate_proc: Optional[Callable[[], Any]] = None,
        clear_gate_handles: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = dict(restart_cfg)
        self.logger = logger
        self.get_gate_proc = get_gate_proc or (lambda: None)
        self.clear_gate_handles = clear_gate_handles or (lambda: None)

    def _psutil_process(self, pid):
        if psutil is None:
            return None
        try:
            return psutil.Process(pid)
        except Exception:
            return None

    def check_and_mark_restarts(
        self,
        procs: List[Any],
        worker_stats: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        cfg = self.cfg
        marked: List[int] = []
        has_emergency = int(cfg.get('emergency_total_mb', 0) or 0) > 0
        has_normal_high = bool(cfg.get('enable', False)) and (int(cfg.get('high_mb', 0) or 0) > 0)
        if not has_emergency and not has_normal_high:
            return marked
        if psutil is None:
            return marked
        # Parent + workers RSS
        try:
            parent_rss = psutil.Process().memory_info().rss
        except Exception:
            parent_rss = 0
        total_rss_children = 0
        rss_by_wid = {}
        for wid, p in enumerate(procs):
            try:
                if not p.is_alive():
                    continue
                proc_obj = self._psutil_process(p.pid)
                if proc_obj is None:
                    continue
                rss = proc_obj.memory_info().rss
                total_rss_children += rss
                rss_by_wid[p.pid] = rss
            except Exception:
                continue
        # Add gate process (if alive)
        gate_rss = 0
        gate_pid = None
        try:
            gp = self.get_gate_proc()
            if gp is not None and gp.is_alive():
                proc_gate = self._psutil_process(gp.pid)
                if proc_gate is not None:
                    gate_rss = proc_gate.memory_info().rss
                    gate_pid = gp.pid
                    total_rss_children += gate_rss
        except Exception:
            pass
        total_rss = parent_rss + total_rss_children
        emergency_threshold = int(cfg.get('emergency_total_mb', 0) or 0) * 1024 * 1024
        emergency = emergency_threshold > 0 and (total_rss >= emergency_threshold)
        high_list = []
        if has_normal_high:
            for wid, p in enumerate(procs):
                if not p.is_alive():
                    continue
                proc_obj = self._psutil_process(p.pid)
                if proc_obj is None:
                    continue
                try:
                    rss = proc_obj.memory_info().rss
                except Exception:
                    continue
                rss_mb = rss / (1024 * 1024)
                st = worker_stats.get(wid)
                if st is None:
                    continue
                if rss_mb >= int(cfg['high_mb']):
                    st['rss_high_count'] = int(st.get('rss_high_count', 0)) + 1
                else:
                    st['rss_high_count'] = 0
                eligible = (
                    st['rss_high_count'] >= int(cfg.get('consecutive', 2) or 2)
                    and (time.time() - float(st.get('last_restart', 0.0))) >= int(cfg.get('min_interval', 600) or 600)
                    and not bool(st.get('pending'))
                )
                if eligible:
                    high_list.append((wid, rss_mb))
        # Pick targets
        target_wids: List[int] = []
        if emergency:
            candidates = []  # (kind, id, rss, ok_interval)
            for wid, p in enumerate(procs):
                if not p.is_alive():
                    continue
                st = worker_stats.get(wid)
                if st is None or bool(st.get('pending')):
                    continue
                proc_obj = self._psutil_process(p.pid)
                if proc_obj is None:
                    continue
                try:
                    rss_now = proc_obj.memory_info().rss
                except Exception:
                    continue
                since = time.time() - float(st.get('last_restart', 0.0))
                ok_interval = since >= int(cfg.get('min_interval', 600) or 600)
                candidates.append(("worker", wid, rss_now, ok_interval))
            if gate_pid is not None and gate_rss > 0:
                candidates.append(("gate", int(gate_pid), gate_rss, True))
            if candidates:
                from math import inf
                c_ok = [c for c in candidates if c[3]]
                pick_from = c_ok if c_ok else candidates
                kind_pick, ident_pick, rss_pick, _ = max(pick_from, key=lambda x: x[2] if x else -1)
                if kind_pick == "worker":
                    target_wids = [int(ident_pick)]
                    if self.logger:
                        try:
                            self.logger.log_text(
                                f"[worker-restart] emergency trigger total_rss={total_rss/(1024**3):.2f}GB parent={parent_rss/(1024**3):.2f}GB threshold={int(cfg.get('emergency_total_mb',0))/1024:.2f}GB wid={int(ident_pick)}"
                            )
                        except Exception:
                            pass
                elif kind_pick == "gate":
                    try:
                        gp = self.get_gate_proc()
                        if gp is not None and gp.is_alive():
                            gp.terminate()
                            try:
                                gp.join(timeout=3)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # 呼び出し側ハンドルをクリア
                    try:
                        self.clear_gate_handles()
                    except Exception:
                        pass
                    try:
                        # 呼び出し側でハンドルをクリアする想定（Trainerが行う）
                        if self.logger:
                            self.logger.log_text(
                                f"[worker-restart] emergency trigger total_rss={total_rss/(1024**3):.2f}GB parent={parent_rss/(1024**3):.2f}GB threshold={int(cfg.get('emergency_total_mb',0))/1024:.2f}GB gate_pid={int(ident_pick)} terminated"
                            )
                    except Exception:
                        pass
        else:
            target_wids = [wid for wid, _ in high_list]
        for wid in target_wids:
            st = worker_stats.setdefault(wid, {})
            st['pending'] = True
            st['grace_start'] = time.time()
            marked.append(wid)
            if self.logger:
                try:
                    self.logger.log_text(f"[worker-restart] mark wid={wid} gen={st.get('generation',0)} reason={'emergency' if emergency else 'high_rss'}")
                except Exception:
                    pass
        return marked

    def issue_graceful_restart(
        self,
        wid_list: List[int],
        procs: List[Any],
        control_queues: List[Any],
        make_control_queue: Callable[[], Any],
        spawn_worker: Callable[[int, Any], Any],
        worker_stats: Dict[int, Dict[str, Any]],
    ) -> None:
        cfg = self.cfg
        # 1) flush
        for wid in wid_list:
            try:
                if 0 <= wid < len(control_queues):
                    control_queues[wid].put('FLUSH_AND_EXIT', block=False)
            except Exception:
                pass
        # 2) grace wait
        deadline = time.time() + int(cfg.get('grace_timeout', 120) or 120)
        still_alive = set([w for w in wid_list if 0 <= w < len(procs)])
        while still_alive and time.time() < deadline:
            for wid in list(still_alive):
                try:
                    if not procs[wid].is_alive():
                        still_alive.discard(wid)
                except Exception:
                    still_alive.discard(wid)
            time.sleep(0.1)
        # 3) terminate
        for wid in list(still_alive):
            try:
                procs[wid].terminate()
            except Exception:
                pass
            try:
                procs[wid].join(timeout=max(1, int(cfg.get('force_kill_sec', 150) or 150)))
            except Exception:
                pass
        # 4) respawn
        for wid in wid_list:
            try:
                try:
                    if procs[wid].is_alive():
                        procs[wid].terminate()
                        procs[wid].join(timeout=1)
                except Exception:
                    pass
                cq = make_control_queue()
                control_queues[wid] = cq
                p = spawn_worker(wid, cq)
                p.start()
                procs[wid] = p
                st = worker_stats.get(wid, {})
                st['pending'] = False
                st['last_restart'] = time.time()
                st['generation'] = int(st.get('generation', 0)) + 1
                st['rss_high_count'] = 0
                worker_stats[wid] = st
            except Exception:
                pass


def auto_replay_water_purge(
    cfg: Dict[str, Any],
    rb: Any,
    logger: Optional[Any] = None,
    *,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Memory(RSS) based auto replay buffer shrinking.

    Moved from Trainer to utils to keep Trainer focused on pure training.
    This function is side-effectful on the given replay buffer and returns
    an updated state dict to keep small time-caches across calls.

    Expected cfg keys mirror the original implementation:
    - auto_replay_water_enabled, buffer_size, replay_memory_* family, etc.
    """
    st = dict(state or {})
    try:
        if cfg.get('purge_replay_after_each_update') or not cfg.get('auto_replay_water_enabled', False):
            return st
        if rb is None:
            return st
        try:
            capacity = int(cfg.get('buffer_size', 1))
            cur = len(rb) if hasattr(rb, '__len__') else None
        except Exception:
            return st
        if not cur or capacity <= 0:
            return st
        # psutil required
        try:
            import psutil as _ps  # type: ignore
            proc = _ps.Process()
        except Exception:
            return st
        include_children = bool(cfg.get('replay_memory_include_children', False))
        child_interval = int(cfg.get('replay_memory_children_recalc_sec', 15) or 15)
        now = time.time()
        try:
            parent_rss = proc.memory_info().rss
            total_mem = _ps.virtual_memory().total
        except Exception:
            return st
        children_rss = 0
        if include_children:
            last_children_ts = float(st.get('last_children_rss_ts', 0.0))
            cached_children = st.get('last_children_rss', None)
            if (now - last_children_ts) < child_interval and cached_children is not None:
                children_rss = int(cached_children)
            else:
                try:
                    total_ch = 0
                    child_infos = []
                    for c in proc.children(recursive=True):
                        try:
                            if not c.is_running():
                                continue
                            mi = c.memory_info().rss
                            total_ch += mi
                            child_infos.append((c.pid, mi, getattr(c, 'name', lambda: '')()))
                        except Exception:
                            continue
                    child_infos.sort(key=lambda x: x[1], reverse=True)
                    top_n = child_infos[:5]
                    if top_n and logger and bool(cfg.get('replay_log_child_rss', False)):
                        try:
                            line = ", ".join([f"pid={pid} rss={rss/(1024**3):.2f}GB" for pid, rss, _nm in top_n])
                            logger.log_text(f"[replay] child_rss_top total_children={len(child_infos)} top5=[{line}]")
                        except Exception:
                            pass
                    children_rss = total_ch
                    st['last_children_rss'] = children_rss
                    st['last_children_rss_ts'] = now
                except Exception:
                    children_rss = 0
        rss = parent_rss + children_rss
        total = total_mem
        high_ratio = float(cfg.get('replay_memory_high_ratio', 0.8) or 0.8)
        low_ratio = float(cfg.get('replay_memory_low_ratio', 0.6) or 0.6)
        abs_mb = int(cfg.get('replay_memory_high_abs_mb', 0) or 0)
        low_abs_mb = int(cfg.get('replay_memory_low_abs_mb', 0) or 0)
        cooldown = int(cfg.get('replay_memory_cooldown_sec', 300) or 300)
        emergency_ratio = float(cfg.get('replay_memory_emergency_ratio', 0.0) or 0.0)
        aggressive_factor = float(cfg.get('replay_memory_aggressive_factor', 0.8) or 0.8)
        min_purge_rows = int(cfg.get('replay_memory_min_purge_rows', 0) or 0)
        debug_log = bool(cfg.get('replay_memory_debug_log', False))
        log_before_after = bool(cfg.get('replay_memory_log_before_after', False))
        force_gc = bool(cfg.get('replay_memory_force_gc', False))
        compact_mode = cfg.get('replay_memory_compact_mode', 'none') or 'none'
        if not (0.0 < low_ratio < high_ratio < 1.0):
            return st
        usage_ratio = rss / total if total else 0.0
        over_ratio = usage_ratio >= high_ratio
        over_abs = abs_mb > 0 and (rss >= abs_mb * 1024 * 1024)
        if debug_log:
            msg_chk = (f"[replay] mem_check ratio={usage_ratio:.4f} parent={parent_rss/(1024**3):.2f}GB "
                       f"children={children_rss/(1024**3):.2f}GB total={rss/(1024**3):.2f}GB high={high_ratio:.2f} "
                       f"low={low_ratio:.2f} over_ratio={over_ratio} over_abs={over_abs}")
            if logger:
                try: logger.log_text(msg_chk)
                except Exception: pass
            else:
                print(msg_chk)
        if not (over_ratio or over_abs):
            return st
        last = float(st.get('last_mem_water_ts', 0.0))
        if (now - last) < cooldown:
            return st
        cur_usage_ratio = usage_ratio
        # --- ターゲットサイズ計算 ---
        target_size = int(capacity * (cur / capacity * (low_ratio / max(cur_usage_ratio, 1e-9))))
        min_target = int(capacity * low_ratio)
        if target_size < min_target:
            target_size = min_target
        if over_abs and low_abs_mb > 0:
            ratio_factor = low_abs_mb / max(abs_mb, 1)
            alt_target = int(cur * ratio_factor)
            if alt_target < target_size:
                target_size = max(0, alt_target)
        if emergency_ratio and usage_ratio >= emergency_ratio and 0.0 < aggressive_factor < 1.0:
            target_size = int(target_size * aggressive_factor)
        if min_purge_rows > 0 and (cur - target_size) < min_purge_rows:
            target_size = max(0, cur - min_purge_rows)
        if target_size < 0:
            target_size = 0
        if target_size >= cur:
            st['last_mem_water_ts'] = now
            return st
        remove_count = cur - target_size
        if remove_count > 0:
            try:
                import os
                from itertools import islice
                import joblib
                snapshot_dir = cfg.get('checkpoint_dir', 'checkpoints')
                os.makedirs(snapshot_dir, exist_ok=True)
                append_path = os.path.join(snapshot_dir, 'replay_autopurge_append.joblib')
                oldest_list = []
                if hasattr(rb, '_data'):
                    lock = getattr(rb, '_lock', None)
                    if lock is not None:
                        with lock:  # type: ignore
                            oldest_list = list(islice(rb._data, 0, remove_count))  # type: ignore[attr-defined]
                    else:
                        oldest_list = list(islice(rb._data, 0, remove_count))  # type: ignore[attr-defined]
                elif hasattr(rb, '__iter__'):
                    try:
                        it = iter(rb)
                        for _ in range(remove_count):
                            try:
                                oldest_list.append(next(it))
                            except StopIteration:
                                break
                    except Exception:
                        oldest_list = []
                allow_keys = {"player_id","state","model_version","feature_version","uid","pi_q","pi_format","legal_ids","actions_format","value_u8"}
                processed = []
                for s in oldest_list:
                    if not isinstance(s, dict):
                        continue
                    try:
                        d = {k: s.get(k) for k in allow_keys if k in s}
                        import time as _t
                        d['autopurge_ts'] = _t.time()
                        processed.append(d)
                    except Exception:
                        continue
                if os.path.exists(append_path):
                    try:
                        obj = joblib.load(append_path)
                        base_list = obj.get('data', []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
                    except Exception:
                        base_list = []
                else:
                    base_list = []
                base_list.extend(processed)
                if len(base_list) > 2_000_000:
                    base_list = base_list[-1_500_000:]
                next_id_val = None
                try:
                    next_id_val = getattr(rb, '_next_id', None)
                except Exception:
                    next_id_val = None
                payload = {'maxlen': getattr(rb, 'maxlen', None), 'next_id': next_id_val, 'data': base_list}
                tmp_ap = append_path + '.tmp'
                try:
                    joblib.dump(payload, tmp_ap, compress=3)
                    os.replace(tmp_ap, append_path)
                except Exception:
                    try:
                        if os.path.exists(tmp_ap):
                            os.remove(tmp_ap)
                    except Exception:
                        pass
                    try:
                        joblib.dump(payload, append_path, compress=0)
                    except Exception:
                        pass
                if logger:
                    try:
                        logger.log_text(f"[replay] autopurge_append_saved path={append_path} added={len(processed)} total={len(base_list)} remove_count={remove_count} target={target_size}")
                    except Exception:
                        pass
            except Exception:
                pass
        if hasattr(rb, 'shrink_to_size'):
            try:
                removed = rb.shrink_to_size(target_size)
                new_size = len(rb)
            except Exception:
                return st
        else:
            removed = 0
            over = cur - target_size
            try:
                if over > 0 and hasattr(rb, 'popleft'):
                    for _ in range(over):
                        try:
                            rb.popleft(); removed += 1
                        except Exception:
                            break
                elif over > 0 and isinstance(rb, list):
                    del rb[:over]; removed = over
                new_size = len(rb)
            except Exception:
                return st
        if compact_mode == 'rebuild' and hasattr(rb, '_data'):
            try:
                from collections import deque as _dq
                if hasattr(rb, '_lock'):
                    with rb._lock:  # type: ignore[attr-defined]
                        data_list = list(rb._data)  # type: ignore[attr-defined]
                        rb._data = _dq(data_list, maxlen=rb.maxlen)  # type: ignore[attr-defined]
                else:
                    data_list = list(rb._data)  # type: ignore[attr-defined]
                    rb._data = _dq(data_list, maxlen=rb.maxlen)  # type: ignore[attr-defined]
            except Exception:
                pass
        after_delete_rss = rss
        if log_before_after and logger:
            try:
                logger.log_text(f"[replay] memory_purge after_delete removed={removed} cur={new_size} target={target_size} ratio_before={cur_usage_ratio:.4f}")
            except Exception:
                pass
        if force_gc:
            try:
                import gc
                gc.collect()
                parent_after = proc.memory_info().rss
                children_after = 0
                if include_children:
                    try:
                        for c in proc.children(recursive=True):
                            try:
                                if c.is_running(): children_after += c.memory_info().rss
                            except Exception:
                                continue
                    except Exception:
                        pass
                total_after = parent_after + children_after
                freed = (after_delete_rss - total_after) / (1024**3)
                if log_before_after:
                    msg_gc = (f"[replay] memory_purge after_gc total={total_after/(1024**3):.2f}GB freed≈{freed:.2f}GB")
                    if logger:
                        try:
                            logger.log_text(msg_gc)
                        except Exception:
                            pass
                    else:
                        print(msg_gc)
            except Exception:
                pass
        st['last_mem_water_ts'] = now
        if removed > 0:
            summary = (f"[replay] auto_memory_purge ratio={cur_usage_ratio:.2f} removed={removed} cur={new_size} "
                       f"cap={capacity} parent={parent_rss/(1024**3):.2f}GB children={children_rss/(1024**3):.2f}GB high={high_ratio:.2f} low={low_ratio:.2f}")
            if logger:
                try:
                    logger.log_text(summary)
                except Exception:
                    pass
            else:
                print(summary)
        return st
    except Exception:
        # Never raise from monitor utility
        return st
