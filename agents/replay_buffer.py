from __future__ import annotations
from __future__ import annotations
from collections import deque
import threading
from typing import Dict, Any, List, Optional
import joblib
import os
import time
import shutil
import numpy as _np


class ReplayBuffer:
    """共有リプレイバッファ (全エージェント共通)。

    Concurrency notes (thread-level):
        - append/save/clear 操作は内部 RLock で直列化。
        - 読み出し (iter_all/sample) はロック下でスナップショット(list) を取得し、
          その後ロックを解放してから yield / random.sample を行うため purge/clear と競合しない。
        - save(purge=True) はロック保持中に self._data を安全化 & 保存し、成功後に clear()。
        - プロセス間共有(multiprocessing) の完全整合性は対象外 (必要なら file lock 等を追加)。

    Race avoidance policy:
        - Trainer スレッドのみが save(purge=True) を呼ぶ想定。エージェント側では
          config.trainer_only_replay_save=True かつ is_trainer_process=False の場合 save をスキップ。
    """

    def __init__(self, maxlen: int, path: Optional[str] = None):
        self._data = deque(maxlen=maxlen)  # type: deque[dict[str, Any]]
        self._next_id = 0
        # prioritized replay support: uid -> priority (float)
        self._priorities: Dict[int, float] = {}
        self.maxlen = maxlen
        self.default_path = path
        self._lock = threading.RLock()
        # --- Async append support ---
        self._async_queue = None  # type: Optional['deque']
        self._async_thread = None  # type: Optional[threading.Thread]
        self._async_enabled = False
        self._async_dropped = 0
        self._async_processed = 0
        self._async_shutdown = False
        # --- On-Full cyclical resize support ---
        self._cycle_on_full = False
        self._cycle_high = None  # type: Optional[int]
        self._cycle_low = None   # type: Optional[int]
        self._cycle_mode = "keep_newest"
        # cycle logging callback (optional): callable(str)
        self._cycle_log = None

        # --- Preallocated contiguous storage (lazy init) ---
        # full_input arrays are large; move them out of per-sample dicts to a float16 matrix
        # Start inactive and enable on first sample that contains full_input.
        self._prealloc_active = False
        self._prealloc_dim = None  # type: Optional[int]
        self._full_input_arr = None  # type: Optional[_np.ndarray]
        self._full_input_occupancy = None  # type: Optional[_np.ndarray]
        self._prealloc_write_pos = 0
        # Parallel arrays for value/value_pred (float16) to avoid per-dict python overhead
        self._values_arr = None  # type: Optional[_np.ndarray]
        self._value_pred_arr = None  # type: Optional[_np.ndarray]
        # pi_q (variable length) cannot be easily packed; keep as list of arrays per slot
        self._pi_q_slots: List[Optional[_np.ndarray]] = []
        # Backup (.bak) save toggle (default True). Trainer/config から無効化可能。
        self._backup_enabled = True

    def set_backup_enabled(self, enabled: bool):
        """Enable/disable creation of .bak backup files during save.

        False にすると atomic save (tmp -> replace) のみ行い、.bak を生成しません。
        破損リスクを最小限にしたい場合は True のままを推奨。"""
        self._backup_enabled = bool(enabled)

    # -------------- Internal helpers (preallocation) --------------
    def _maybe_init_prealloc(self, sample: Dict[str, Any]):
        if self._prealloc_active:
            return
        try:
            st = sample.get('state') or {}
            fi = st.get('full_input')
            if isinstance(fi, _np.ndarray) and fi.ndim == 1:
                dim = int(fi.shape[0])
                self._prealloc_dim = dim
                # Allocate contiguous float16 arrays
                self._full_input_arr = _np.zeros((self.maxlen, dim), dtype=_np.float16)
                self._full_input_occupancy = _np.zeros((self.maxlen,), dtype=_np.bool_)
                self._values_arr = _np.zeros((self.maxlen,), dtype=_np.float16)
                self._value_pred_arr = _np.zeros((self.maxlen,), dtype=_np.float16)
                self._pi_q_slots = [None] * self.maxlen
                self._prealloc_active = True
        except Exception:
            pass

    def _store_into_prealloc(self, sample: Dict[str, Any]):
        if not self._prealloc_active:
            return
        slot = self._prealloc_write_pos
        self._prealloc_write_pos = (self._prealloc_write_pos + 1) % self.maxlen
        try:
            st = sample.get('state') or {}
            fi = st.get('full_input')
            if isinstance(fi, _np.ndarray) and fi.ndim == 1 and fi.shape[0] == self._prealloc_dim:
                # store float16 version
                if fi.dtype != _np.float16:
                    fi16 = fi.astype(_np.float16)
                else:
                    fi16 = fi
                self._full_input_arr[slot] = fi16[:self._prealloc_dim]
                self._full_input_occupancy[slot] = True
                # remove heavy array from per-sample dict
                st['full_input'] = None
                st['full_input_slot'] = slot
                st['full_input_dim'] = self._prealloc_dim
                st['full_input_dtype'] = 'float16'
            # value/value_pred (already quantized upstream to value_u8 / value_pred_u8, but keep original if present)
            val = sample.get('value')
            if isinstance(val, (int, float)):
                self._values_arr[slot] = _np.float16(val)
            else:
                self._values_arr[slot] = _np.float16(0.0)
            vpred = sample.get('value_pred')
            if isinstance(vpred, (int, float)):
                self._value_pred_arr[slot] = _np.float16(vpred)
            else:
                self._value_pred_arr[slot] = _np.float16(0.0)
            # pi_q (np.ndarray uint16) optional
            pi_q = sample.get('pi_q')
            if isinstance(pi_q, _np.ndarray):
                self._pi_q_slots[slot] = pi_q.astype(pi_q.dtype, copy=True)
                # remove reference from dict to avoid duplication
                sample['pi_q_slot'] = slot
            # mark eviction if overwriting occupied slot: we cannot easily clear old dict; acceptable.
        except Exception:
            pass

    # ---------------- Basic ops ----------------
    def append(self, sample: Dict[str, Any]) -> int:
        if not isinstance(sample, dict):
            return -1
        with self._lock:
            # Lazy preallocation if first time we see a state.full_input
            try:
                self._maybe_init_prealloc(sample)
            except Exception:
                pass
            # サイズサイクル: 満杯になったタイミングで high<->low を切替
            try:
                if self._cycle_on_full and len(self._data) == self.maxlen:
                    if (self._cycle_high is not None and self._cycle_low is not None):
                        cur_len = len(self._data)
                        if self.maxlen == self._cycle_high and self._cycle_low > 0:
                            # 高→低（古い分布を間引く）
                            old = int(self.maxlen)
                            new = int(self._cycle_low)
                            removed = self.resize_maxlen(new, shrink_mode=self._cycle_mode)
                            if self._cycle_log:
                                try:
                                    self._cycle_log(f"[replay-cycle] high->low {old}->{new} removed={removed} at_len={cur_len}")
                                except Exception:
                                    pass
                        elif self.maxlen == self._cycle_low and self._cycle_high > 0:
                            # 低→高（再度蓄積フェーズへ）
                            old = int(self.maxlen)
                            new = int(self._cycle_high)
                            _ = self.resize_maxlen(new, shrink_mode=self._cycle_mode)
                            if self._cycle_log:
                                try:
                                    self._cycle_log(f"[replay-cycle] low->high {old}->{new} at_len={cur_len}")
                                except Exception:
                                    pass
            except Exception:
                pass
            if len(self._data) == self.maxlen:
                evicted = self._data.popleft()
                if isinstance(evicted, dict):
                    try:
                        evicted["in_buffer"] = False
                    except Exception:
                        pass
            sample["uid"] = self._next_id
            self._next_id += 1
            # initialize priority: if provided use it, else use max existing or 1.0
            try:
                if 'priority' in sample and isinstance(sample['priority'], (int, float)):
                    p = float(sample['priority'])
                else:
                    p = max(self._priorities.values()) if self._priorities else 1.0
            except Exception:
                p = 1.0
            sample['priority'] = float(p)
            try:
                self._priorities[sample['uid']] = float(p)
            except Exception:
                pass
            sample["in_buffer"] = True
            # Preallocated storage path
            if self._prealloc_active:
                self._store_into_prealloc(sample)
            self._data.append(sample)
            return sample["uid"]

    # ---------------- Async Append ----------------
    def enable_async(self, max_queue: int = 50000):
        """Enable asynchronous append using an internal deque as a queue.

        Non-blocking: if queue full -> drop (oldest or newest policy configurable upstream).
        """
        if self._async_enabled:
            return
        from collections import deque as _dq
        self._async_queue = _dq(maxlen=max_queue)  # raw deque for low overhead
        self._async_shutdown = False
        def _worker():
            while not self._async_shutdown:
                try:
                    item = None
                    try:
                        item = self._async_queue.popleft()  # type: ignore[attr-defined]
                    except IndexError:
                        # empty -> sleep brief
                        time.sleep(0.0005)
                        continue
                    if item is None:
                        continue
                    self.append(item)
                    self._async_processed += 1
                except Exception:
                    # swallow & continue
                    time.sleep(0.0005)
                    continue
        t = threading.Thread(target=_worker, name="ReplayAsyncAppend", daemon=True)
        t.start()
        self._async_thread = t
        self._async_enabled = True

    def append_async(self, sample: Dict[str, Any], *, drop_oldest: bool = True) -> int:
        if not self._async_enabled or self._async_queue is None:
            return self.append(sample)
        # Fast path: space available
        if len(self._async_queue) < self._async_queue.maxlen:  # type: ignore[attr-defined]
            self._async_queue.append(sample)  # type: ignore[attr-defined]
            return -2  # async pending UID assigned later
        # Queue full
        try:
            if drop_oldest:
                # discard oldest pending to make room
                try:
                    _ = self._async_queue.popleft()  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._async_queue.append(sample)  # type: ignore[attr-defined]
            else:
                # drop new sample
                self._async_dropped += 1
                return -1
        except Exception:
            self._async_dropped += 1
            return -1
        return -2

    def async_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._async_enabled,
            "queue_len": len(self._async_queue) if self._async_queue is not None else 0,
            "queue_cap": self._async_queue.maxlen if self._async_queue is not None else 0,  # type: ignore[attr-defined]
            "processed": self._async_processed,
            "dropped": self._async_dropped,
        }

    def disable_async(self, wait: bool = True):
        if not self._async_enabled:
            return
        self._async_shutdown = True
        if wait and self._async_thread is not None:
            try:
                self._async_thread.join(timeout=1.0)
            except Exception:
                pass
        self._async_enabled = False
        self._async_thread = None
        self._async_queue = None

    # ---------------- Resize / Cyclical Refresh ----------------
    def resize_maxlen(self, new_maxlen: int, *, shrink_mode: str = "keep_newest") -> int:
        """Resize internal capacity. If shrinking, remove oldest items based on mode.

        Returns: number of removed samples (if shrinking) else 0.
        """
        if new_maxlen <= 0:
            new_maxlen = 1
        removed = 0
        with self._lock:
            cur_len = len(self._data)
            old_max = self.maxlen
            if new_maxlen == old_max:
                return 0
            # Rebuild deque with new maxlen
            if new_maxlen < cur_len:
                if shrink_mode == "clear":
                    for s in self._data:
                        if isinstance(s, dict):
                            try: s["in_buffer"] = False
                            except Exception: pass
                    self._data = deque(maxlen=new_maxlen)
                    removed = cur_len
                else:  # keep_newest
                    # keep rightmost newest new_maxlen items
                    keep = list(self._data)[-new_maxlen:]
                    removed = cur_len - len(keep)
                    for s in list(self._data)[:-new_maxlen]:
                        if isinstance(s, dict):
                            try: s["in_buffer"] = False
                            except Exception: pass
                    self._data = deque(keep, maxlen=new_maxlen)
            else:
                # expansion: just wrap existing list into larger deque
                self._data = deque(list(self._data), maxlen=new_maxlen)
            self.maxlen = new_maxlen
        return removed

    # ---------------- Configure on-full cycle ----------------
    def configure_cycle_on_full(self, high_size: int, low_size: int, *, mode: str = "keep_newest"):
        with self._lock:
            self._cycle_high = int(max(1, high_size))
            self._cycle_low = int(max(1, low_size))
            self._cycle_mode = mode or "keep_newest"
            self._cycle_on_full = True

    # optional: hook logger for cycle events
    def set_cycle_logger(self, logger_fn):
        # logger_fn should be callable(str)
        try:
            self._cycle_log = logger_fn if callable(logger_fn) else None
        except Exception:
            self._cycle_log = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._data)

    def iter_all(self, owner_pid: Optional[int] = None):
        with self._lock:
            snapshot = list(self._data)
        if owner_pid is None:
            for s in snapshot:
                yield s
        else:
            for s in snapshot:
                if s.get("player_id") == owner_pid:
                    yield s

    def sample(self, n: int, owner_pid: Optional[int] = None) -> List[Dict[str, Any]]:
        import random
        with self._lock:
            if owner_pid is None:
                pool = list(self._data)
            else:
                pool = [s for s in self._data if s.get("player_id") == owner_pid]
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)
        return random.sample(pool, n)

    # ---------------- Prioritized sampling ----------------
    def sample_prioritized(self, n: int, owner_pid: Optional[int] = None, *, alpha: float = 0.6, eps: float = 1e-6, return_weights: bool = False):
        """Simple prioritized sampling (stochastic, without complex trees).

        - n: number of samples to draw (without replacement when possible)
        - owner_pid: filter by player_id
        - alpha: exponent on priorities (0 = uniform, 1 = full priority)
        - eps: small constant to ensure non-zero priority
        - return_weights: if True, also return importance-sampling weights (unnormalized)

        Returns: samples, uids, weights? (if return_weights True)
        """
        import random as _r
        with self._lock:
            if owner_pid is None:
                pool = list(self._data)
            else:
                pool = [s for s in self._data if s.get("player_id") == owner_pid]
            if not pool:
                return ([], [], []) if return_weights else []
            # assemble priority list in same order
            uids = [s.get('uid') for s in pool]
            prios = []
            for s in pool:
                uid = s.get('uid')
                p = None
                try:
                    if uid is not None and uid in self._priorities:
                        p = float(self._priorities.get(uid, 0.0))
                except Exception:
                    p = None
                if p is None:
                    try:
                        p = float(s.get('priority', 0.0) or 0.0)
                    except Exception:
                        p = 0.0
                prios.append(max(p, eps))
            # apply alpha
            try:
                weights = [p ** float(alpha) for p in prios]
            except Exception:
                weights = [max(float(v), eps) for v in prios]
            total = float(sum(weights))
            if total <= 0:
                # fallback uniform
                probs = [1.0 / len(weights)] * len(weights)
            else:
                probs = [w / total for w in weights]
            # sample without replacement by cumulative selection (simple, sufficient for small batches)
            idxs = []
            if n >= len(pool):
                idxs = list(range(len(pool)))
            else:
                cumulative = []
                s = 0.0
                for p in probs:
                    s += p
                    cumulative.append(s)
                selected = set()
                tries = 0
                while len(idxs) < n and tries < n * 10:
                    r = _r.random() * cumulative[-1]
                    # linear search (OK for small pool/batch)
                    j = 0
                    while j < len(cumulative) and cumulative[j] < r:
                        j += 1
                    if j >= len(cumulative):
                        j = len(cumulative) - 1
                    if j in selected:
                        tries += 1
                        continue
                    selected.add(j)
                    idxs.append(j)
                    tries = 0
                # if failed to get enough (rare), fill uniformly
                if len(idxs) < n:
                    rest = [i for i in range(len(pool)) if i not in selected]
                    _r.shuffle(rest)
                    for k in rest[:(n-len(idxs))]:
                        idxs.append(k)
            # assemble outputs in sampled order
            sampled = [pool[i] for i in idxs]
            sampled_uids = [uids[i] for i in idxs]
            # importance-sampling weights (unnormalized): w_i = 1 / (N * P(i))
            is_weights = []
            N = max(1, len(pool))
            for i in idxs:
                p_i = probs[i] if i < len(probs) else (1.0 / N)
                w = 1.0 / (N * max(p_i, 1e-12))
                is_weights.append(w)
            # normalize IS weights to max=1 for stability
            try:
                maxw = max(is_weights) if is_weights else 1.0
                is_weights = [float(w / maxw) for w in is_weights]
            except Exception:
                pass
            if return_weights:
                return sampled, sampled_uids, is_weights
            return sampled

    # ---------------- Fast prioritized sampling with tensor batch ----------------
    def sample_prioritized_fast(self, n: int, owner_pid: Optional[int] = None, *, alpha: float = 0.6, eps: float = 1e-6,
                                 return_weights: bool = False, return_full_input: bool = True):
        """高速優先度付きサンプリング (torch.multinomial 使用)。

        戻り値(dict): {
            'samples': List[Dict[str, Any]],
            'uids': List[int],
            'is_weights': Optional[List[float]],
            'full_input': Optional[numpy.ndarray]  # shape=(B, D) float32
        }
        full_input 行列は prealloc スロットか各サンプルの full_input / full_compact から再構成。
        """
        try:
            import torch
            import numpy as _np
        except Exception:
            # フォールバック: 通常版
            if return_weights:
                sampled, sampled_uids, is_w = self.sample_prioritized(n, owner_pid=owner_pid, alpha=alpha, eps=eps, return_weights=True)
                return {'samples': sampled, 'uids': list(sampled_uids), 'is_weights': list(is_w), 'full_input': None}
            sampled = self.sample_prioritized(n, owner_pid=owner_pid, alpha=alpha, eps=eps, return_weights=False)
            return {'samples': sampled, 'uids': [s.get('uid') for s in sampled], 'is_weights': None, 'full_input': None}

        with self._lock:
            if owner_pid is None:
                pool = list(self._data)
            else:
                pool = [s for s in self._data if s.get('player_id') == owner_pid]
            if not pool:
                return {'samples': [], 'uids': [], 'is_weights': [] if return_weights else None, 'full_input': None}
            prios = []
            uids = []
            for s in pool:
                uid = s.get('uid')
                uids.append(uid)
                p = None
                try:
                    if uid is not None and uid in self._priorities:
                        p = float(self._priorities.get(uid, 1.0))
                except Exception:
                    p = None
                if p is None:
                    p = 1.0
                prios.append(max(float(p), eps))
            prios_t = torch.tensor(prios, dtype=torch.float32)
            prealloc_dim = self._prealloc_dim
            full_input_arr_snapshot = None
            if prealloc_dim is not None and self._full_input_arr is not None:
                full_input_arr_snapshot = self._full_input_arr
        try:
            weights_t = prios_t.pow(float(alpha))
        except Exception:
            weights_t = prios_t.clone()
        total = float(weights_t.sum().item())
        if total <= 0:
            probs_t = torch.full_like(weights_t, 1.0 / float(weights_t.numel()))
        else:
            probs_t = weights_t / total
        k = min(int(n), probs_t.numel())
        try:
            idxs_t = torch.multinomial(probs_t, k, replacement=False)
        except Exception:
            idxs_t = torch.randperm(probs_t.numel())[:k]
        idxs = idxs_t.tolist()
        sampled = [pool[i] for i in idxs]
        sampled_uids = [uids[i] for i in idxs]

        is_w_list = None
        if return_weights:
            N = max(1, probs_t.numel())
            sel_probs = probs_t[idxs_t].detach().cpu().tolist()
            raw_ws = [1.0 / (N * max(p, 1e-12)) for p in sel_probs]
            mw = max(raw_ws) if raw_ws else 1.0
            is_w_list = [float(w / mw) for w in raw_ws]

        full_arr = None
        if return_full_input and sampled:
            dim = None
            if prealloc_dim is not None:
                dim = int(prealloc_dim)
            if dim is None:
                for s in sampled:
                    st = s.get('state') or {}
                    fi = st.get('full_input')
                    if fi is not None and hasattr(fi, 'shape') and getattr(fi, 'ndim', 1) == 1:
                        try:
                            dim = int(fi.shape[0])
                            break
                        except Exception:
                            pass
                    if 'full_compact' in st and isinstance(st.get('full_compact'), dict):
                        cf = st['full_compact']
                        try:
                            bin_len = int(cf.get('binary_len', 0))
                            floats = cf.get('floats')
                            flen = len(floats) if floats is not None else 0
                            dim = bin_len + flen
                            break
                        except Exception:
                            pass
            if dim is not None:
                rows = []
                for s in sampled:
                    st = s.get('state') or {}
                    vec = None
                    slot = st.get('full_input_slot')
                    if slot is not None and full_input_arr_snapshot is not None:
                        try:
                            vec = full_input_arr_snapshot[int(slot)][:dim]
                        except Exception:
                            vec = None
                    if vec is None:
                        fi = st.get('full_input')
                        if fi is not None:
                            try:
                                import numpy as _np
                                vec = _np.asarray(fi, dtype=_np.float32)
                            except Exception:
                                vec = None
                    if vec is None and 'full_compact' in st and isinstance(st.get('full_compact'), dict):
                        cf = st['full_compact']
                        try:
                            import numpy as _np
                            bin_len = int(cf.get('binary_len', 0))
                            packed = cf.get('packed_bits', b'')
                            floats = cf.get('floats')
                            if isinstance(packed, (bytes, bytearray)) and bin_len > 0:
                                bits_arr = _np.unpackbits(_np.frombuffer(packed, dtype=_np.uint8))[:bin_len].astype(_np.float32)
                            else:
                                bits_arr = _np.zeros(bin_len, dtype=_np.float32)
                            if floats is not None:
                                try:
                                    floats_arr = _np.asarray(floats, dtype=_np.float16).astype(_np.float32)
                                except Exception:
                                    floats_arr = _np.asarray(list(floats), dtype=_np.float32)
                            else:
                                floats_arr = _np.zeros(0, dtype=_np.float32)
                            vec = _np.concatenate([bits_arr, floats_arr])
                        except Exception:
                            vec = None
                    if vec is None:
                        import numpy as _np
                        vec = _np.zeros(dim, dtype=_np.float32)
                    if vec.shape[0] != dim:
                        import numpy as _np
                        if vec.shape[0] < dim:
                            pad = _np.zeros(dim, dtype=_np.float32)
                            pad[:vec.shape[0]] = vec
                            vec = pad
                        else:
                            vec = vec[:dim]
                    rows.append(vec.astype(_np.float32, copy=False))
                try:
                    import numpy as _np
                    full_arr = _np.stack(rows, axis=0)
                except Exception:
                    full_arr = None

        return {
            'samples': sampled,
            'uids': sampled_uids,
            'is_weights': is_w_list if return_weights else None,
            'full_input': full_arr
        }

    def clear(self):
        with self._lock:
            for s in self._data:
                if isinstance(s, dict):
                    try:
                        s["in_buffer"] = False
                    except Exception:
                        pass
            self._data.clear()

    # ---------------- High/Low Water Support ----------------
    def shrink_to_size(self, target_size: int) -> int:
        """古いサンプルから削り target_size 以下に縮小。

        Returns: 削除した件数
        """
        if target_size < 0:
            target_size = 0
        removed = 0
        with self._lock:
            cur = len(self._data)
            if cur <= target_size:
                return 0
            need = cur - target_size
            for _ in range(need):
                try:
                    ev = self._data.popleft()
                    if isinstance(ev, dict):
                        try:
                            ev["in_buffer"] = False
                        except Exception:
                            pass
                    removed += 1
                except Exception:
                    break
        return removed

    # ---------------- Persistence ----------------
    def save(self, path: str, purge: bool = False):
        with self._lock:
            self._save_locked(path, purge)

    def _save_locked(self, path: str, purge: bool = False):
        import sys, traceback
        # allow_keys から "pi" と "legal_actions" を除外し、量子化済み/ID化済みの最小構造のみを保存する。
        # これにより再帰的な複雑構造の混入 (特に raw pi / backup legal_actions による巨大ネスト) リスクを下げる。
        allow_keys = {
            "player_id", "state", "value", "model_version", "feature_version", "value_pred",
            "uid", "pi_q", "pi_format", "legal_ids", "actions_format", "value_u8", "value_pred_u8",
            "split"
        }

        # Contiguous pack format: build arrays on the fly for compact saving
        try:
            snapshot = list(self._data)
            n = len(snapshot)
            # determine feature dimension from either per-sample full_input or prealloc slots
            dim = None
            if n > 0:
                for s in snapshot:
                    st0 = (s.get('state') if isinstance(s, dict) else None) or {}
                    fi0 = st0.get('full_input')
                    if isinstance(fi0, _np.ndarray) and fi0.ndim == 1:
                        dim = int(fi0.shape[0])
                        break
                    slot0 = st0.get('full_input_slot')
                    if (slot0 is not None) and (self._full_input_arr is not None):
                        try:
                            dim = int(self._full_input_arr.shape[1])
                            break
                        except Exception:
                            pass
            if dim is not None and n > 0:
                full_input_arr = _np.zeros((n, dim), dtype=_np.float16)
                values_arr = _np.zeros((n,), dtype=_np.float16)
                value_pred_arr = _np.zeros((n,), dtype=_np.float16)
                pi_q_slots: List[Optional[_np.ndarray]] = [None] * n
                meta_list: List[Dict[str, Any]] = []
                for i, s in enumerate(snapshot):
                    if not isinstance(s, dict):
                        continue
                    d = {k: s.get(k) for k in allow_keys if k in s}
                    st = s.get('state') or {}
                    # choose source of full_input: direct array or prealloc slot
                    fi = st.get('full_input')
                    if not (isinstance(fi, _np.ndarray) and fi.ndim == 1):
                        slot = st.get('full_input_slot')
                        if (slot is not None) and (self._full_input_arr is not None):
                            try:
                                fi = self._full_input_arr[int(slot)]
                            except Exception:
                                fi = None
                    if isinstance(fi, _np.ndarray) and fi.ndim == 1:
                        full_input_arr[i] = (fi.astype(_np.float16, copy=False) if fi.dtype == _np.float16 else fi.astype(_np.float16))[:dim]
                        st_light = {
                            'array_idx': i,
                            'full_input_dim': dim,
                            'full_input_dtype': 'float16'
                        }
                        for mk in ('hand_size', 'field_size', 'turn', 'full_input_len'):
                            if mk in st:
                                st_light[mk] = st[mk]
                        d['state'] = st_light
                    # values: prefer dict value, else fallback to prealloc slot if available
                    try:
                        v = s.get('value')
                        if v is None:
                            slot = st.get('full_input_slot')
                            if (slot is not None) and (self._values_arr is not None):
                                v = float(self._values_arr[int(slot)])
                        values_arr[i] = _np.float16(float(v)) if v is not None else _np.float16(0.0)
                    except Exception:
                        values_arr[i] = _np.float16(0.0)
                    try:
                        vp = s.get('value_pred')
                        if vp is None:
                            slot = st.get('full_input_slot')
                            if (slot is not None) and (self._value_pred_arr is not None):
                                vp = float(self._value_pred_arr[int(slot)])
                        value_pred_arr[i] = _np.float16(float(vp)) if vp is not None else _np.float16(0.0)
                    except Exception:
                        value_pred_arr[i] = _np.float16(0.0)
                    # pi_q slot: use per-sample if present else from slot array
                    pq = s.get('pi_q')
                    if not isinstance(pq, _np.ndarray):
                        pq_slot = s.get('pi_q_slot')
                        if pq_slot is None:
                            pq_slot = st.get('full_input_slot')
                        if (pq_slot is not None) and self._pi_q_slots:
                            try:
                                pq = self._pi_q_slots[int(pq_slot)]
                            except Exception:
                                pq = None
                    if isinstance(pq, _np.ndarray):
                        pi_q_slots[i] = pq.astype(pq.dtype, copy=True)
                        d['pi_q_slot'] = i
                    meta_list.append(d)
                payload = {
                    'maxlen': self.maxlen,
                    'next_id': self._next_id,
                    'prealloc_dim': int(dim),
                    'full_input': full_input_arr,
                    'values': values_arr,
                    'value_pred': value_pred_arr,
                    'pi_q_slots': pi_q_slots,
                    'samples': meta_list,
                }
                tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
                bak_path = f"{path}.bak"
                try:
                    if self._backup_enabled and os.path.exists(path):
                        try:
                            shutil.copy2(path, bak_path)
                        except Exception:
                            pass
                    joblib.dump(payload, tmp_path, compress=3)
                    try:
                        os.replace(tmp_path, path)
                    except Exception:
                        shutil.move(tmp_path, path)
                    if purge:
                        for s in self._data:
                            if isinstance(s, dict):
                                try:
                                    s['in_buffer'] = False
                                except Exception:
                                    pass
                        self._data.clear()
                    return
                except Exception as e:
                    print(f"[WARN] replay contiguous save failed, falling back: {e}")
                    # fall through to legacy path
        except Exception:
            pass

        # 簡易ネスト深さ推定 (dict/list/tuple のみ辿る)。深すぎる場合は後でログに残す。
        def _approx_depth(o, max_depth: int = 40):
            stack = [(o, 1)]
            md = 0
            seen = set()
            try:
                while stack:
                    obj, d = stack.pop()
                    if d > md:
                        md = d
                    if d >= max_depth:
                        return md, True
                    oid = id(obj)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    if isinstance(obj, dict):
                        for v in obj.values():
                            if isinstance(v, (dict, list, tuple)):
                                stack.append((v, d + 1))
                    elif isinstance(obj, (list, tuple)):
                        for v in obj:
                            if isinstance(v, (dict, list, tuple)):
                                stack.append((v, d + 1))
                return md, False
            except Exception:
                return md, False

        def _make_state_safe(st):
            if not isinstance(st, dict):
                return None
            st_safe: Dict[str, Any] = {}
            if "full_input" in st:
                fi = st["full_input"]
                orig_len = None
                try:  # compress to float16
                    import numpy as _np
                    if isinstance(fi, _np.ndarray):
                        orig_len = fi.shape[0]
                        if fi.dtype != _np.float16:
                            try:
                                fi = fi.astype(_np.float16)
                            except Exception:
                                pass
                        arr16 = fi
                    elif isinstance(fi, (list, tuple)):
                        orig_len = len(fi)
                        try:
                            arr = _np.asarray(fi, dtype=_np.float32)
                            arr16 = arr.astype(_np.float16)
                        except Exception:
                            arr16 = _np.asarray(list(fi), dtype=_np.float16)
                    else:
                        arr16 = None
                except Exception:
                    arr16 = None
                    orig_len = len(fi) if isinstance(fi, (list, tuple)) else None
                if arr16 is not None:
                    if getattr(arr16, 'shape', [0])[0] > 5000:  # safety crop
                        arr16 = arr16[:5000]
                    st_safe["full_input"] = arr16
                    if orig_len is not None:
                        st_safe["full_input_len"] = int(orig_len)
                    st_safe["full_input_dtype"] = "float16"
            for mk in ("hand_size", "field_size", "turn", "full_input_dim"):
                if mk in st:
                    st_safe[mk] = st[mk]
            return st_safe

        safe_list: List[Dict[str, Any]] = []
        deepest = (0, None)  # (depth, uid)
        for idx, s in enumerate(self._data):
            if not isinstance(s, dict):
                continue
            try:
                d = {k: s.get(k) for k in allow_keys if k in s}
                st = d.get("state")
                if st is not None:
                    d["state"] = _make_state_safe(st)
                # ネスト深さ診断 (state 以外も含む) ※コスト低なので毎回
                depth, clipped = _approx_depth(d)
                if depth > deepest[0]:
                    deepest = (depth, d.get("uid"))
                if clipped:
                    # 深さが閾値超え -> state を更に縮約 (full_input だけ残し他キー削減)
                    try:
                        st2 = d.get("state") or {}
                        if isinstance(st2, dict):
                            ks = {"full_input", "full_input_len", "full_input_dtype"}
                            d["state"] = {k: v for k, v in st2.items() if k in ks}
                    except Exception:
                        pass
                safe_list.append(d)
            except RecursionError as e:
                print(f"[WARN] recursion while sanitizing sample idx={idx}: {e}")
                print("[WARN] sample keys=", list(s.keys()))
                continue
            except Exception:
                continue

        # 追加の安全策: 異常に深い場合は最後に通知 (初回のみ表示) & 深さ>50なら shallow モード再生成
        if deepest[0] > 50:
            if not hasattr(self, '_warned_deep_sample'):
                print(f"[WARN] replay save: detected deep nested sample depth={deepest[0]} uid={deepest[1]} -> shallow sanitizing")
                self._warned_deep_sample = True
            new_list = []
            for d in safe_list:
                depth, clipped = _approx_depth(d)
                if depth > 50:
                    try:
                        # shallow: state を完全除去 (再学習には pi_q / value 系で十分)
                        d2 = {k: v for k, v in d.items() if k != 'state'}
                        new_list.append(d2)
                    except Exception:
                        new_list.append(d)
                else:
                    new_list.append(d)
            safe_list = new_list

        payload = {"maxlen": self.maxlen, "next_id": self._next_id, "data": safe_list}
        orig_limit = sys.getrecursionlimit()
        if orig_limit < 5000:
            try:
                sys.setrecursionlimit(5000)
            except Exception:
                pass
        try:
            # --- Atomic save with backup to mitigate partial/EOF reads ---
            tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
            bak_path = f"{path}.bak"
            try:
                if self._backup_enabled and os.path.exists(path):
                    shutil.copy2(path, bak_path)
            except Exception:
                pass
            joblib.dump(payload, tmp_path, compress=3)
            try:
                os.replace(tmp_path, path)
            except Exception:
                shutil.move(tmp_path, path)
            if purge:
                for s in self._data:
                    if isinstance(s, dict):
                        try:
                            s["in_buffer"] = False
                        except Exception:
                            pass
                self._data.clear()
            return
        except RecursionError as e:
            # First, log once and try no-compression atomic save
            if not hasattr(self, '_recursion_first'):
                print(f"[WARN] replay save recursion error (compress=3): {e}")
                import traceback as _tb
                tb = ''.join(_tb.format_exc()[-2000:])
                print(f"[WARN] traceback tail:\n{tb}")
                self._recursion_first = True
            try:
                tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
                joblib.dump(payload, tmp_path, compress=0)
                try:
                    os.replace(tmp_path, path)
                except Exception:
                    shutil.move(tmp_path, path)
                return
            except RecursionError:
                pass
            # Ultra-minimal: strip state and retry
            try:
                minimal = []
                for d in safe_list:
                    d2 = {k: v for k, v in d.items() if k != 'state'}
                    minimal.append(d2)
                tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
                joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": minimal}, tmp_path, compress=0)
                try:
                    os.replace(tmp_path, path)
                except Exception:
                    shutil.move(tmp_path, path)
                print(f"[WARN] ultra-minimal replay saved (state stripped) samples={len(minimal)}")
                if purge:
                    for s in self._data:
                        if isinstance(s, dict):
                            try:
                                s["in_buffer"] = False
                            except Exception:
                                pass
                    self._data.clear()
                return
            except Exception:
                pass
            # Identify problematic portion and save fallback subset
            def _can_dump(sub):
                try:
                    joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": sub}, path + '.probe', compress=0)
                    return True
                except RecursionError:
                    return False
                except Exception:
                    return True
            lo, hi = 0, len(safe_list)
            attempts = 0
            while lo < hi and attempts < 10:
                mid = (lo + hi) // 2
                if _can_dump(safe_list[:mid]):
                    lo = mid + 1
                else:
                    hi = mid
                attempts += 1
            bad_idx = lo - 1 if lo <= len(safe_list) else None
            if bad_idx is not None and 0 <= bad_idx < len(safe_list):
                print(f"[WARN] suspect sample causing recursion idx={bad_idx} (will exclude & fallback)")
                try:
                    del safe_list[bad_idx]
                except Exception:
                    pass
            fallback = safe_list[-1000:] if len(safe_list) > 1000 else safe_list
            try:
                tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
                joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": fallback}, tmp_path, compress=0)
                try:
                    os.replace(tmp_path, path)
                except Exception:
                    shutil.move(tmp_path, path)
                print(f"[WARN] fallback replay saved with {len(fallback)}/{len(safe_list)} samples")
                if purge:
                    for s in self._data:
                        if isinstance(s, dict):
                            try:
                                s["in_buffer"] = False
                            except Exception:
                                pass
                    self._data.clear()
                return
            except Exception as ee2:
                print(f"[ERROR] replay save ultimate fallback failed: {ee2}")
        except Exception as e:
            if not hasattr(self, '_warned_save'):
                print(f"[WARN] replay save failed once: {e}")
                self._warned_save = True
        finally:
            try:
                if orig_limit and orig_limit != sys.getrecursionlimit():
                    sys.setrecursionlimit(orig_limit)
            except Exception:
                pass

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        def _try_load(p: str):
            return joblib.load(p)
        obj = None
        last_err = None
        try:
            obj = _try_load(path)
        except Exception as e:
            last_err = e
            # Attempt to recover from backup
            bak = f"{path}.bak"
            try:
                if os.path.exists(bak):
                    print(f"[WARN] replay load failed from '{path}': {e} -> trying backup '{bak}'")
                    obj = _try_load(bak)
            except Exception as e2:
                last_err = e2
                obj = None
        if obj is None:
            # Final fallback: rename corrupt file and return empty buffer
            try:
                if os.path.exists(path):
                    corrupt = f"{path}.corrupt.{int(time.time())}"
                    try:
                        os.replace(path, corrupt)
                    except Exception:
                        shutil.move(path, corrupt)
                    print(f"[WARN] replay file moved to '{corrupt}' due to load error: {last_err}")
            except Exception:
                pass
            # Return empty buffer with default maxlen
            maxlen_guess = 50000
            return cls(maxlen=maxlen_guess)

        maxlen = obj.get("maxlen") or obj.get("buffer_size") or 50000
        rb = cls(maxlen=maxlen)
        # New preallocated format
        if isinstance(obj, dict) and ('samples' in obj) and ('full_input' in obj):
            try:
                rb._prealloc_dim = int(obj.get('prealloc_dim') or 0)
            except Exception:
                rb._prealloc_dim = 0
            try:
                rb._full_input_arr = obj.get('full_input')
                rb._values_arr = obj.get('values')
                rb._value_pred_arr = obj.get('value_pred')
                rb._pi_q_slots = obj.get('pi_q_slots') or [None] * maxlen
                rb._full_input_occupancy = _np.zeros((maxlen,), dtype=_np.bool_)
                rb._prealloc_active = True
            except Exception:
                # if arrays missing, fall back to legacy branch below
                rb._prealloc_active = False
            samples = obj.get('samples') or []
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if 'uid' not in s:
                    s['uid'] = rb._next_id
                rb._next_id = max(rb._next_id, int(s['uid']) + 1)
                # mark occupancy if slot present
                try:
                    st = s.get('state') or {}
                    # reconstruct full_input from array index
                    idx = st.get('array_idx', None)
                    if (idx is not None) and (rb._full_input_arr is not None) and 0 <= int(idx) < rb._full_input_arr.shape[0]:
                        st['full_input'] = rb._full_input_arr[int(idx)]
                        rb._full_input_occupancy[int(idx)] = True
                    s['state'] = st
                    # reconstruct pi_q if slot marker present
                    pq_idx = s.get('pi_q_slot', None)
                    if pq_idx is not None and rb._pi_q_slots is not None:
                        try:
                            s['pi_q'] = rb._pi_q_slots[int(pq_idx)]
                        except Exception:
                            pass
                except Exception:
                    pass
                rb._data.append(s)
            try:
                nxt = int(obj.get('next_id') or rb._next_id)
                rb._next_id = max(rb._next_id, nxt)
            except Exception:
                pass
            return rb
        # Legacy list/dict format
        data_list = obj.get("data")
        if data_list is None and isinstance(obj, list):
            data_list = obj
        if not data_list:
            return rb
        for s in data_list:
            if "uid" not in s:
                s["uid"] = rb._next_id
            rb._next_id = max(rb._next_id, s["uid"] + 1)
            # restore priority if present
            try:
                if isinstance(s, dict) and 'priority' in s:
                    rb._priorities[s['uid']] = float(s.get('priority', 1.0) or 1.0)
            except Exception:
                pass
            rb._data.append(s)
        return rb

    def update_priorities(self, uid_to_priority: Dict[int, float]):
        """Update stored priorities for given uids (atomic under lock)."""
        with self._lock:
            for uid, p in uid_to_priority.items():
                try:
                    self._priorities[int(uid)] = float(max(p, 0.0))
                except Exception:
                    continue
            # also reflect into sample dicts if present
            try:
                for s in self._data:
                    if not isinstance(s, dict):
                        continue
                    uid = s.get('uid')
                    if uid is None:
                        continue
                    if uid in self._priorities:
                        s['priority'] = float(self._priorities[uid])
            except Exception:
                pass

    @property
    def has_prioritized(self) -> bool:
        return True


__all__ = ["ReplayBuffer"]
