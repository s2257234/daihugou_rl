from __future__ import annotations
from __future__ import annotations
from collections import deque
import threading
from typing import Dict, Any, List, Optional
import joblib


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
        self.maxlen = maxlen
        self.default_path = path
        self._lock = threading.RLock()

    # ---------------- Basic ops ----------------
    def append(self, sample: Dict[str, Any]) -> int:
        if not isinstance(sample, dict):
            return -1
        with self._lock:
            if len(self._data) == self.maxlen:
                evicted = self._data.popleft()
                if isinstance(evicted, dict):
                    try:
                        evicted["in_buffer"] = False
                    except Exception:
                        pass
            sample["uid"] = self._next_id
            self._next_id += 1
            sample["in_buffer"] = True
            self._data.append(sample)
            return sample["uid"]

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

    def clear(self):
        with self._lock:
            for s in self._data:
                if isinstance(s, dict):
                    try:
                        s["in_buffer"] = False
                    except Exception:
                        pass
            self._data.clear()

    # ---------------- Persistence ----------------
    def save(self, path: str, purge: bool = False):
        with self._lock:
            self._save_locked(path, purge)

    def _save_locked(self, path: str, purge: bool = False):
        import sys, traceback
        allow_keys = {"player_id", "state", "pi", "value", "model_version", "feature_version", "legal_actions", "value_pred", "uid", "pi_q", "pi_format", "legal_ids", "actions_format", "value_u8", "value_pred_u8"}

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
        for idx, s in enumerate(self._data):
            if not isinstance(s, dict):
                continue
            try:
                d = {k: s.get(k) for k in allow_keys if k in s}
                st = d.get("state")
                if st is not None:
                    d["state"] = _make_state_safe(st)
                safe_list.append(d)
            except RecursionError as e:
                print(f"[WARN] recursion while sanitizing sample idx={idx}: {e}")
                print("[WARN] sample keys=", list(s.keys()))
                continue
            except Exception:
                continue

        payload = {"maxlen": self.maxlen, "next_id": self._next_id, "data": safe_list}
        orig_limit = sys.getrecursionlimit()
        if orig_limit < 5000:
            try:
                sys.setrecursionlimit(5000)
            except Exception:
                pass
        try:
            joblib.dump(payload, path, compress=3)
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
            if not hasattr(self, '_recursion_first'):
                print(f"[WARN] replay save recursion error (compress=3): {e}")
                import traceback as _tb
                tb = ''.join(_tb.format_exc()[-2000:])
                print(f"[WARN] traceback tail:\n{tb}")
                self._recursion_first = True
            try:
                joblib.dump(payload, path, compress=0)
                return
            except RecursionError:
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
                    joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": fallback}, path, compress=0)
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
        obj = joblib.load(path)
        maxlen = obj.get("maxlen") or obj.get("buffer_size") or 50000
        rb = cls(maxlen=maxlen)
        data_list = obj.get("data")
        if data_list is None and isinstance(obj, list):
            data_list = obj
        if not data_list:
            return rb
        for s in data_list:
            if "uid" not in s:
                s["uid"] = rb._next_id
            rb._next_id = max(rb._next_id, s["uid"] + 1)
            rb._data.append(s)
        return rb


__all__ = ["ReplayBuffer"]
