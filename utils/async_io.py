from __future__ import annotations
"""Async IO process for heavy serialization (torch.save / joblib.dump).

設計:
  - 別プロセス (multiprocessing.Process) を起動し、(op, args) を Queue で受信して実行。
  - メイン側は enqueue して即 return し、I/O 待ちでブロックしない。
  - 終了時 flush(): SENTINEL を送り join。timeout 超過で強制 terminate (安全側)。
  - ジョブ形式:
        {"op": "torch_save", "path": str, "object": obj, "kwargs": {...}}
        {"op": "joblib_dump", "path": str, "object": obj, "kwargs": {...}}
  - Python オブジェクトは Queue で pickle 経由転送 (このオーバーヘッドよりも大型 I/O 待ち削減効果を優先)。

注意:
  - torch.save で map_location は不要 (保存のみ) のため kwargs で受理。
  - joblib.compress パラメータ等も kwargs に含められる。
  - 失敗時は標準出力へ WARN を 1 行出し継続。
"""
import multiprocessing as mp
import time, os, traceback
from typing import Any, Dict, Optional

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_SENTINEL = {"op": "__STOP__"}

class AsyncIOProcess:
    def __init__(self, maxsize: int = 32, log_events: bool = False, warn_queue_full: bool = True):
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=maxsize)
        self._proc: Optional[mp.Process] = None
        self._started = False
        self._warned_full = False
        self._log_events = log_events
        self._warn_queue_full = warn_queue_full
        self._jobs_enqueued = 0
        self._jobs_finished = 0
        self._last_event_ts = 0.0

    def start(self):
        if self._started:
            return
        self._proc = self._ctx.Process(target=self._worker, args=(self._queue, self._log_events))
        self._proc.daemon = True
        self._proc.start()
        self._started = True

    def enqueue_torch_save(self, obj: Any, path: str, **kwargs):
        self._enqueue({"op": "torch_save", "object": obj, "path": path, "kwargs": kwargs})

    def enqueue_joblib_dump(self, obj: Any, path: str, **kwargs):
        self._enqueue({"op": "joblib_dump", "object": obj, "path": path, "kwargs": kwargs})

    def enqueue_text_write(self, text: str, path: str, **kwargs):
        """Enqueue a text write operation that will atomically write the provided
        text to `path` (via tmp -> fsync -> replace) in the async process.
        """
        self._enqueue({"op": "text_write", "text": text, "path": path, "kwargs": kwargs})

    def _enqueue(self, job: Dict[str, Any]):
        if not self._started:
            self.start()
        try:
            self._queue.put_nowait(job)
            self._jobs_enqueued += 1
        except Exception:
            # queue full -> fallback synchronous
            if self._warn_queue_full and not self._warned_full:
                print(f"[async-io][WARN] queue full -> fallback sync op={job.get('op')} path={job.get('path')}")
                self._warned_full = True
            self._execute_job(job)  # 同期実行 (最悪性能だが安全)

    def flush(self, timeout: float | None = 30.0):
        if not self._started:
            return
        try:
            self._queue.put(_SENTINEL, block=True, timeout=1)
        except Exception:
            pass
        if self._proc is None:
            return
        self._proc.join(timeout=timeout)
        if self._proc.is_alive():  # pragma: no cover (タイムアウト経路)
            try:
                print("[async-io][WARN] flush timeout -> terminate")
                self._proc.terminate()
            except Exception:
                pass
        self._started = False

    # ---------------- Worker ----------------
    @staticmethod
    def _worker(q: mp.Queue, log_events: bool):  # pragma: no cover (別プロセス)
        pid = os.getpid()
        while True:
            try:
                job = q.get()
            except Exception:
                break
            if not isinstance(job, dict):
                continue
            if job.get("op") == "__STOP__":
                break
            AsyncIOProcess._execute_job(job, pid=pid, log_events=log_events)

    @staticmethod
    def _execute_job(job: Dict[str, Any], pid: Optional[int] = None, log_events: bool = False):
        op = job.get("op")
        path = job.get("path")
        kwargs = job.get("kwargs", {}) or {}
        started = time.time()
        ok = True
        err_msg = None
        try:
            if op == "torch_save":
                if torch is None:
                    raise RuntimeError("torch not available in async proc")
                torch.save(job.get("object"), path, **kwargs)
            elif op == "joblib_dump":
                if joblib is None:
                    raise RuntimeError("joblib not available in async proc")
                # Atomic write: dump to a tmp file in same dir, fsync, then replace
                try:
                    dirp = os.path.dirname(path) or '.'
                    os.makedirs(dirp, exist_ok=True)
                    tmp = path + ".tmp"
                    # joblib.dump will create/overwrite tmp
                    joblib.dump(job.get("object"), tmp, **kwargs)
                    # best-effort fsync the file to reduce partial-write visibility
                    try:
                        fd = os.open(tmp, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    except Exception:
                        pass
                    # atomic replace
                    try:
                        os.replace(tmp, path)
                    except Exception:
                        # fallback: shutil.move
                        import shutil
                        shutil.move(tmp, path)
                except Exception:
                    # If atomic path fails, try a direct dump as last resort
                    joblib.dump(job.get("object"), path, **kwargs)
            elif op == "text_write":
                # Atomic text write: write to tmp, fsync, then replace
                try:
                    dirp = os.path.dirname(path) or '.'
                    os.makedirs(dirp, exist_ok=True)
                    tmp = path + ".tmp"
                    # write text file
                    with open(tmp, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
                        f.write(job.get('text') or '')
                    # best-effort fsync
                    try:
                        fd = os.open(tmp, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    except Exception:
                        pass
                    try:
                        os.replace(tmp, path)
                    except Exception:
                        import shutil
                        shutil.move(tmp, path)
                except Exception:
                    # fallback: direct write
                    try:
                        with open(path, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
                            f.write(job.get('text') or '')
                    except Exception:
                        raise
            else:
                ok = False
                err_msg = f"unknown op {op}"
        except Exception as e:  # pragma: no cover (失敗パス)
            ok = False
            err_msg = f"{type(e).__name__}: {e}"
            try:
                tb = ''.join(traceback.format_exc()[-1000:])
                print(f"[async-io][ERROR] op={op} path={path} {err_msg}\n{tb}")
            except Exception:
                print(f"[async-io][ERROR] op={op} path={path} {err_msg}")
        finally:
            if log_events:
                try:
                    dur = (time.time() - started) * 1000
                    print(f"[async-io] pid={pid} op={op} ok={ok} ms={dur:.1f} path={path} err={err_msg}")
                except Exception:
                    pass

# --------- Global helper (lazy singleton) ---------
_global_async_io: AsyncIOProcess | None = None

def get_async_io(cfg: dict | None = None) -> AsyncIOProcess | None:
    global _global_async_io
    if cfg and not cfg.get("enable_async_io", False):
        return None
    if _global_async_io is None:
        qsize = int(cfg.get("async_io_queue_maxsize", 32)) if cfg else 32
        log_events = bool(cfg.get("async_io_log_events", False)) if cfg else False
        warn_full = bool(cfg.get("async_io_warn_queue_full", True)) if cfg else True
        _global_async_io = AsyncIOProcess(maxsize=qsize, log_events=log_events, warn_queue_full=warn_full)
        _global_async_io.start()
    return _global_async_io

def async_flush(cfg: dict | None = None):
    global _global_async_io
    if _global_async_io is None:
        return
    timeout = None
    if cfg:
        t = cfg.get("async_io_flush_timeout_sec")
        if t is not None:
            try:
                timeout = float(t)
            except Exception:
                timeout = None
    _global_async_io.flush(timeout=timeout)
    _global_async_io = None

__all__ = ["get_async_io", "async_flush", "AsyncIOProcess"]
