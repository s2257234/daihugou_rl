"""Optional fast MCTS helpers.

If a compiled Cython extension is available (agents._mcts_fast), use it.
Otherwise, provide a pure-Python fallback with identical API.
"""
from __future__ import annotations

_FALLBACK_LOGGED = set()

def _log_fallback_once(key: str, msg: str, exc: Exception | None = None):
    if key in _FALLBACK_LOGGED:
        return
    _FALLBACK_LOGGED.add(key)
    try:
        if exc is not None:
            print(f"{msg} ({type(exc).__name__}: {exc})")
        else:
            print(msg)
    except Exception:
        # 最低限の安全策
        pass

_FAST_IMPORT_ERROR = None
try:
    # Compiled module produced from _mcts_fast.pyx
    from agents._mcts_fast import (
        puct_select_index_fast as _fast,
        puct_backup_generic as _backup_generic,
        puct_backup_scalar as _backup_scalar,
    )
except Exception as e:  # pragma: no cover
    _FAST_IMPORT_ERROR = e
    _fast = None
    _backup_generic = None
    _backup_scalar = None

if _FAST_IMPORT_ERROR is not None:
    _log_fallback_once("cython_import", "[mcts-fallback] Cython MCTS extension unavailable; using pure-Python paths", _FAST_IMPORT_ERROR)

def puct_select_index_fast(priors, values, visits, virtual_counts, c_puct: float, total_visits: int):
    """Return index of best child by PUCT. Fallback to Python if Cython not present.

    Args:
        priors, values: list[float]
        visits, virtual_counts: list[int]
        c_puct: float
        total_visits: int
    Returns:
        int index (>=0) or -1 if none
    """
    if _fast is not None:
        # Convert to contiguous memoryviews by building python lists; Cython wrapper accepts them
        try:
            return int(_fast(priors, values, visits, virtual_counts, float(c_puct), int(total_visits)))
        except Exception:
            # エラー時は自動的にPythonフォールバックへ（agents/mcts.py側でハンドリング済み）
            pass
    # Pure-Python fallback
    n = len(priors)
    if n == 0:
        return -1
    tv = max(1, int(total_visits))
    import math
    sqrt_total = math.sqrt(tv)
    best = -1e300
    best_idx = -1
    for i in range(n):
        ve = int(visits[i]) + int(virtual_counts[i])
        u = c_puct * float(priors[i]) * sqrt_total / (1.0 + ve)
        score = float(values[i]) + u
        if score > best:
            best = score
            best_idx = i
    return best_idx


def puct_backup_scalar(node, leaf_value):
    """Fast backup when leaf_value is a scalar.

    Falls back to Python loop if Cython extension is not available.
    """
    if _backup_scalar is not None:
        try:
            return _backup_scalar(node, float(leaf_value))
        except Exception as e:
            _log_fallback_once("backup_scalar_exception", "[mcts-fallback] puct_backup_scalar failed; using Python fallback", e)
    else:
        _log_fallback_once("backup_scalar_missing", "[mcts-fallback] puct_backup_scalar not available; using Python fallback")
    # Fallback
    try:
        v = float(leaf_value)
    except Exception:
        v = 0.0
    cur = node
    while cur is not None:
        try:
            cur.visit_count += 1
            cur.value_sum += v
        except Exception:
            pass
        try:
            cur = cur.parent
        except Exception:
            cur = None


def puct_backup_generic(node, leaf_value):
    """Backup with per-node to_play handling.

    If Cython extension exists, use it; otherwise do a Python fallback.
    """
    # If scalar, route to scalar path
    if isinstance(leaf_value, (int, float)):
        return puct_backup_scalar(node, float(leaf_value))
    if _backup_generic is not None:
        try:
            return _backup_generic(node, leaf_value)
        except Exception as e:
            _log_fallback_once("backup_generic_exception", "[mcts-fallback] puct_backup_generic failed; using Python fallback", e)
    else:
        _log_fallback_once("backup_generic_missing", "[mcts-fallback] puct_backup_generic not available; using Python fallback")
    # Python fallback
    cur = node
    while cur is not None:
        try:
            pid = int(getattr(cur, 'to_play', 0))
        except Exception:
            pid = 0
        # Resolve value component
        if isinstance(leaf_value, dict):
            tmp = leaf_value.get(pid, 0.0)
        elif isinstance(leaf_value, (list, tuple)):
            try:
                tmp = leaf_value[pid]
            except Exception:
                tmp = 0.0
        else:
            tmp = leaf_value
        try:
            cur.visit_count += 1
            cur.value_sum += float(tmp)
        except Exception:
            pass
        try:
            cur = cur.parent
        except Exception:
            cur = None
