# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from libc.math cimport sqrt
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t puct_select_index_fast(
    double[:] priors,
    double[:] values,
    long[:] visits,
    long[:] virtual_counts,
    double c_puct,
    long total_visits,
):
    cdef Py_ssize_t n = priors.shape[0]
    cdef Py_ssize_t i
    cdef double best = -1e300
    cdef Py_ssize_t best_idx = -1
    cdef long tv = 0
    cdef double sqrt_total
    cdef double u, score
    cdef long ve
    if n == 0:
        return -1
    # Compute total_visits inside C if not provided (>0 expected). This avoids Python-side sum.
    if total_visits < 1:
        tv = 0
        for i in range(n):
            tv += visits[i] + virtual_counts[i]
        if tv < 1:
            tv = 1
        total_visits = tv
    sqrt_total = sqrt(<double>total_visits)
    for i in range(n):
        ve = visits[i] + virtual_counts[i]
        u = c_puct * priors[i] * sqrt_total / (1.0 + ve)
        score = values[i] + u
        if score > best:
            best = score
            best_idx = i
    return best_idx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void puct_backup_scalar(object node, double leaf_value):
    """Fast backup when leaf_value is a single float.

    Walks parent links and updates visit_count and value_sum.
    Resilient to nodes missing attributes (ignores errors).
    """
    cdef object cur = node
    while cur is not None:
        try:
            cur.visit_count += 1
            cur.value_sum += leaf_value
        except Exception:
            pass
        try:
            cur = cur.parent
        except Exception:
            cur = None


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void puct_backup_generic(object node, object leaf_value):
    """Backup with per-node to_play support.

    leaf_value can be:
      - float/int: same value for all nodes
      - dict: {player_id: value}
      - list/tuple: index is player_id
    Each node's to_play is used to select the appropriate component.
    """
    cdef object cur = node
    cdef bint is_dict = isinstance(leaf_value, dict)
    cdef bint is_list = False
    cdef bint is_tuple = False
    cdef Py_ssize_t seq_len = 0
    if not is_dict:
        is_list = isinstance(leaf_value, list)
        if not is_list:
            is_tuple = isinstance(leaf_value, tuple)
        if is_list or is_tuple:
            try:
                seq_len = len(leaf_value)
            except Exception:
                seq_len = 0
    cdef double val
    cdef int pid
    cdef object tmp
    while cur is not None:
        try:
            pid = int(getattr(cur, 'to_play', 0))
        except Exception:
            pid = 0
        if is_dict:
            try:
                tmp = leaf_value.get(pid, 0.0)
            except Exception:
                tmp = 0.0
            try:
                val = float(tmp)
            except Exception:
                val = 0.0
        elif is_list or is_tuple:
            if 0 <= pid < seq_len:
                try:
                    tmp = leaf_value[pid]
                except Exception:
                    tmp = 0.0
            else:
                tmp = 0.0
            try:
                val = float(tmp)
            except Exception:
                val = 0.0
        else:
            try:
                val = float(leaf_value)
            except Exception:
                val = 0.0
        try:
            cur.visit_count += 1
            cur.value_sum += val
        except Exception:
            pass
        try:
            cur = cur.parent
        except Exception:
            cur = None
