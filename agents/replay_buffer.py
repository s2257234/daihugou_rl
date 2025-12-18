from __future__ import annotations
from collections import deque
import threading
from typing import Dict, Any, List, Optional
import joblib
import os
import time
import shutil
import numpy as _np
import tempfile


VALUE_U8_NONE = 255


def _encode_value_u8(value: Optional[float]) -> int:
    if value is None:
        return VALUE_U8_NONE
    try:
        # Map [0, 1] to [0, 254] (255 is reserved for VALUE_U8_NONE)
        return int(max(0, min(254, round(float(value) * 254))))
    except Exception:
        return VALUE_U8_NONE


def _decode_value_u8(value_u8: Any) -> Optional[float]:
    try:
        val = int(value_u8)
    except Exception:
        return None
    if 0 <= val < VALUE_U8_NONE:
        return val / 254.0
    return None


def make_replay_sample(
    state: Dict[str, Any],
    legal_actions: List[Any],
    pi: Any,
    value: Optional[float],
    *,
    uid: Optional[int] = None,
    hand_size: Optional[int] = None,
    is_terminal: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a single replay sample dict.

    Responsibility: Replay schema construction (moved from drl_agent.py).

    Notes:
    - Keeps backwards-compatible fields used across the codebase.
    - Stores value in quantized form `value_u8` (255 means None).
    """
    sample: Dict[str, Any] = {
        'state': state if isinstance(state, dict) else {},
        'legal_actions': legal_actions if isinstance(legal_actions, list) else [],
        'pi': pi,
        'value_u8': _encode_value_u8(value),
        'is_terminal': bool(is_terminal),
    }
    if uid is not None:
        try:
            sample['uid'] = int(uid)
        except Exception:
            pass
    if hand_size is not None:
        try:
            sample['hand_size'] = int(hand_size)
        except Exception:
            pass
    if extra and isinstance(extra, dict):
        for k, v in extra.items():
            if k in sample:
                continue
            sample[k] = v
    return sample


def _agent_log(agent: Any, msg: str):
    logger = getattr(agent, 'logger', None)
    if logger is not None and hasattr(logger, 'log_text'):
        try:
            logger.log_text(str(msg))
            return
        except Exception:
            pass
    try:
        print(str(msg))
    except Exception:
        pass


def _to_numpy_if_torch_tensor(obj: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj
    try:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        return obj
    return obj


def canonicalize_state(agent: Any, st: Optional[dict]) -> dict:
    """Canonicalize a state dict so that the stored viewpoint becomes self=0.

    Ported from AlphaZeroAgent._canonicalize_state to keep drl_agent.py thin.
    Best-effort: if layout is unexpected or any step fails, returns the original.
    """
    cfg = getattr(agent, 'config', {}) or {}
    # Allow disabling canonicalization/rotation via config for debugging or dataset compatibility.
    if not bool(cfg.get('enable_canonicalization', True)):
        return st or {}
    if not isinstance(st, dict):
        return st or {}
    fi = st.get('full_input')
    if fi is None:
        return st

    # prefer torch operations for GPU-capable rotation; fallback to numpy
    try:
        import torch as _t
        has_torch = True
    except Exception:
        _t = None  # type: ignore
        has_torch = False

    try:
        N = int(cfg.get('num_players', getattr(getattr(agent, 'model', None), 'num_players', 4)))
    except Exception:
        N = 4
    if N < 2:
        return st

    self_dim = 55

    t = None
    arr = None
    if has_torch and _t is not None:
        try:
            dev = None
            mdl = getattr(agent, 'model', None)
            if mdl is not None and hasattr(mdl, 'device'):
                try:
                    dev = getattr(mdl, 'device')
                except Exception:
                    dev = None
            if dev is None:
                dev = 'cpu'
            t = _t.as_tensor(fi, dtype=_t.float32, device=dev)
        except Exception:
            t = None
    if t is None:
        try:
            arr = _np.asarray(fi, dtype=_np.float32)
        except Exception:
            return st

    def _abort():
        return st

    opp_summary_dim = 5 * (N - 1)
    field_dim = 22
    fieldcards_dim = 53
    turn_dim = N
    opp_discards_dim = 53 * (N - 1)
    pass_matrix_dim = 13 * (N - 1)

    try:
        s_pid = int(st.get('self_player_id', st.get('player_id', getattr(agent, 'player_id', 0))))
    except Exception:
        s_pid = int(getattr(agent, 'player_id', 0) or 0)
    if s_pid < 0 or s_pid >= N:
        s_pid = 0

    if t is not None and _t is not None:
        if int(t.numel()) < (self_dim + 1):
            return _abort()
        context = t[self_dim:]
        total_needed = opp_summary_dim + field_dim + fieldcards_dim + turn_dim + opp_discards_dim + pass_matrix_dim
        if int(context.numel()) < total_needed:
            return _abort()
        idx = 0
        opp_summary = context[idx: idx + opp_summary_dim]; idx += opp_summary_dim
        field = context[idx: idx + field_dim]; idx += field_dim
        fieldcards = context[idx: idx + fieldcards_dim]; idx += fieldcards_dim
        turn = context[idx: idx + turn_dim]; idx += turn_dim
        opp_discards = context[idx: idx + opp_discards_dim]; idx += opp_discards_dim
        pass_matrix = context[idx: idx + pass_matrix_dim]; idx += pass_matrix_dim
        tail = context[idx:]

        per_sum = _t.zeros((N, 5), dtype=_t.float32, device=t.device)
        per_disc = _t.zeros((N, 53), dtype=_t.float32, device=t.device)
        per_pass = _t.zeros((N, 13), dtype=_t.float32, device=t.device)

        if N > 1 and opp_summary_dim > 0:
            try:
                sum_chunks = opp_summary.view(N - 1, 5)
            except Exception:
                sum_chunks = _t.stack([opp_summary[i * 5:(i + 1) * 5] for i in range(N - 1)])
            it = 0
            for pid in range(N):
                if pid == s_pid:
                    continue
                per_sum[pid] = sum_chunks[it]
                it += 1

        if N > 1 and opp_discards_dim > 0:
            try:
                disc_chunks = opp_discards.view(N - 1, 53)
            except Exception:
                disc_chunks = _t.stack([opp_discards[i * 53:(i + 1) * 53] for i in range(N - 1)])
            it = 0
            for pid in range(N):
                if pid == s_pid:
                    continue
                per_disc[pid] = disc_chunks[it]
                it += 1

        if N > 1 and pass_matrix_dim > 0:
            try:
                pass_chunks = pass_matrix.view(N - 1, 13)
            except Exception:
                pass_chunks = _t.stack([pass_matrix[i * 13:(i + 1) * 13] for i in range(N - 1)])
            it = 0
            for pid in range(N):
                if pid == s_pid:
                    continue
                per_pass[pid] = pass_chunks[it]
                it += 1

        rot_per_sum = _t.roll(per_sum, shifts=-s_pid, dims=0)
        rot_per_disc = _t.roll(per_disc, shifts=-s_pid, dims=0)
        rot_per_pass = _t.roll(per_pass, shifts=-s_pid, dims=0)

        comp_sum = rot_per_sum[1:].reshape(-1) if N > 1 else _t.empty((0,), device=t.device)
        comp_disc = rot_per_disc[1:].reshape(-1) if N > 1 else _t.empty((0,), device=t.device)
        comp_pass = rot_per_pass[1:].reshape(-1) if N > 1 else _t.empty((0,), device=t.device)

        new_context = _t.cat(
            [comp_sum, field.to(t.device), fieldcards.to(t.device), turn.to(t.device), comp_disc, comp_pass, tail.to(t.device)],
            dim=0,
        )
        new_self = t[:self_dim]
        new_full = _t.cat([new_self, new_context], dim=0)

        new_state = dict(st)
        new_state['full_input'] = new_full
        new_state['self_player_id'] = 0

        # Rotate hand_labels if present
        try:
            hl_raw = st.get('hand_labels')
            if hl_raw is not None and N > 1:
                hl = _t.as_tensor(list(hl_raw), dtype=_t.float32, device=t.device).flatten()
                expected_hlen = 53 * (N - 1)
                if int(hl.numel()) == expected_hlen:
                    try:
                        hand_chunks = hl.view(N - 1, 53)
                    except Exception:
                        hand_chunks = _t.stack([hl[i * 53:(i + 1) * 53] for i in range(N - 1)])
                    per_hand = _t.zeros((N, 53), dtype=_t.float32, device=t.device)
                    it_h = 0
                    for pid in range(N):
                        if pid == s_pid:
                            continue
                        per_hand[pid] = hand_chunks[it_h]
                        it_h += 1
                    rot_per_hand = _t.roll(per_hand, shifts=-s_pid, dims=0)
                    new_hand_compact = rot_per_hand[1:].reshape(-1)
                    new_state['hand_labels'] = new_hand_compact
        except Exception:
            pass

        # Remap last_action_player
        try:
            lap = st.get('last_action_player')
            if lap is not None:
                lap_i = int(lap)
                new_lap = (lap_i - s_pid) % N
                new_state['last_action_player'] = int(new_lap)
        except Exception:
            pass

        return new_state

    # numpy fallback
    try:
        if arr is None:
            arr = _np.asarray(fi, dtype=_np.float32)
        if int(arr.size) < (self_dim + 1):
            return _abort()
        context = arr[self_dim:]
        total_needed = opp_summary_dim + field_dim + fieldcards_dim + turn_dim + opp_discards_dim + pass_matrix_dim
        if int(context.size) < total_needed:
            return _abort()
        idx = 0
        opp_summary = context[idx: idx + opp_summary_dim]; idx += opp_summary_dim
        field = context[idx: idx + field_dim]; idx += field_dim
        fieldcards = context[idx: idx + fieldcards_dim]; idx += fieldcards_dim
        turn = context[idx: idx + turn_dim]; idx += turn_dim
        opp_discards = context[idx: idx + opp_discards_dim]; idx += opp_discards_dim
        pass_matrix = context[idx: idx + pass_matrix_dim]; idx += pass_matrix_dim
        tail = context[idx:]

        opp_summary_chunks = _np.split(opp_summary, N - 1) if opp_summary_dim > 0 else []
        opp_discards_chunks = _np.split(opp_discards, N - 1) if opp_discards_dim > 0 else []
        pass_matrix_chunks = _np.split(pass_matrix, N - 1) if pass_matrix_dim > 0 else []

        per_sum = [None] * N
        per_disc = [None] * N
        per_pass = [None] * N
        it = 0
        for pid in range(N):
            if pid == s_pid:
                per_sum[pid] = _np.zeros((5,), dtype=_np.float32)
                per_disc[pid] = _np.zeros((53,), dtype=_np.float32)
                per_pass[pid] = _np.zeros((13,), dtype=_np.float32)
            else:
                per_sum[pid] = opp_summary_chunks[it] if opp_summary_chunks else _np.zeros((5,), dtype=_np.float32)
                per_disc[pid] = opp_discards_chunks[it] if opp_discards_chunks else _np.zeros((53,), dtype=_np.float32)
                per_pass[pid] = pass_matrix_chunks[it] if pass_matrix_chunks else _np.zeros((13,), dtype=_np.float32)
                it += 1

        rot_per_sum = [per_sum[(s_pid + new_pid) % N] for new_pid in range(N)]
        rot_per_disc = [per_disc[(s_pid + new_pid) % N] for new_pid in range(N)]
        rot_per_pass = [per_pass[(s_pid + new_pid) % N] for new_pid in range(N)]

        comp_sum = _np.concatenate([rot_per_sum[i] for i in range(1, N)]) if N > 1 else _np.zeros((0,), dtype=_np.float32)
        comp_disc = _np.concatenate([rot_per_disc[i] for i in range(1, N)]) if N > 1 else _np.zeros((0,), dtype=_np.float32)
        comp_pass = _np.concatenate([rot_per_pass[i] for i in range(1, N)]) if N > 1 else _np.zeros((0,), dtype=_np.float32)

        new_context = _np.concatenate(
            [
                comp_sum,
                _np.asarray(field, dtype=_np.float32),
                _np.asarray(fieldcards, dtype=_np.float32),
                _np.asarray(turn, dtype=_np.float32),
                comp_disc,
                comp_pass,
                _np.asarray(tail, dtype=_np.float32),
            ]
        )
        new_self = arr[:self_dim]
        new_full = _np.concatenate([new_self, new_context])
        new_state = dict(st)
        new_state['full_input'] = new_full.astype(_np.float32)
        new_state['self_player_id'] = 0

        try:
            hl_raw = st.get('hand_labels')
            if hl_raw is not None:
                hl_arr = _np.asarray(list(hl_raw), dtype=_np.float32).flatten()
                expected_hlen = 53 * (N - 1)
                if int(hl_arr.size) == expected_hlen and N > 1:
                    hand_chunks = _np.split(hl_arr, N - 1)
                    per_hand = [None] * N
                    it_h = 0
                    for pid in range(N):
                        if pid == s_pid:
                            per_hand[pid] = _np.zeros((53,), dtype=_np.float32)
                        else:
                            per_hand[pid] = hand_chunks[it_h]
                            it_h += 1
                    rot_per_hand = [per_hand[(s_pid + new_pid) % N] for new_pid in range(N)]
                    new_hand_compact = _np.concatenate([rot_per_hand[i] for i in range(1, N)]) if N > 1 else _np.zeros((0,), dtype=_np.float32)
                    new_state['hand_labels'] = new_hand_compact
        except Exception:
            pass

        try:
            lap = st.get('last_action_player')
            if lap is not None:
                lap_i = int(lap)
                new_lap = (lap_i - s_pid) % N
                new_state['last_action_player'] = int(new_lap)
        except Exception:
            pass

        return new_state
    except Exception:
        return st


def store_replay_sample(agent: Any, state: Any, legal_actions: Any, pi: Any, value: Optional[float]):
    """Store a replay sample for an agent.

    This is the implementation that used to live in AlphaZeroAgent._store_sample.
    AlphaZeroAgent should call this function and do no heavy work itself.

    Notes:
    - Best-effort normalization/canonicalization is applied depending on config.
    - Does not silently swallow unexpected failures: logs on notable exceptions.
    """
    cfg = getattr(agent, 'config', {}) or {}

    if not isinstance(state, dict):
        state = {}
    if not isinstance(legal_actions, list):
        try:
            legal_actions = list(legal_actions)
        except Exception:
            legal_actions = []

    # Normalize hand_labels / full_input if they are torch tensors.
    try:
        if 'hand_labels' in state:
            state['hand_labels'] = _to_numpy_if_torch_tensor(state.get('hand_labels'))
            hl_val = state.get('hand_labels')
            if hl_val is not None and 'hand_labels_dim' not in state:
                try:
                    state['hand_labels_dim'] = int(len(hl_val))
                except Exception:
                    pass
        if 'full_input' in state:
            state['full_input'] = _to_numpy_if_torch_tensor(state.get('full_input'))
    except Exception as e:
        _agent_log(agent, f"[WARN][replay] normalize tensors failed: {e}")

    # If hand_labels missing at store time, try to re-extract from current env_ref.
    if 'hand_labels' not in state:
        env_ref = getattr(agent, 'env_ref', None)
        if env_ref is not None and hasattr(agent, '_extract_state'):
            try:
                probe = agent._extract_state(env_ref)
                if isinstance(probe, dict) and ('hand_labels' in probe):
                    state['hand_labels'] = _to_numpy_if_torch_tensor(probe.get('hand_labels'))
                    try:
                        state['hand_labels_dim'] = int(probe.get('hand_labels_dim', len(state.get('hand_labels') or [])))
                    except Exception:
                        pass
            except Exception as e:
                _agent_log(agent, f"[WARN][replay] probe hand_labels via _extract_state failed: {e}")

    # Optional: canonicalize on save to avoid mixed-format datasets.
    if bool(cfg.get('force_canonicalize_on_save', True)):
        try:
            st_can = canonicalize_state(agent, state)
            if isinstance(st_can, dict):
                # normalize tensors after canonicalization
                if 'full_input' in st_can:
                    st_can['full_input'] = _to_numpy_if_torch_tensor(st_can.get('full_input'))
                if 'hand_labels' in st_can:
                    hl = st_can.get('hand_labels')
                    hl = _to_numpy_if_torch_tensor(hl)
                    # Prefer float32 numpy array for persistence
                    if hl is not None and not isinstance(hl, _np.ndarray):
                        try:
                            hl = _np.asarray(list(hl), dtype=_np.float32)
                        except Exception:
                            pass
                    st_can['hand_labels'] = hl
                st_can['self_player_id'] = 0
                state = st_can
        except Exception as e:
            _agent_log(agent, f"[WARN][replay] canonicalize_on_save failed: {e}")

    player_id = getattr(agent, 'player_id', 0)
    model_version = getattr(agent, 'model_version', 0)

    # Lossless mode: keep raw pi and legal_actions.
    if bool(cfg.get('strict_lossless', False)):
        feature_version = 0
        if isinstance(state, dict) and ('full_input' in state or 'full_compact' in state):
            feature_version = state.get('full_input_version', 1)
        sample = {
            'player_id': player_id,
            'state': state,
            'legal_actions': legal_actions,
            'pi': list(pi) if pi is not None else None,
            'value_u8': _encode_value_u8(value),
            'model_version': model_version,
            'feature_version': feature_version,
            'lossless': True,
        }

        if bool(cfg.get('use_full_features')) and sample.get('feature_version') == 0:
            return sample
        return _append_sample_to_agent_buffer(agent, sample, cfg)

    # --- Memory reduction: compress full_input into full_compact when configured ---
    try:
        if (
            bool(cfg.get('use_full_features'))
            and bool(cfg.get('enable_compact_full_input', True))
            and isinstance(state, dict)
            and 'full_input' in state
            and 'full_compact' not in state
        ):
            fi = _to_numpy_if_torch_tensor(state.get('full_input'))
            fi_arr = _np.asarray(fi, dtype=_np.float32)
            total_len = int(fi_arr.shape[0]) if fi_arr.ndim >= 1 else 0

            layout_version = None
            N = None
            if total_len >= 125:
                cand = (total_len - 125) / 59
                if abs(cand - int(cand)) < 1e-6 and 2 <= int(cand) <= 10 and 59 * int(cand) + 125 == total_len:
                    layout_version = 4
                    N = int(cand)

            if layout_version is not None and N is not None:
                binary_indices: List[int] = []
                float_indices: List[int] = []
                cursor = 0
                binary_indices.extend(range(cursor, cursor + 53))
                binary_indices.append(cursor + 53)
                float_indices.append(cursor + 54)
                cursor += 55

                opp_cnt = N - 1
                per_opp = 1 + 4
                for _ in range(opp_cnt):
                    float_indices.append(cursor)
                    binary_indices.extend(range(cursor + 1, cursor + 1 + 4))
                    cursor += per_opp

                if cursor + 22 <= total_len:
                    binary_indices.append(cursor)
                    cursor += 1
                    binary_indices.extend(range(cursor, cursor + 7))
                    cursor += 7
                    binary_indices.extend(range(cursor, cursor + 13))
                    cursor += 13
                    float_indices.append(cursor)
                    cursor += 1

                    if cursor + 53 <= total_len:
                        binary_indices.extend(range(cursor, cursor + 53))
                        cursor += 53
                    if cursor + 53 <= total_len:
                        binary_indices.extend(range(cursor, cursor + 53))
                        cursor += 53

                    belief_len = 53 * (N - 1)
                    if cursor + belief_len <= total_len:
                        float_indices.extend(range(cursor, cursor + belief_len))
                        cursor += belief_len
                    if cursor + N <= total_len:
                        binary_indices.extend(range(cursor, cursor + N))
                        cursor += N

                if cursor == total_len and binary_indices and float_indices:
                    bin_vals = fi_arr[binary_indices]
                    bin_bits = (bin_vals > 0.5).astype(_np.uint8)
                    packed = _np.packbits(bin_bits).tobytes()
                    float_vals = fi_arr[float_indices].astype(_np.float16)
                    state['full_compact'] = {
                        'packed_bits': packed,
                        'floats': float_vals,
                        'binary_len': int(bin_bits.shape[0]),
                        'num_players': int(N),
                        'format': 'cfv1',
                        'full_input_dim': int(total_len),
                        'layout_version': int(layout_version),
                    }
                    if not bool(cfg.get('store_full_input', True)):
                        state.pop('full_input', None)
                else:
                    # Generic fallback compression for other layouts.
                    bin_mask = (fi_arr <= 1e-6) | (fi_arr >= 1.0 - 1e-6)
                    binary_idx = _np.where(bin_mask)[0].tolist()
                    float_idx = _np.where(~bin_mask)[0].tolist()
                    if binary_idx or float_idx:
                        bin_vals = fi_arr[binary_idx] if binary_idx else fi_arr[0:0]
                        bin_bits = (bin_vals > 0.5).astype(_np.uint8) if binary_idx else _np.zeros(0, dtype=_np.uint8)
                        packed = _np.packbits(bin_bits).tobytes() if bin_bits.size > 0 else b''
                        float_vals = fi_arr[float_idx].astype(_np.float16) if float_idx else fi_arr[0:0].astype(_np.float16)
                        state['full_compact'] = {
                            'packed_bits': packed,
                            'floats': float_vals,
                            'binary_len': int(bin_bits.shape[0]),
                            'num_players': int(cfg.get('num_players', 4)),
                            'format': 'cfv1',
                            'full_input_dim': int(total_len),
                            'layout_version': -1,
                        }
                        if not bool(cfg.get('store_full_input', True)):
                            state.pop('full_input', None)
    except Exception as e:
        _agent_log(agent, f"[WARN][replay] full_input compression failed: {e}")

    # --- Quantize pi to uint16 normalized to 65535 ---
    try:
        pi_arr = _np.asarray(pi, dtype=_np.float32)
        if pi_arr.ndim != 1:
            pi_arr = pi_arr.reshape(-1)
    except Exception as e:
        _agent_log(agent, f"[WARN][replay] pi conversion failed: {e}")
        pi_arr = _np.zeros((0,), dtype=_np.float32)

    s = float(pi_arr.sum()) if pi_arr.size > 0 else 0.0
    if s <= 0.0 and pi_arr.size > 0:
        pi_arr[:] = 1.0 / float(pi_arr.size)
        s = 1.0
    if pi_arr.size == 0:
        pi_q = _np.zeros((0,), dtype=_np.uint16)
    else:
        scale = 65535.0 / s
        pi_q = _np.clip(_np.round(pi_arr * scale), 0, 65535).astype(_np.uint16)
        diff = int(65535 - int(pi_q.sum()))
        if diff != 0:
            i = int(_np.argmax(pi_q))
            new_val = int(pi_q[i]) + diff
            if 0 <= new_val <= 65535:
                pi_q[i] = new_val  # type: ignore[index]

    value_u8 = _encode_value_u8(value)
    feature_version = 0
    if isinstance(state, dict) and ('full_input' in state or 'full_compact' in state):
        feature_version = state.get('full_input_version', 1)

    sample: Dict[str, Any] = {
        'player_id': player_id,
        'state': state,
        'pi_q': pi_q,
        'pi_format': 'u16_norm65535',
        'value_u8': value_u8,
        'model_version': model_version,
        'feature_version': feature_version,
    }

    # --- Terminal metadata ---
    hand_size = None
    if isinstance(state, dict):
        hs = state.get('hand_size')
        if hs is not None:
            try:
                hand_size = int(hs)
            except Exception:
                hand_size = None
        if hand_size is None:
            sh = state.get('self_hand_indices') or state.get('self_hand')
            if isinstance(sh, (list, tuple)):
                hand_size = len(sh)
    if hand_size is None:
        env_ref = getattr(agent, 'env_ref', None)
        g = getattr(env_ref, 'game', None) if env_ref is not None else None
        pid = getattr(g, 'turn', None) if g is not None else None
        if pid is None and g is not None:
            pid = getattr(g, 'current_player', None)
        players = getattr(g, 'players', None) if g is not None else None
        if pid is not None and isinstance(players, (list, tuple)) and 0 <= int(pid) < len(players):
            hand_attr = getattr(players[int(pid)], 'hand', None)
            if isinstance(hand_attr, (list, tuple)):
                hand_size = len(hand_attr)
    if hand_size is None:
        hand_size = -1

    sample['hand_size'] = int(hand_size)
    sample['is_terminal'] = (int(hand_size) == 0)

    # Ensure saved state contains the self-player perspective marker.
    if isinstance(sample.get('state'), dict):
        try:
            sample['state']['self_player_id'] = int(sample.get('player_id', player_id))
        except Exception:
            sample['state']['self_player_id'] = sample.get('player_id', player_id)

    # Ensure hand_labels persisted: if missing, try reconstruct from env_ref.game.
    st = sample.get('state')
    if isinstance(st, dict) and ('hand_labels' not in st):
        env_ref = getattr(agent, 'env_ref', None)
        g = getattr(env_ref, 'game', None) if env_ref is not None else None
        players = getattr(g, 'players', None) if g is not None else None
        if isinstance(players, (list, tuple)) and len(players) > 0:
            pid = getattr(g, 'turn', 0)
            try:
                pid = int(pid)
            except Exception:
                pid = 0
            opponents = [i for i in range(len(players)) if i != pid]

            suit_order = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}

            def _card_index_local(card: Any) -> int:
                try:
                    if getattr(card, 'is_joker', False):
                        return 52
                    s = getattr(card, 'suit', 'S')
                    r = int(getattr(card, 'rank', 1))
                    return int(suit_order.get(s, 0)) * 13 + (r - 1)
                except Exception:
                    return 52

            hand_labels: List[float] = []
            for i in opponents:
                vec = [0.0] * 53
                for c in getattr(players[i], 'hand', []) or []:
                    idx = _card_index_local(c)
                    if 0 <= idx < 53:
                        vec[idx] = 1.0
                hand_labels.extend(vec)
            if hand_labels:
                st['hand_labels'] = hand_labels
                st['hand_labels_dim'] = len(hand_labels)

    # Attempt to fill full_input / self_hand_indices if missing using env_ref._extract_state().
    if isinstance(st, dict):
        def _is_missing_like(v: Any) -> bool:
            if v is None:
                return True
            # numpy arrays: avoid truth-value ambiguity
            if isinstance(v, _np.ndarray):
                return v.size == 0
            # common containers
            if isinstance(v, (list, tuple, dict, bytes, bytearray)):
                return len(v) == 0
            return False

        need_full = (
            ('full_input' not in st or _is_missing_like(st.get('full_input')))
            and ('full_compact' not in st or _is_missing_like(st.get('full_compact')))
        )
        need_self_idx = _is_missing_like(st.get('self_hand_indices'))
        if (need_full or need_self_idx) and hasattr(agent, '_extract_state'):
            env_ref = getattr(agent, 'env_ref', None)
            if env_ref is not None:
                try:
                    probe = agent._extract_state(env_ref)
                    if isinstance(probe, dict):
                        if need_full and probe.get('full_input'):
                            st['full_input'] = _to_numpy_if_torch_tensor(probe.get('full_input'))
                            st['full_input_dim'] = probe.get('full_input_dim', len(st.get('full_input') or []))
                            st['full_input_version'] = probe.get('full_input_version', st.get('full_input_version'))
                        if need_self_idx and probe.get('self_hand_indices'):
                            st['self_hand_indices'] = probe.get('self_hand_indices')
                except Exception as e:
                    _agent_log(agent, f"[WARN][replay] probe full_input/self_hand_indices failed: {e}")

    # ------------------ Duplicate sample filter ------------------
    if bool(getattr(agent, '_dup_enabled', False)):
        sig_type = cfg.get('duplicate_signature_type', 'top_value_len')
        top_idx = int(_np.argmax(pi_q)) if pi_q.size > 0 else -1
        legal_len = int(pi_q.size)
        try:
            vq = int(sample.get('value_u8', VALUE_U8_NONE))
        except Exception:
            vq = VALUE_U8_NONE
        if sig_type == 'top_value_len':
            sig = (top_idx, vq, legal_len)
        else:
            sig = (top_idx, vq, legal_len)
        max_cnt = int(cfg.get('duplicate_signature_max_count', 50) or 50)
        q = getattr(agent, '_dup_sig_queue', None)
        counts = getattr(agent, '_dup_sig_counts', None)
        if q is not None and counts is not None:
            c = counts.get(sig, 0) + 1
            counts[sig] = c
            q.append(sig)
            if getattr(q, 'maxlen', None) and len(q) == q.maxlen and (len(q) % 997 == 0):
                new_counts: Dict[Any, int] = {}
                for s_ in q:
                    new_counts[s_] = new_counts.get(s_, 0) + 1
                counts.clear()
                counts.update(new_counts)
            if c > max_cnt:
                try:
                    agent._dup_skipped += 1
                except Exception:
                    pass
                sample['in_buffer'] = False
                log_int = int(cfg.get('duplicate_log_interval', 0) or 0)
                if log_int > 0:
                    try:
                        total_seen = agent._dup_skipped + agent._dup_kept
                        if total_seen - agent._dup_last_log >= log_int:
                            _agent_log(agent, f"[dup] skipped={agent._dup_skipped} kept={agent._dup_kept} ratio={(agent._dup_skipped/max(1,total_seen)):.3f}")
                            agent._dup_last_log = total_seen
                    except Exception:
                        pass
                return sample
            try:
                agent._dup_kept += 1
            except Exception:
                pass
            sample['dup_sig'] = sig
            sample['in_buffer'] = True

    # (Option) keep raw pi for analysis.
    if not bool(cfg.get('drop_raw_pi', True)):
        try:
            sample['pi'] = _np.asarray(pi, dtype=_np.float16)
        except Exception:
            pass

    # Optional legal_actions backup.
    if bool(cfg.get('enable_legal_actions_backup', False)):
        sample['legal_actions'] = legal_actions

    if bool(cfg.get('use_full_features')) and sample.get('feature_version') == 0:
        return sample

    # worker_zero_buffer mode.
    if getattr(agent, 'replay_buffer', None) is None:
        return sample

    return _append_sample_to_agent_buffer(agent, sample, cfg)


def _append_sample_to_agent_buffer(agent: Any, sample: Dict[str, Any], cfg: Dict[str, Any]):
    rb = getattr(agent, 'replay_buffer', None)
    if rb is None:
        return sample

    use_shared = bool(getattr(agent, '_use_shared', False))
    if use_shared and hasattr(rb, 'append'):
        try:
            if bool(cfg.get('replay_async_enabled', False)) and hasattr(rb, 'append_async'):
                drop_oldest = bool(cfg.get('replay_async_drop_oldest', True))
                rb.append_async(sample, drop_oldest=drop_oldest)
            else:
                rb.append(sample)
        except Exception as e:
            _agent_log(agent, f"[WARN][replay] shared append failed: {e}")
        return sample

    # Local (deque/list) compatibility path.
    if isinstance(rb, deque):
        rb.append(sample)
        return sample

    try:
        max_buffer_size = int(getattr(agent, 'max_buffer_size', 0) or 0)
    except Exception:
        max_buffer_size = 0

    if isinstance(rb, list):
        if max_buffer_size > 0 and len(rb) >= max_buffer_size:
            rb.pop(0)
        rb.append(sample)
        return sample

    # Unknown buffer type: best-effort append if possible.
    if hasattr(rb, 'append'):
        try:
            rb.append(sample)
        except Exception as e:
            _agent_log(agent, f"[WARN][replay] append failed (unknown buffer): {e}")
    return sample


def _atomic_joblib_dump(obj, path: str, *, compress: int = 3, backup: bool = True):
    """Atomically write `obj` to `path` using joblib.dump.

    - Writes to a temporary file in the same directory and fsyncs the file,
      then atomically replaces the destination via ``os.replace``.
    - If ``backup`` is True and the destination exists, creates ``path + '.bak'``
      via ``shutil.copy2`` (best-effort).
    - Ensures the temporary file is removed on error.
    """
    d = os.path.dirname(path) or '.'
    # create temp file in same dir to avoid cross-filesystem move issues
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=d)
    try:
        os.close(fd)
    except Exception:
        pass
    bak_path = f"{path}.bak"
    try:
        if backup and os.path.exists(path):
            try:
                shutil.copy2(path, bak_path)
            except Exception:
                pass
        # Write payload to tmp file
        joblib.dump(obj, tmp_path, compress=compress)
        # Try to ensure data hits disk (best-effort)
        try:
            with open(tmp_path, 'rb') as _f:
                _f.flush()
                try:
                    os.fsync(_f.fileno())
                except Exception:
                    pass
        except Exception:
            pass
        # atomic replace
        try:
            os.replace(tmp_path, path)
        except Exception:
            shutil.move(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


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
        # pi_q (variable length) cannot be easily packed; keep as list of arrays per slot
        self._pi_q_slots: List[Optional[_np.ndarray]] = []
        # Backup (.bak) save toggle (default True). Trainer/config から無効化可能。
        self._backup_enabled = True

    def _tensor_to_numpy(self, obj):
        """If obj is a torch.Tensor, convert to CPU numpy array; otherwise return obj."""
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                try:
                    return obj.detach().cpu().numpy()
                except Exception:
                    return obj
        except Exception:
            pass
        return obj

    def _normalize_value_fields(self, sample: Dict[str, Any]):
        if not isinstance(sample, dict):
            return
        try:
            vu = sample.get('value_u8')
        except Exception:
            vu = None
        need_encode = False
        if vu is not None:
            try:
                vu_int = int(vu)
                if 0 <= vu_int <= 255:
                    sample['value_u8'] = vu_int
                else:
                    need_encode = True
            except Exception:
                need_encode = True
        else:
            need_encode = True
        if need_encode:
            val = sample.get('value') if isinstance(sample, dict) else None
            sample['value_u8'] = _encode_value_u8(val if isinstance(val, (int, float)) else None)
        for legacy_key in ('value', 'value_pred', 'value_pred_u8'):
            if legacy_key in sample:
                try:
                    del sample[legacy_key]
                except Exception:
                    pass

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
            fi = self._tensor_to_numpy(fi)
            if isinstance(fi, _np.ndarray) and fi.ndim == 1:
                dim = int(fi.shape[0])
                self._prealloc_dim = dim
                # Allocate contiguous float16 arrays
                self._full_input_arr = _np.zeros((self.maxlen, dim), dtype=_np.float16)
                self._full_input_occupancy = _np.zeros((self.maxlen,), dtype=_np.bool_)
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
            fi = self._tensor_to_numpy(fi)
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
            try:
                self._normalize_value_fields(sample)
            except Exception:
                pass
            # Preallocated storage path
            if self._prealloc_active:
                self._store_into_prealloc(sample)
            self._data.append(sample)
            return sample["uid"]

    def extend(self, samples: List[Dict[str, Any]]) -> List[int]:
        """Batch-append multiple samples with a single lock acquisition.

        Returns list of assigned uids (same order as input). This minimizes
        Python-level call overhead compared to calling `append` repeatedly.
        """
        if not isinstance(samples, (list, tuple)):
            return []
        assigned = []
        to_add = []
        with self._lock:
            # lazy prealloc init may depend on first samples; call per-sample but under single lock
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                try:
                    self._maybe_init_prealloc(sample)
                except Exception:
                    pass
                # prepare sample metadata
                if len(self._data) == self.maxlen:
                    evicted = self._data.popleft()
                    if isinstance(evicted, dict):
                        try:
                            evicted["in_buffer"] = False
                        except Exception:
                            pass
                sample["uid"] = self._next_id
                self._next_id += 1
                # priority
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
                sample['in_buffer'] = True
                try:
                    self._normalize_value_fields(sample)
                except Exception:
                    pass
                # prealloc store
                try:
                    if self._prealloc_active:
                        self._store_into_prealloc(sample)
                except Exception:
                    pass
                to_add.append(sample)
                assigned.append(sample['uid'])
            # bulk extend
            try:
                self._data.extend(to_add)
            except Exception:
                # fallback to per-item append
                for s in to_add:
                    try:
                        self._data.append(s)
                    except Exception:
                        pass
        return assigned

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
                            try:
                                s["in_buffer"] = False
                            except Exception:
                                pass
                    self._data = deque(maxlen=new_maxlen)
                    removed = cur_len
                    # Also drop preallocated arrays to free memory
                    try:
                        self._prealloc_active = False
                        self._prealloc_dim = None
                        self._full_input_arr = None
                        self._full_input_occupancy = None
                        self._pi_q_slots = []
                        self._prealloc_write_pos = 0
                    except Exception:
                        pass
                else:  # keep_newest
                    # keep rightmost newest new_maxlen items
                    keep = list(self._data)[-new_maxlen:]
                    removed = cur_len - len(keep)
                    # Mark older samples as out-of-buffer
                    for s in list(self._data)[:-new_maxlen]:
                        if isinstance(s, dict):
                            try:
                                s["in_buffer"] = False
                            except Exception:
                                pass
                    # If prealloc is active, rebuild arrays and remap slots to new indices
                    if self._prealloc_active and (self._prealloc_dim is not None) and (self._prealloc_dim > 0):
                        dim = int(self._prealloc_dim)
                        try:
                            new_fi = _np.zeros((new_maxlen, dim), dtype=_np.float16)
                            new_occ = _np.zeros((new_maxlen,), dtype=_np.bool_)
                            new_piq = [None] * new_maxlen  # type: ignore[list-item]
                            # Copy newest samples into new arrays with contiguous slots [0..new_maxlen-1]
                            for i, s in enumerate(keep):
                                if not isinstance(s, dict):
                                    continue
                                st = s.get('state') or {}
                                src_slot = st.get('full_input_slot')
                                copied = False
                                # Prefer copying from old prealloc slot when available
                                if src_slot is not None and self._full_input_arr is not None:
                                    try:
                                        src_idx = int(src_slot)
                                        vec = self._full_input_arr[src_idx]
                                        if vec is not None:
                                            new_fi[i] = vec[:dim].astype(_np.float16, copy=False)
                                            new_occ[i] = True
                                            copied = True
                                    except Exception:
                                        copied = False
                                # Fallback: if sample still carries full_input directly
                                if not copied:
                                    fi = st.get('full_input')
                                    if isinstance(fi, _np.ndarray) and getattr(fi, 'ndim', 1) == 1:
                                        try:
                                            new_fi[i] = (fi.astype(_np.float16, copy=False) if fi.dtype == _np.float16 else fi.astype(_np.float16))[:dim]
                                            new_occ[i] = True
                                            copied = True
                                        except Exception:
                                            copied = False
                                # Update slot reference only if copied
                                try:
                                    if copied:
                                        st['full_input_slot'] = i
                                        st['full_input_dim'] = dim
                                        st['full_input_dtype'] = 'float16'
                                        st['full_input'] = None
                                    else:
                                        # old slot becomes invalid after shrink; drop it
                                        if 'full_input_slot' in st:
                                            del st['full_input_slot']
                                except Exception:
                                    pass
                                s['state'] = st
                                # pi_q
                                pq = s.get('pi_q')
                                if not isinstance(pq, _np.ndarray):
                                    pq_slot = s.get('pi_q_slot', None)
                                    if pq_slot is None:
                                        pq_slot = src_slot
                                    if (pq_slot is not None) and self._pi_q_slots:
                                        try:
                                            pq = self._pi_q_slots[int(pq_slot)]
                                        except Exception:
                                            pq = None
                                if isinstance(pq, _np.ndarray):
                                    new_piq[i] = pq.astype(pq.dtype, copy=True)
                                    try:
                                        s['pi_q_slot'] = i
                                    except Exception:
                                        pass
                            # Swap in new arrays
                            self._full_input_arr = new_fi
                            self._full_input_occupancy = new_occ
                            self._pi_q_slots = new_piq
                            self._prealloc_active = True
                            self._prealloc_write_pos = len(keep) % new_maxlen
                        except Exception:
                            # If anything goes wrong, disable prealloc to maintain safety
                            self._prealloc_active = False
                            self._prealloc_dim = None
                            self._full_input_arr = None
                            self._full_input_occupancy = None
                            self._pi_q_slots = []
                            self._prealloc_write_pos = 0
                    # Rebuild deque from kept samples
                    self._data = deque(keep, maxlen=new_maxlen)
            else:
                # expansion: just wrap existing list into larger deque
                self._data = deque(list(self._data), maxlen=new_maxlen)
                # If prealloc is active, expand arrays to new size keeping indices stable
                if self._prealloc_active and (self._prealloc_dim is not None) and (self._prealloc_dim > 0):
                    dim = int(self._prealloc_dim)
                    try:
                        old_cap = int(old_max)
                        new_fi = _np.zeros((new_maxlen, dim), dtype=_np.float16)
                        new_occ = _np.zeros((new_maxlen,), dtype=_np.bool_)
                        new_piq = [None] * new_maxlen  # type: ignore[list-item]
                        if self._full_input_arr is not None:
                            new_fi[:min(old_cap, self._full_input_arr.shape[0])] = self._full_input_arr[:min(old_cap, self._full_input_arr.shape[0])]
                        if self._full_input_occupancy is not None:
                            new_occ[:min(old_cap, self._full_input_occupancy.shape[0])] = self._full_input_occupancy[:min(old_cap, self._full_input_occupancy.shape[0])]
                        if isinstance(self._pi_q_slots, list) and self._pi_q_slots:
                            copy_n = min(old_cap, len(self._pi_q_slots))
                            for i in range(copy_n):
                                new_piq[i] = self._pi_q_slots[i]
                        self._full_input_arr = new_fi
                        self._full_input_occupancy = new_occ
                        self._pi_q_slots = new_piq
                        # keep write_pos as-is (still modulo new maxlen on next write)
                    except Exception:
                        # if expansion fails, leave prealloc as-is (safe but capacity-limited)
                        pass
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
                        fi = self._tensor_to_numpy(fi)
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
            # Free preallocated contiguous storage to release RAM
            try:
                self._prealloc_active = False
                self._prealloc_dim = None
                self._full_input_arr = None
                self._full_input_occupancy = None
                self._pi_q_slots = []
                self._prealloc_write_pos = 0
            except Exception:
                pass

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
            "player_id", "state", "model_version", "feature_version",
            "uid", "pi_q", "pi_format", "legal_ids", "actions_format", "value_u8",
            "split"
        }

        # Contiguous pack format: build arrays on the fly for compact saving
        try:
            snapshot = list(self._data)
            n = len(snapshot)
            dim = None
            if n > 0:
                for s in snapshot:
                    st0 = (s.get('state') if isinstance(s, dict) else None) or {}
                    fi0 = self._tensor_to_numpy(st0.get('full_input'))
                    if isinstance(fi0, _np.ndarray) and fi0.ndim == 1:
                        dim = int(fi0.shape[0])
                        break
                    slot0 = st0.get('full_input_slot')
                    if (slot0 is not None) and (self._full_input_arr is not None):
                        try:
                            dim = int(self._full_input_arr.shape[1])
                            break
                        except Exception:
                            continue
            if dim is not None and n > 0:
                full_input_arr = _np.zeros((n, dim), dtype=_np.float16)
                pi_q_slots: List[Optional[_np.ndarray]] = [None] * n
                meta_list: List[Dict[str, Any]] = []
                for i, s in enumerate(snapshot):
                    if not isinstance(s, dict):
                        continue
                    d = {k: s.get(k) for k in allow_keys if k in s}
                    st = s.get('state') or {}
                    fi = self._tensor_to_numpy(st.get('full_input'))
                    if not (isinstance(fi, _np.ndarray) and fi.ndim == 1):
                        slot = st.get('full_input_slot')
                        if (slot is not None) and (self._full_input_arr is not None):
                            try:
                                fi = self._full_input_arr[int(slot)]
                            except Exception:
                                fi = None
                    if isinstance(fi, _np.ndarray) and fi.ndim == 1:
                        full_input_arr[i] = (fi.astype(_np.float16, copy=False)
                                             if fi.dtype == _np.float16 else fi.astype(_np.float16))[:dim]
                        st_light = {
                            'array_idx': i,
                            'full_input_dim': dim,
                            'full_input_dtype': 'float16'
                        }
                        for mk in ('hand_size', 'field_size', 'turn', 'full_input_len'):
                            if mk in st:
                                st_light[mk] = st[mk]
                        d['state'] = st_light
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
                    'pi_q_slots': pi_q_slots,
                    'samples': meta_list,
                }
                try:
                    _atomic_joblib_dump(payload, path, compress=3, backup=self._backup_enabled)
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
                fi = self._tensor_to_numpy(fi)
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
            try:
                _atomic_joblib_dump(payload, path, compress=3, backup=self._backup_enabled)
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
                # propagate RecursionError to outer handler, otherwise fall through to legacy path
                raise
        except RecursionError as e:
            # First, log once and try no-compression atomic save
            if not hasattr(self, '_recursion_first'):
                print(f"[WARN] replay save recursion error (compress=3): {e}")
                import traceback as _tb
                tb = ''.join(_tb.format_exc()[-2000:])
                print(f"[WARN] traceback tail:\n{tb}")
                self._recursion_first = True
            try:
                _atomic_joblib_dump(payload, path, compress=0, backup=self._backup_enabled)
                return
            except RecursionError:
                pass
            # Ultra-minimal: strip state and retry
            try:
                minimal = []
                for d in safe_list:
                    d2 = {k: v for k, v in d.items() if k != 'state'}
                    minimal.append(d2)
                try:
                    _atomic_joblib_dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": minimal}, path, compress=0, backup=self._backup_enabled)
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
                _atomic_joblib_dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": fallback}, path, compress=0, backup=self._backup_enabled)
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
                rb._pi_q_slots = obj.get('pi_q_slots') or [None] * maxlen
                rb._full_input_occupancy = _np.zeros((maxlen,), dtype=_np.bool_)
                rb._prealloc_active = True
            except Exception:
                # if arrays missing, fall back to legacy branch below
                rb._prealloc_active = False
            samples = obj.get('samples') or []
            legacy_values = obj.get('values')
            for idx, s in enumerate(samples):
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
                    try:
                        rb._normalize_value_fields(s)
                    except Exception:
                        pass
                    if 'value_u8' not in s and isinstance(legacy_values, _np.ndarray):
                        try:
                            v_float = float(legacy_values[idx]) if idx < len(legacy_values) else None
                        except Exception:
                            v_float = None
                        s['value_u8'] = _encode_value_u8(v_float)
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
            try:
                rb._normalize_value_fields(s)
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
