"""Validation utilities for AlphaZero-like agent.

エージェントクラスから検証ステップのロジックを分離し、SRP（単一責任の原則）とDRY（重複排除の原則）に従う。
エージェントはこのモジュールを呼び出して検証を実行でき、
実装の詳細を完全に所有する必要はない。
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


VALUE_U8_NONE = 255  # Legacy constant for backwards compatibility


def _decode_value_u8(value_u8: Any) -> Optional[float]:
    """Decode legacy value_u8 to float. For backwards compatibility only."""
    try:
        val = int(value_u8)
    except Exception:
        return None
    if 0 <= val < VALUE_U8_NONE:
        return val / 254.0
    return None


def _extract_value(sample: Any) -> Optional[float]:
    """Extract value label from sample as float.
    
    Value is stored as raw float in 'value' field. Returns None if not set.
    For backwards compatibility, also checks legacy 'value_u8' field.
    """
    if not isinstance(sample, dict):
        return None
    # Primary: raw float value
    v = sample.get('value')
    if v is not None:
        try:
            return float(v)
        except Exception:
            return None
    # Backwards compatibility: decode legacy value_u8
    vu = sample.get('value_u8')
    return _decode_value_u8(vu)


def _has_value_label(sample: Any) -> bool:
    return _extract_value(sample) is not None


def _hand_loss_like_train(agent, logits, target, mask_unknown=None):
    """学習と同じ定義で手札予測Lossを計算する。
    - BCEWithLogits(reduction='none')
    - Focal係数 (1-pt)^gamma を適用（gamma は config.hand_focal_gamma）
    - 未知カードマスクでの平均（mask 未指定なら単純平均）
    """
    import torch as _t
    import torch.nn as _nn
    import torch
    device = getattr(logits, 'device', None) or torch.device('cpu')
    logits = logits.view(-1).to(device)
    target = target.view(-1).to(device=device, dtype=_t.float32)

    # BCE per-element (no reduction)
    bce = _nn.BCEWithLogitsLoss(reduction='none')
    base = bce(logits, target)

    # No focal term: use standard BCE behavior (focal removed)
    focal = 1.0

    # If mask provided, apply unknown-mask averaging and dynamic class-weighting
    if mask_unknown is not None:
        import numpy as _np
        if not isinstance(mask_unknown, _t.Tensor):
            try:
                m_np = _np.asarray(mask_unknown, dtype=_np.float32)
                if not m_np.flags.writeable:
                    m_np = m_np.copy()
                m = _t.tensor(m_np, dtype=_t.float32)
            except Exception:
                m = _t.tensor(mask_unknown, dtype=_t.float32)
        else:
            m = mask_unknown
        m = m.to(device=device, dtype=_t.float32).view(-1)
        # align shapes
        n = min(base.numel(), m.numel(), target.view(-1).numel())
        if base.numel() != n:
            base = base[:n]
        if m.numel() != n:
            m = m[:n]
        tgt = target.view(-1)[:n]

        eps = 1e-6
        active = m.sum()
        # dynamic per-batch positive-class weight (match training logic)
        pos_area = (tgt * m).sum()
        p = (pos_area / (active + eps)).clamp(min=eps, max=1.0)
        pos_w = ((1.0 - p) / (p + eps)).clamp(1.0, 20.0)
        w_class = 1.0 + (pos_w - 1.0) * tgt

        weighted = base * w_class * m
        return weighted.sum() / (active + eps)
    else:
        # No mask: fallback to simple mean
        return base.mean()


def _build_unknown_mask_from_state(st: Dict[str, Any], hand_dim: int) -> List[float]:
    """state メタから未知カードマスク(length=hand_dim)を構築する。
    - 自手札 + 場 + これまでの捨て札(和集合) = 観測済み -> 0
    - 未知 = 1
    hand_dim は 53*(N-1) を想定。
    """
    try:
        import numpy as _np
        K = max(1, hand_dim // 53)
        self_idx = set(st.get('self_hand_indices') or [])
        field_idx = set(st.get('field_card_indices') or [])
        discard_idx = set(st.get('discard_union_indices') or [])
        seen = self_idx | field_idx | discard_idx
        unknown = [ci for ci in range(53) if ci not in seen]
        mask = _np.zeros(hand_dim, dtype=_np.float32)
        off = 0
        for _ in range(K):
            if unknown:
                for ci in unknown:
                    pos = off + ci
                    if 0 <= pos < hand_dim:
                        mask[pos] = 1.0
            off += 53
        return mask.tolist()
    except Exception:
        # 失敗時は全有効
        return [1.0] * int(hand_dim)


def _filter_validation_pool(agent) -> List[Dict[str, Any]]:
    """Collect validation samples for the given agent from its replay buffer."""
    if agent._use_shared and hasattr(agent.replay_buffer, 'iter_all'):
        return [
            s for s in agent.replay_buffer.iter_all(owner_pid=agent.player_id)
            if _has_value_label(s) and s.get('split') == 'val'
        ]
    try:
        return [
            s for s in agent.replay_buffer
            if isinstance(s, dict) and _has_value_label(s) and (s.get('split') == 'val')
        ]
    except Exception:
        return []


def validate_on_agent(agent, batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Run a single validation pass for the agent and return metrics.

    The logic mirrors the previous `AlphaZeroAgent.validate_step` but is
    implemented in a standalone function to keep the agent lightweight.
    """
    try:
        import torch
    except ImportError:
        return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "torch_not_installed"}

    # main validation logic
    
    if agent.model is None:
        return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_model"}

    pool = _filter_validation_pool(agent)
    if not pool:
        # Fallback: 直列モードなどで 'val' split が存在しない場合、学習バッファからランダム抽出し擬似検証を行う
        try:
            # shared replay 対応
            if getattr(agent, '_use_shared', False) and hasattr(agent.replay_buffer, 'iter_all'):
                fallback_all = [
                    s for s in agent.replay_buffer.iter_all(owner_pid=agent.player_id)
                    if isinstance(s, dict) and _has_value_label(s)
                ]
            else:
                fallback_all = [
                    s for s in agent.replay_buffer
                    if isinstance(s, dict) and _has_value_label(s)
                ]
        except Exception:
            fallback_all = []
        # フィーチャ条件などを一部適用（use_full_features オプション）
        if agent.config.get('use_full_features') and fallback_all:
            try:
                fallback_all = [s for s in fallback_all if (s.get('feature_version', 0) >= 1)] or fallback_all
            except Exception:
                pass
        # 上限サンプル数
        max_samples_fb = int(agent.config.get('val_fallback_max_samples', agent.config.get('val_max_samples', 0) or 0))
        if max_samples_fb > 0 and len(fallback_all) > max_samples_fb:
            import random as _r
            fallback_all = _r.sample(fallback_all, max_samples_fb)
        if not fallback_all:
            return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_val_data"}
        pool = fallback_all  # 擬似検証プール
        # 後段処理で通常と同じ経路を通す。reason を残したい場合は out に追加するが互換性のため省略。

    # Cap samples if configured
    max_samples = int(agent.config.get('val_max_samples', 0) or 0)
    if max_samples > 0 and len(pool) > max_samples:
        pool = random.sample(pool, max_samples)

    # オプション: 検証はval分割全体を使用（安定した指標のため）
    use_full_val = bool(getattr(agent, 'config', {}).get('val_use_full_split', False))
    if use_full_val:
        batch = pool
    else:
        bs = int(batch_size if batch_size is not None else (agent.config.get('val_batch_size') or agent.config.get('batch_size', 256)))
        batch = pool if len(pool) <= bs else random.sample(pool, bs)

    # Filter legacy samples when using full features
    if agent.config.get('use_full_features'):
        filtered = [s for s in batch if s.get('feature_version', 0) >= 1]
        if not filtered:
            return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_full_feature_samples"}
        batch = filtered

    # Switch model to eval mode and disable grad during validation
    was_training = False
    try:
        was_training = bool(getattr(agent.model, 'training', False))
        agent.model.eval()
    except Exception:
        pass

    policy_losses = []
    value_losses = []
    entropies = []
    # 手札予測損失: 検証時にも policy/value と同時に計算する。
    # デフォルトで有効化。ただし config で明示的に無効化可能。
    hand_losses = []  # 手札予測損失 (BCEWithLogits)
    # 手札予測の再現率計測用累積 (TP / actual)
    hand_recall_accum_tp = 0
    hand_recall_accum_actual = 0
    try:
        cfg_flag = bool(getattr(agent, 'config', {}).get('val_compute_hand_pred', True))
    except Exception:
        cfg_flag = True
    # Also enable if training config uses hand_pred_loss_coef > 0
    try:
        coeff = float(getattr(agent, 'config', {}).get('hand_pred_loss_coef', 0.0) or 0.0)
    except Exception:
        coeff = 0.0
    compute_hand_pred = cfg_flag or (coeff > 0.0)
    valid = 0

    collected_pi = []
    collected_model = []
    collected_v_pred = []
    collected_v_t = []

    variable = getattr(agent.model, 'supports_variable_actions', False) and hasattr(agent.model, 'evaluate')

    import torch as _t
    with _t.no_grad():
        # Vectorized path when supported
        vectorized_ok = False
        if not variable and hasattr(agent.model, 'forward_batch'):
            try:
                import numpy as _np
                states: List[Dict[str, Any]] = []
                pi_arrays = []
                v_targets_list = []
                lengths: List[int] = []
                hand_label_list: List[Optional[_np.ndarray]] = []  # 手札ラベル (存在するサンプルのみ格納、無い場合 None)
                any_hand_labels = False

                for sample in batch:
                    # Restore pi
                    if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                        pi_q = sample.get('pi_q')
                        try:
                            arr = pi_q.astype(_np.float32, copy=False) if hasattr(pi_q, 'astype') else _np.asarray(list(pi_q), dtype=_np.float32)
                            s_q = float(arr.sum())
                            pi_arr = (arr / s_q) if s_q > 0 else (_np.ones_like(arr, dtype=_np.float32)/max(1, arr.size))
                        except Exception:
                            raw = sample.get('pi')
                            if not raw:
                                continue
                            pi_arr = _np.asarray(list(raw), dtype=_np.float32)
                    else:
                        raw = sample.get('pi')
                        if not raw:
                            continue
                        pi_arr = _np.asarray(list(raw), dtype=_np.float32)

                    # Restore value target
                    v_target = _extract_value(sample)
                    if v_target is None:
                        continue

                    if agent.config.get('use_full_features') and agent.config.get('skip_zero_padded_full_samples', True):
                        st = sample.get('state') or {}
                        if ('full_compact' not in st) and ('full_input' not in st):
                            continue

                    if pi_arr.size == 0:
                        continue
                    states.append(sample['state'])
                    pi_arrays.append(pi_arr)
                    v_targets_list.append(float(v_target))
                    lengths.append(int(pi_arr.shape[0]))
                    # 手札ラベル収集 (vectorized hand head 対応)
                    hl = None
                    try:
                        st = sample.get('state') or {}
                        if isinstance(st, dict) and ('hand_labels' in st):
                            raw_hl = st.get('hand_labels')
                            if raw_hl is not None:
                                if hasattr(raw_hl, 'astype'):
                                    hl = raw_hl.astype(_np.float32, copy=False)
                                else:
                                    hl = _np.asarray(list(raw_hl), dtype=_np.float32)
                    except Exception:
                        hl = None
                    hand_label_list.append(hl)
                    if hl is not None and hl.size > 0:
                        any_hand_labels = True

                if states:
                    # --- バッチ forward (policy/value/hand) ---
                    # 既存の forward_batch では hand_logits を返さないため、内部パスを再構築
                    # states -> tensor エンコード
                    import torch as _t
                    xs = _t.stack([agent.model._encode_state(s) for s in states], dim=0)
                    # モデル内部処理を再現 (models.PolicyValueNet._forward_from_tensor と同等)
                    x2d = xs.to(agent.model.device).float()
                    if x2d.dim() == 1:
                        x2d = x2d.unsqueeze(0)
                    # pad/truncate to full_feature_dim
                    try:
                        fdim = int(getattr(agent.model, 'full_feature_dim', x2d.size(-1)))
                    except Exception:
                        fdim = x2d.size(-1)
                    cur = x2d.size(-1)
                    if cur != fdim:
                        if cur < fdim:
                            pad = _t.zeros(x2d.size(0), fdim - cur, device=agent.model.device, dtype=x2d.dtype)
                            x2d = _t.cat([x2d, pad], dim=1)
                        else:
                            x2d = x2d[:, :fdim]
                    # split self / context
                    self_dim = int(getattr(agent.model, 'self_dim', 32))
                    context_dim = int(getattr(agent.model, 'context_dim', fdim - self_dim))
                    parts = _t.split(x2d, [self_dim, context_dim], dim=1)
                    if len(parts) != 2:
                        # フォールバック: zeros backbone
                        h = _t.zeros(x2d.size(0), int(getattr(agent.model, 'backbone_in_dim', 512)), device=agent.model.device)
                        policy_logits_batch = agent.model.policy_head(h)
                        value_logits_batch = agent.model.value_head(h)
                        hand_logits_batch = None
                    else:
                        self_feat, context_feat = parts
                        self_emb = agent.model.self_encoder(self_feat)
                        context_emb = agent.model.context_encoder(context_feat)
                        combined = _t.cat([self_emb, context_emb], dim=1)
                        h = agent.model.backbone(combined)
                        policy_logits_batch = agent.model.policy_head(h)
                        value_logits_batch = agent.model.value_head(h)
                        if getattr(agent.model, 'enable_hand_prediction_head', False) and getattr(agent.model, 'hand_head', None) is not None:
                            try:
                                try:
                                    hand_logits_batch = agent.model.hand_head(h)
                                except Exception as _e:
                                    # Debug: expose failures in hand_head forward
                                    try:
                                        import traceback as _tb
                                        _tb.print_exc()
                                    except Exception:
                                        pass
                                    hand_logits_batch = None
                            except Exception:
                                hand_logits_batch = None
                        else:
                            hand_logits_batch = None
                    device = getattr(policy_logits_batch, 'device', None)
                    orig_max_len = max(lengths)
                    B = len(states)
                    policy_out_dim = getattr(policy_logits_batch, 'shape', (None, None))[1] or policy_logits_batch.size(1)
                    if orig_max_len > policy_out_dim:
                        try:
                            print(f"[WARN] validation: pi length {orig_max_len} > model.policy_dim {policy_out_dim}; truncating targets")
                        except Exception:
                            pass
                    capped_max_len = min(orig_max_len, int(policy_out_dim))
                    pi_pad = _np.zeros((B, capped_max_len), dtype=_np.float32)
                    for i, arr in enumerate(pi_arrays):
                        take = min(int(arr.shape[0]), capped_max_len)
                        pi_pad[i, :take] = arr[:take]
                    # Use _t.tensor to copy and avoid non-writable numpy array warnings
                    pi_pad_t = _t.tensor(pi_pad, device=device)
                    # truncate lengths for masking to match capped width
                    lengths_trunc = [min(int(l), capped_max_len) for l in lengths]
                    lengths_t = _t.tensor(lengths_trunc, device=device)
                    logits_slice = policy_logits_batch[:, :capped_max_len]
                    arange = _t.arange(capped_max_len, device=device).unsqueeze(0).expand(B, -1)
                    mask = (arange < lengths_t.unsqueeze(1)).float()
                    LARGE_NEG = -1e9
                    masked_logits = logits_slice * mask + (1 - mask) * LARGE_NEG
                    log_probs = _t.log_softmax(masked_logits, dim=1)
                    probs = _t.exp(log_probs) * mask
                    policy_loss_all = - (pi_pad_t * log_probs).sum(dim=1)
                    pid = getattr(agent, 'player_id', 0)
                    if value_logits_batch.ndim == 2 and pid < value_logits_batch.shape[1]:
                        v_logits = value_logits_batch[:, pid]
                    else:
                        v_logits = value_logits_batch[:, 0]
                    v_targets_t = _t.tensor(v_targets_list, dtype=_t.float32, device=device).view(-1)
                    if 'bce_logits_loss_fn' not in agent.__dict__:
                        import torch.nn as _nn
                        try:
                            pw = float(agent.pos_weight)
                        except Exception:
                            pw = 1.0
                        pos_w_tensor = None
                        if pw != 1.0:
                            pos_w_tensor = _t.tensor([pw], dtype=_t.float32, device=device)
                        agent.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()
                    value_loss_all = agent.bce_logits_loss_fn(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1))
                    entropy_all = - (probs * log_probs).sum(dim=1)

                    policy_losses.append(policy_loss_all.mean())
                    value_losses.append(value_loss_all)
                    entropies.append(entropy_all.mean())
                    valid = len(states)

                    for i in range(len(states)):
                        n = min(lengths[i], capped_max_len)
                        collected_pi.append(pi_pad_t[i, :n].detach())
                        collected_model.append(probs[i, :n].detach())
                        v_prob = _t.sigmoid(v_logits[i])
                        collected_v_pred.append(v_prob.detach())
                        collected_v_t.append(v_targets_t[i].detach())
                    # --- 手札予測損失 (vectorized) ---
                    # Disabled by default because it is expensive; enable via compute_hand_pred=True
                    if compute_hand_pred and any_hand_labels and hand_logits_batch is not None:
                        try:
                            hlog = hand_logits_batch
                            if hlog.dim() == 1:
                                hlog = hlog.unsqueeze(0)
                            hand_dim = int(hlog.size(1))
                            hp_losses_sample = []
                            for i, hl in enumerate(hand_label_list):
                                if i < 3:
                                    if hl is None or hl.size == 0:
                                        continue
                                # pad/truncate hl to hand_dim
                                if hl.shape[0] != hand_dim:
                                    if hl.shape[0] < hand_dim:
                                        pad = _np.zeros(hand_dim, dtype=_np.float32)
                                        pad[:hl.shape[0]] = hl
                                        hlv = pad
                                    else:
                                        hlv = hl[:hand_dim]
                                else:
                                    hlv = hl
                                hl_t = _t.tensor(hlv.astype(_np.float32), device=device)
                                # 未知マスクを state メタから生成
                                st_i = states[i] if isinstance(states[i], dict) else {}
                                mask_u = _build_unknown_mask_from_state(st_i, hand_dim)
                                loss_i = _hand_loss_like_train(agent, hlog[i], hl_t, mask_u)
                                hp_losses_sample.append(loss_i)
                                # hand recall accumulate: predicted positives vs actual positives
                                try:
                                    probs_i = _t.sigmoid(hlog[i].view(-1))
                                    pred_pos = (probs_i > 0.5)
                                    actual_pos = (hl_t.view(-1) > 0.5)
                                    tp = int((_t.logical_and(pred_pos, actual_pos).sum()).item())
                                    total_actual = int(actual_pos.sum().item())
                                    hand_recall_accum_tp += int(tp)
                                    hand_recall_accum_actual += int(total_actual)
                                except Exception:
                                    pass
                            if hp_losses_sample:
                                hand_losses.append(_t.stack(hp_losses_sample).mean())
                        except Exception:
                            pass
                    vectorized_ok = True
            except Exception:
                vectorized_ok = False

        # Fallback per-sample path
        if not vectorized_ok:
            for sample in batch:
                legal_actions = sample.get("legal_actions")
                # 可変長モデル時のみ id_v1 の復元を試みる（固定ヘッドでは不要）
                if variable and legal_actions is None and sample.get('actions_format') == 'id_v1' and 'legal_ids' in sample:
                    try:
                        # ACTION_ID_LIST 非依存にするため、固定ヘッドではスキップ
                        legal_actions = None
                    except Exception:
                        legal_actions = None

                # Restore π
                if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                    try:
                        import numpy as _np
                        pi_q = sample['pi_q']
                        pi_arr = pi_q.astype(_np.float32) if hasattr(pi_q, 'astype') else _np.asarray(list(pi_q), dtype=_np.float32)
                        s_q = float(pi_arr.sum())
                        if s_q <= 0:
                            pi_target = [1.0 / len(pi_arr)] * int(len(pi_arr)) if len(pi_arr) > 0 else []
                        else:
                            pi_target = (pi_arr / s_q).tolist()
                    except Exception:
                        pi_target = sample.get('pi')
                else:
                    pi_target = sample.get('pi')

                # Restore value target
                v_target = _extract_value(sample)

                # 固定ヘッドでは legal_actions は不要。可変長のみ必須とする。
                if (variable and not legal_actions) or (not pi_target) or (v_target is None):
                    continue
                if agent.config.get('use_full_features') and agent.config.get('skip_zero_padded_full_samples', True):
                    try:
                        st = sample.get('state') or {}
                        if ('full_compact' not in st) and ('full_input' not in st):
                            continue
                    except Exception:
                        pass

                # 可変長: 合法手数、固定ヘッド: πの長さを使用
                n = (len(legal_actions) if variable else len(pi_target))
                # Model forward
                if variable:
                    logits_raw, v_pred_raw = agent.model.evaluate(sample['state'], legal_actions)
                else:
                    logits_raw, v_out_logits = agent.model.forward(sample['state'])
                    # v_out_logits may be:
                    # - a scalar (0-d tensor / float)
                    # - a 1-d array/tensor with per-player logits
                    # - other shapes. Handle safely to avoid IndexError on shape[0].
                    try:
                        shape = getattr(v_out_logits, 'shape', None)
                        # If shape is a tuple-like and has at least one dim
                        if shape is None:
                            # not array-like, treat as scalar
                            v_pred_raw = v_out_logits
                        else:
                            # some numpy/tensor types expose empty tuple for scalars
                            try:
                                # obtain length of first dimension if possible
                                dim0 = shape[0]
                            except Exception:
                                # scalar (0-d), use as-is
                                v_pred_raw = v_out_logits
                            else:
                                pid = getattr(agent, 'player_id', 0)
                                try:
                                    # prefer pid index when available
                                    if isinstance(dim0, int) and 0 <= pid < int(dim0):
                                        v_pred_raw = v_out_logits[pid]
                                    else:
                                        # fallback to first element when available
                                        if int(dim0) > 0:
                                            v_pred_raw = v_out_logits[0]
                                        else:
                                            v_pred_raw = v_out_logits
                                except Exception:
                                    # final fallback
                                    v_pred_raw = v_out_logits
                    except Exception:
                        v_pred_raw = v_out_logits

                try:
                    # determine model device so newly created tensors live on same device
                    try:
                        _model_dev = getattr(agent.model, 'device', None)
                        if _model_dev is None:
                            try:
                                _model_dev = next(agent.model.parameters()).device
                            except Exception:
                                import torch as _torch
                                _model_dev = _torch.device('cpu')
                    except Exception:
                        import torch as _torch
                        _model_dev = _torch.device('cpu')

                    if hasattr(logits_raw, 'shape'):
                        logits_t = logits_raw
                        if logits_t.shape[0] < n:
                            pad = _t.zeros(n - logits_t.shape[0], device=logits_t.device)
                            logits_t = _t.cat([logits_t, pad], dim=0)
                        else:
                            logits_t = logits_t[:n]
                    else:
                        logits_list = list(logits_raw)
                        if len(logits_list) < n:
                            logits_list += [0.0] * (n - len(logits_list))
                        logits_t = _t.tensor(logits_list[:n], dtype=_t.float32, device=_model_dev)

                    log_probs = logits_t.log_softmax(dim=0)
                    probs = log_probs.exp()

                    pi_t = _t.tensor(pi_target, dtype=_t.float32, device=log_probs.device)
                    if pi_t.shape[0] != log_probs.shape[0]:
                        m = min(pi_t.shape[0], log_probs.shape[0])
                        pi_t = pi_t[:m]
                        log_probs = log_probs[:m]
                        probs = probs[:m]

                    policy_loss = -(pi_t * log_probs).sum()
                    # ensure v_logit is a tensor on the model device
                    if isinstance(v_pred_raw, float) or isinstance(v_pred_raw, (int,)):
                        try:
                            v_logit = _t.tensor(v_pred_raw, dtype=_t.float32, device=_model_dev)
                        except Exception:
                            v_logit = _t.tensor(v_pred_raw, dtype=_t.float32)
                    else:
                        try:
                            v_logit = v_pred_raw.float()
                        except Exception:
                            try:
                                v_logit = _t.tensor(float(v_pred_raw), dtype=_t.float32, device=_model_dev)
                            except Exception:
                                v_logit = _t.tensor(float(v_pred_raw), dtype=_t.float32)
                    v_t = _t.tensor(float(v_target), dtype=_t.float32, device=v_logit.device)

                    if 'bce_logits_loss_fn' not in agent.__dict__:
                        import torch.nn as _nn
                        try:
                            pw = float(agent.pos_weight)
                        except Exception:
                            pw = 1.0
                        pos_w_tensor = None
                        if pw != 1.0:
                            pos_w_tensor = _t.tensor([pw], dtype=_t.float32, device=v_logit.device)
                        agent.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()

                    value_loss = agent.bce_logits_loss_fn(v_logit.unsqueeze(0), v_t.unsqueeze(0))
                    v_prob = _t.sigmoid(v_logit)
                    entropy = -(probs * log_probs).sum()

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    valid += 1
                    collected_pi.append(pi_t.detach())
                    collected_model.append(probs.detach())
                    collected_v_pred.append(v_prob.detach())
                    collected_v_t.append(v_t.detach())
                    # --- 手札予測損失: 学習と同じ定義（未知マスク + Focal） ---
                    try:
                        # Disabled by default to avoid heavy CPU work during validation
                        if compute_hand_pred and getattr(agent.model, 'enable_hand_prediction_head', False) and hasattr(agent.model, 'forward_with_belief'):
                            st = sample.get('state') or {}
                            hl = None
                            if isinstance(st, dict) and ('hand_labels' in st):
                                import numpy as _np
                                raw = st.get('hand_labels')
                                if hasattr(raw, 'astype'):
                                    hl = raw.astype(_np.float32, copy=False)
                                else:
                                    try:
                                        hl = _np.asarray(list(raw), dtype=_np.float32)
                                    except Exception:
                                        hl = None
                            if hl is not None and hl.size > 0:
                                import torch as _t
                                pol_l, val_l, hand_logits = agent.model.forward_with_belief(st)
                                if hand_logits is not None:
                                    # pad/truncate labels to match logits
                                    hand_dim = int(hand_logits.view(-1).numel())
                                    if hl.shape[0] != hand_dim:
                                        if hl.shape[0] < hand_dim:
                                            pad = _np.zeros(hand_dim, dtype=_np.float32)
                                            pad[:hl.shape[0]] = hl
                                            hl = pad
                                        else:
                                            hl = hl[:hand_dim]
                                    hl_t = _t.tensor(hl.astype(_np.float32), device=hand_logits.device)
                                    # 未知マスク
                                    mask_u = _build_unknown_mask_from_state(st, hand_dim)
                                    hp_loss = _hand_loss_like_train(agent, hand_logits.view(-1), hl_t.view(-1), mask_u)
                                    hand_losses.append(hp_loss)
                                    # accumulate recall for this sample
                                    try:
                                        probs_h = _t.sigmoid(hand_logits.view(-1))
                                        pred_pos_h = (probs_h > 0.5)
                                        actual_pos_h = (hl_t.view(-1) > 0.5)
                                        tp_h = int((_t.logical_and(pred_pos_h, actual_pos_h).sum()).item())
                                        total_actual_h = int(actual_pos_h.sum().item())
                                        hand_recall_accum_tp += int(tp_h)
                                        hand_recall_accum_actual += int(total_actual_h)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                except Exception:
                    continue

    # Diagnostic: hand prediction disabled by default

    # restore training mode if needed
    try:
        if was_training:
            agent.model.train()
    except Exception:
        pass

    if valid == 0:
        return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_valid_samples"}

    import torch as _t
    policy_loss_mean = _t.stack(policy_losses).mean()
    value_loss_mean = _t.stack(value_losses).mean()
    entropy_mean = _t.stack(entropies).mean()
    # hand prediction: compute mean if any hand losses were collected
    try:
        if hand_losses:
            hand_loss_mean = _t.stack(hand_losses).mean()
        else:
            hand_loss_mean = None
    except Exception:
        hand_loss_mean = None

    # Common metrics
    try:
        from utils.metrics import compute_policy_value_metrics
        metrics_extra = compute_policy_value_metrics(collected_pi, collected_model, collected_v_pred, collected_v_t)
    except Exception:
        metrics_extra = {"policy_kl": None, "policy_top1_match": None, "value_acc": None, "value_brier": None}

    out = {
        "policy_loss": float(policy_loss_mean.item()),
        "value_loss": float(value_loss_mean.item()),
        "entropy": float(entropy_mean.item()),
        "samples": valid,
        **metrics_extra,
    }
    # 手札予測損失が計算できた場合は追加
    try:
        if hand_loss_mean is not None:
            out["hand_pred_loss"] = float(hand_loss_mean.item())
        else:
            out["hand_pred_loss"] = None
    except Exception:
        out["hand_pred_loss"] = None
    # hand_recall: accumulated TP / actual across hand-labeled samples (if any)
    try:
        if hand_recall_accum_actual > 0:
            out['hand_recall'] = float(hand_recall_accum_tp / hand_recall_accum_actual)
        else:
            out['hand_recall'] = None
    except Exception:
        out['hand_recall'] = None
    return out


def apply_hand_prediction_constraint(hand_probs, num_opponents: int = 3):
    """手札予測確率にカード枚数制約を適用して正規化する。
    
    各カードについて、全相手プレイヤーの予測確率の合計が4を超えないように
    正規化します（各カードはデッキに最大4枚しか存在しないため）。
    
    Args:
        hand_probs: 手札予測確率の配列 (shape: [num_opponents * 53])
                   各相手プレイヤーについて53次元（53種類のカード）
        num_opponents: 相手プレイヤー数（デフォルト: 3）
    
    Returns:
        制約を適用した手札予測確率の配列（同じshape）
    """
    import numpy as np
    
    # numpy配列に変換
    if not isinstance(hand_probs, np.ndarray):
        hand_probs = np.asarray(hand_probs, dtype=np.float32)
    
    # コピーを作成（元の配列を変更しない）
    hand_probs_constrained = hand_probs.copy()
    
    # 各カード（53種類）について処理
    for card_idx in range(53):
        # このカードについて全相手プレイヤーの確率を取得
        card_probs = []
        for opp_idx in range(num_opponents):
            offset = opp_idx * 53 + card_idx
            if offset < len(hand_probs_constrained):
                card_probs.append(hand_probs_constrained[offset])
            else:
                card_probs.append(0.0)
        
        # 確率の合計を計算
        total_prob = sum(card_probs)
        
        # 合計が4を超える場合は正規化
        if total_prob > 4.0:
            scale = 4.0 / total_prob
            for opp_idx in range(num_opponents):
                offset = opp_idx * 53 + card_idx
                if offset < len(hand_probs_constrained):
                    hand_probs_constrained[offset] *= scale
    
    return hand_probs_constrained
