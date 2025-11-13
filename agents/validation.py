"""Validation utilities for AlphaZero-like agent.

Moves validation step logic out of the agent class to follow SRP and DRY.
The agent can call into this module to perform validation without owning
the full implementation details.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


def _filter_validation_pool(agent) -> List[Dict[str, Any]]:
    """Collect validation samples for the given agent from its replay buffer."""
    if agent._use_shared and hasattr(agent.replay_buffer, 'iter_all'):
        return [
            s for s in agent.replay_buffer.iter_all(owner_pid=agent.player_id)
            if s.get("value") is not None and s.get('split') == 'val'
        ]
    try:
        return [
            s for s in agent.replay_buffer
            if isinstance(s, dict) and (s.get('value') is not None) and (s.get('split') == 'val')
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

    if agent.model is None:
        return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_model"}

    pool = _filter_validation_pool(agent)
    if not pool:
        return {"policy_loss": None, "value_loss": None, "entropy": None, "reason": "no_val_data"}

    # Cap samples if configured
    max_samples = int(agent.config.get('val_max_samples', 0) or 0)
    if max_samples > 0 and len(pool) > max_samples:
        pool = random.sample(pool, max_samples)

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
    hand_losses = []  # 手札予測損失 (BCEWithLogits)
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
                    v_target = sample.get('value')
                    if v_target is None and 'value_u8' in sample:
                        vu = sample.get('value_u8')
                        if isinstance(vu, int) and vu != 255:
                            v_target = vu / 255.0
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
                                hand_logits_batch = agent.model.hand_head(h)
                            except Exception:
                                hand_logits_batch = None
                        else:
                            hand_logits_batch = None
                    device = getattr(policy_logits_batch, 'device', None)
                    max_len = max(lengths)
                    B = len(states)
                    pi_pad = _np.zeros((B, max_len), dtype=_np.float32)
                    for i, arr in enumerate(pi_arrays):
                        pi_pad[i, :arr.shape[0]] = arr
                    pi_pad_t = _t.from_numpy(pi_pad).to(device)
                    lengths_t = _t.tensor(lengths, device=device)
                    logits_slice = policy_logits_batch[:, :max_len]
                    arange = _t.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
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
                    v_targets_t = _t.tensor(v_targets_list, dtype=_t.float32, device=device)
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
                        n = lengths[i]
                        collected_pi.append(pi_pad_t[i, :n].detach())
                        collected_model.append(probs[i, :n].detach())
                        v_prob = _t.sigmoid(v_logits[i])
                        collected_v_pred.append(v_prob.detach())
                        collected_v_t.append(v_targets_t[i].detach())
                    # --- 手札予測損失 (vectorized) ---
                    if any_hand_labels and hand_logits_batch is not None:
                        try:
                            # BCEWithLogitsLoss (reduction='none') で各サンプル損失を計算
                            import torch.nn as _nn
                            hlog = hand_logits_batch
                            if hlog.dim() == 1:
                                hlog = hlog.unsqueeze(0)
                            # hand_logits_batch.shape = [B, hand_pred_dim]
                            hand_dim = hlog.size(1)
                            bce_hand = _nn.BCEWithLogitsLoss(reduction='none')
                            hp_losses_sample = []
                            for i, hl in enumerate(hand_label_list):
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
                                hl_t = _t.from_numpy(hlv.astype(_np.float32)).to(device)
                                logits_i = hlog[i]
                                # per element losses -> mean
                                per_elem = bce_hand(logits_i.view(-1), hl_t.view(-1))
                                hp_losses_sample.append(per_elem.mean())
                            if hp_losses_sample:
                                # 平均を hand_losses へ集約 (同一インターフェイス維持)
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
                v_target = sample.get('value')
                if v_target is None and 'value_u8' in sample:
                    vu = sample.get('value_u8')
                    try:
                        if isinstance(vu, int) and vu != 255:
                            v_target = vu / 255.0
                    except Exception:
                        pass

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
                    if hasattr(v_out_logits, 'shape'):
                        pid = getattr(agent, 'player_id', 0)
                        if 0 <= pid < v_out_logits.shape[0]:
                            v_pred_raw = v_out_logits[pid]
                        else:
                            v_pred_raw = v_out_logits[0]
                    else:
                        v_pred_raw = v_out_logits

                try:
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
                        logits_t = _t.tensor(logits_list[:n], dtype=_t.float32)

                    log_probs = logits_t.log_softmax(dim=0)
                    probs = log_probs.exp()

                    pi_t = _t.tensor(pi_target, dtype=_t.float32, device=log_probs.device)
                    if pi_t.shape[0] != log_probs.shape[0]:
                        m = min(pi_t.shape[0], log_probs.shape[0])
                        pi_t = pi_t[:m]
                        log_probs = log_probs[:m]
                        probs = probs[:m]

                    policy_loss = -(pi_t * log_probs).sum()
                    if isinstance(v_pred_raw, float):
                        v_logit = _t.tensor(v_pred_raw, dtype=_t.float32)
                    else:
                        v_logit = v_pred_raw.float()
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
                    # --- 手札予測損失 ---
                    try:
                        if getattr(agent.model, 'enable_hand_prediction_head', False) and hasattr(agent.model, 'forward_with_belief'):
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
                                    hl_t = _t.from_numpy(hl).to(hand_logits.device)
                                    # pad/truncate 次元調整
                                    n = min(int(hl_t.numel()), int(hand_logits.numel()))
                                    if n > 0:
                                        if hl_t.numel() != n:
                                            hl_t = hl_t[:n]
                                        if hand_logits.numel() != n:
                                            hand_logits = hand_logits.view(-1)[:n]
                                        if 'bce_logits_loss_fn_hand' not in agent.__dict__:
                                            import torch.nn as _nn
                                            agent.bce_logits_loss_fn_hand = _nn.BCEWithLogitsLoss()
                                        hp_loss = agent.bce_logits_loss_fn_hand(hand_logits.view(-1)[:n], hl_t.view(-1)[:n])
                                        hand_losses.append(hp_loss)
                    except Exception:
                        pass
                except Exception:
                    continue

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
    hand_loss_mean = _t.stack(hand_losses).mean() if hand_losses else None

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
    return out
