"""Training logic extracted from agents.drl_agent.

Goal: keep AlphaZeroAgent lean by delegating the heavy `train_step` and related
DataLoader helpers into a separate module (SRP).

This module is intentionally agent-implementation-agnostic: it operates on the
passed agent object and only assumes the agent exposes the attributes/methods
used by the original implementation.
"""

from __future__ import annotations

import abc
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Inference Engine Abstraction
# ============================================================

class InferenceClient(abc.ABC):
    """Abstract interface for policy-value inference.
    
    Decouples the inference backend (local GPU/CPU, remote queue, etc.)
    from the agent's decision logic.
    """
    
    @abc.abstractmethod
    def infer_policy_value(self, state: Dict[str, Any], legal_actions: List[Any]) -> Tuple[Dict[Any, float], float]:
        """Return (policy_dict, value_scalar) for the given state.
        
        Args:
            state: Canonicalized state representation
            legal_actions: List of legal actions (variable length)
            
        Returns:
            policy_dict: {action -> probability}
            value_scalar: Win probability (0..1) or dict per player
        """
        pass


class LocalInferenceClient(InferenceClient):
    """Direct local inference using the agent's model (GPU/CPU)."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def infer_policy_value(self, state: Dict[str, Any], legal_actions: List[Any]) -> Tuple[Dict[Any, float], float]:
        """Execute forward pass on local model."""
        model = self.agent.model
        if model is None:
            # Fallback: uniform policy, neutral value
            n = len(legal_actions) if legal_actions else 1
            # リスト型のアクションを tuple に変換してキーとして使用可能にする
            policy = {(tuple(a) if isinstance(a, list) else a): 1.0 / n for a in legal_actions} if legal_actions else {}
            return policy, 0.5
        
        # Variable-length model
        if getattr(model, 'supports_variable_actions', False) and hasattr(model, 'evaluate'):
            try:
                import torch
                model.eval()
                with torch.no_grad():
                    policy_dict, value_scalar = model.evaluate(state, legal_actions)
                    return policy_dict, value_scalar
            except (RuntimeError, AttributeError) as e:
                # Model evaluation failed, fall through to fixed-head or fallback
                pass
        
        # Fixed-head model: forward with state, then map to legal actions
        try:
            import torch
            model.eval()
            with torch.no_grad():
                logits_raw, value_raw = model.forward(state)
                
                # Handle value output format (keep as list/dict if multi-player)
                if isinstance(value_raw, dict):
                    value_scalar = {int(k): float(v) for k, v in value_raw.items()}
                elif hasattr(value_raw, 'tolist'):
                    # Tensor -> list
                    try:
                        value_scalar = value_raw.sigmoid().detach().cpu().tolist()
                    except (RuntimeError, AttributeError):
                        value_scalar = value_raw.detach().cpu().tolist()
                elif isinstance(value_raw, (list, tuple)):
                    value_scalar = [float(x) for x in value_raw]
                else:
                    # Single scalar value
                    value_scalar = float(value_raw)
                
                # Convert logits to policy dict
                if hasattr(logits_raw, 'cpu'):
                    logits = logits_raw.cpu().numpy().tolist()
                elif isinstance(logits_raw, (list, tuple)):
                    logits = list(logits_raw)
                else:
                    logits = [float(logits_raw)]
                
                # Align with legal actions
                n = len(legal_actions)
                if len(logits) < n:
                    logits.extend([float('-inf')] * (n - len(logits)))
                elif len(logits) > n:
                    logits = logits[:n]
                
                # Softmax
                import math
                mx = max(logits) if logits else 0.0
                exps = [math.exp(x - mx) for x in logits]
                s = sum(exps)
                probs = [e / s for e in exps] if s > 0 else [1.0 / n] * n
                
                # legal_actions は List[List[str]] だが、辞書のキーは hashable である必要があるため tuple に変換
                policy_dict = {(tuple(legal_actions[i]) if isinstance(legal_actions[i], list) else legal_actions[i]): probs[i] for i in range(n)}
                return policy_dict, value_scalar
                
        except (ImportError, RuntimeError, AttributeError, ValueError) as e:
            # Model forward failed - log if possible and use fallback
            # Note: cannot reliably access agent.logger here, so just use fallback
            n = len(legal_actions) if legal_actions else 1
            # Fallback でも同様に tuple に変換
            policy = {(tuple(a) if isinstance(a, list) else a): 1.0 / n for a in legal_actions} if legal_actions else {}
            return policy, 0.5


class RemoteInferenceClient(InferenceClient):
    """Queue-based remote inference for distributed training."""
    
    def __init__(self, agent):
        self.agent = agent
        self.timeout = float(agent.config.get('remote_infer_timeout', 5.0) or 5.0)
    
    def infer_policy_value(self, state: Dict[str, Any], legal_actions: List[Any]) -> Tuple[Dict[Any, float], float]:
        """Send inference request to master process via queue."""
        try:
            import uuid
            rq = getattr(self.agent, '_remote_request_q', None)
            rsp_q = getattr(self.agent, '_remote_response_q', None)
            wid = getattr(self.agent, '_remote_worker_id', None)
            
            if rq is None or rsp_q is None or wid is None:
                # Fallback to local if queue not configured
                return LocalInferenceClient(self.agent).infer_policy_value(state, legal_actions)
            
            # Send request with unique ID
            req_id = uuid.uuid4().hex
            try:
                rq.put((int(wid), req_id, state))
            except (OSError, ValueError, TypeError):
                # Queue put failed - fallback to local
                return LocalInferenceClient(self.agent).infer_policy_value(state, legal_actions)
            
            # Wait for response
            start = time.time()
            while (time.time() - start) < self.timeout:
                try:
                    rid, out = rsp_q.get(timeout=0.5)
                    if rid == req_id:
                        # out is (policy_dict, value_scalar) or similar
                        if isinstance(out, tuple) and len(out) == 2:
                            return out[0], out[1]
                        elif isinstance(out, dict):
                            return out, 0.5
                        else:
                            break
                except (OSError, ValueError, EOFError):
                    # Queue get failed or timeout
                    continue
            
            # Timeout: fallback to local
            return LocalInferenceClient(self.agent).infer_policy_value(state, legal_actions)
            
        except (AttributeError, ImportError, RuntimeError):
            # Critical error in remote setup - fallback to local
            return LocalInferenceClient(self.agent).infer_policy_value(state, legal_actions)


# ============================================================
# Value Encoding/Decoding
# ============================================================

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


def _extract_value(sample: Any) -> Optional[float]:
    getter = None
    try:
        if hasattr(sample, 'get'):
            getter = sample.get
    except Exception:
        getter = None
    if getter is None:
        return None
    try:
        v = getter('value', None)
    except Exception:
        v = None
    if v is not None:
        try:
            return float(v)
        except Exception:
            return None
    try:
        vu = getter('value_u8', None)
    except Exception:
        vu = None
    return _decode_value_u8(vu)


def _has_value_label(sample: Any) -> bool:
    getter = None
    try:
        if isinstance(sample, dict) or hasattr(sample, 'get'):
            getter = sample.get
    except Exception:
        getter = None
    if getter is None:
        return False
    try:
        if getter('value', None) is not None:
            return True
    except Exception:
        pass
    try:
        vu = getter('value_u8', None)
    except Exception:
        vu = None
    if isinstance(vu, int):
        # VALUE_U8_NONE (255) is reserved for "unset"; valid values are 0-254
        return 0 <= int(vu) < VALUE_U8_NONE
    return False


# ---------------- DataLoader helpers (top-level for picklable workers) ----------------
try:
    import torch
    from torch.utils.data import Dataset, Sampler
except Exception:  # pragma: no cover
    Dataset = object  # type: ignore
    Sampler = object  # type: ignore


class ReplaySnapshotDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], uid2weight: Optional[Dict[int, float]] = None):
        # A plain list snapshot; workers will index into this list
        self.samples = samples
        self.uid2weight = uid2weight or {}

    def __len__(self):  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]
        import numpy as _np
        # π 復元 (u16規格優先)
        if 'pi_q' in s and s.get('pi_format') == 'u16_norm65535':
            pi_q = s.get('pi_q')
            try:
                arr = pi_q.astype(_np.float32, copy=False) if hasattr(pi_q, 'astype') else _np.asarray(list(pi_q), dtype=_np.float32)
                s_q = float(arr.sum())
                pi_arr = (arr / s_q) if s_q > 0 else (_np.ones_like(arr, dtype=_np.float32) / max(1, arr.size))
            except Exception:
                raw = s.get('pi')
                pi_arr = _np.asarray(list(raw), dtype=_np.float32) if raw else _np.zeros((0,), dtype=_np.float32)
        else:
            raw = s.get('pi')
            pi_arr = _np.asarray(list(raw), dtype=_np.float32) if raw else _np.zeros((0,), dtype=_np.float32)
        # value target
        v_target = _extract_value(s)
        v_target = float(v_target) if v_target is not None else None
        uid = s.get('uid')
        is_w = 1.0
        try:
            if uid is not None:
                is_w = float(self.uid2weight.get(int(uid), 1.0))
        except Exception:
            is_w = 1.0
        return {
            'state': s.get('state') or {},
            'pi_arr': pi_arr,
            'v_target': v_target,
            'uid': int(uid) if uid is not None else -1,
            'is_w': is_w,
            'hand_size': int(s.get('hand_size')) if isinstance(s.get('hand_size', None), (int, float)) else (s.get('hand_size') if s.get('hand_size') is not None else -1),
            'is_terminal': bool(s.get('is_terminal', False)),
        }


class RawSampleDataset(Dataset):
    """Raw sample dataset: workers just fetch original sample dicts.

    DataLoader の collate_fn で Agent._prepare_batch を呼び出すため、ここでは加工しない。
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # pragma: no cover - trivial
        return self.samples[index]


class PreselectedSampler(Sampler):
    def __init__(self, indices: List[int]):
        self.indices = list(indices)

    def __iter__(self) -> Iterable[int]:  # pragma: no cover - simple
        return iter(self.indices)

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.indices)


def collate_prepared(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as _np
    states = [it['state'] for it in batch_items]
    pi_arrays = [it['pi_arr'] for it in batch_items]
    v_targets = [float(it['v_target']) for it in batch_items]
    is_weights = [float(it.get('is_w', 1.0)) for it in batch_items]
    uids = [int(it.get('uid', -1)) for it in batch_items]
    lengths = [int(arr.shape[0]) for arr in pi_arrays]
    return {
        'states': states,
        'pi_arrays': pi_arrays,
        'v_targets': v_targets,
        'is_weights': is_weights,
        'uids': uids,
        'lengths': lengths,
    }


def collate_prepare_batch(batch_items: List[Dict[str, Any]], agent_ref: Any, uid2weight: Optional[Dict[int, float]] = None, fast_full_input_np=None) -> Dict[str, Any]:
    """トップレベル関数: functools.partial で Agent 参照と重みを束縛して DataLoader workers から利用可能。

    Returns same dict as _prepare_batch. fast_full_input_np は通常 None (fast path 未使用)。
    """
    try:
        return agent_ref._prepare_batch(batch_items, uid2weight=uid2weight, fast_full_input_np=fast_full_input_np)  # type: ignore[attr-defined]
    except Exception:
        # フォールバック: 以前の加工済み形式に類似した最低限の構造
        import numpy as _np
        states = [s.get('state') for s in batch_items]
        pi_arrays = []
        v_targets = []
        is_weights = []
        lengths = []
        for s in batch_items:
            raw = s.get('pi') or []
            arr = _np.asarray(list(raw), dtype=_np.float32) if raw else _np.zeros((0,), dtype=_np.float32)
            if arr.size == 0:
                continue
            pi_arrays.append(arr)
            lengths.append(arr.shape[0])
            v = None
            vu = s.get('value_u8')
            if isinstance(vu, int) and vu != VALUE_U8_NONE:
                v = _decode_value_u8(vu)
            if v is None:
                continue
            v_targets.append(float(v))
            uid = s.get('uid')
            try:
                iw = float(uid2weight.get(int(uid), 1.0)) if (uid2weight is not None and uid is not None) else 1.0
            except Exception:
                iw = 1.0
            is_weights.append(iw)
        return {
            'states': states,
            'pi_arrays': pi_arrays,
            'v_targets': v_targets,
            'is_weights': is_weights,
            'hand_labels': [],
            'used_uids': [],
            'fast_full_input_np': fast_full_input_np,
        }


class TrainStepMixin:
    # ---------------- Batch Preparation ----------------
    def _prepare_batch(self, batch: List[Dict[str, Any]], uid2weight: Optional[Dict[int, float]] = None, fast_full_input_np=None) -> Dict[str, Any]:
        """生サンプル辞書群 -> モデル forward_batch 用の前処理結果をまとめて返す。

        戻り値キー:
          states, pi_arrays, v_targets, is_weights, lengths, hand_labels, used_uids

        役割:
          - π 量子化形式(u16_norm65535) 正規化
          - value_u8 -> value (0..1) 復元
          - full feature ゼロパディング除外 (設定で有効時)
          - importance weight (prioritized) 付与
        """
        import numpy as _np

        states: List[Any] = []
        pi_arrays: List[_np.ndarray] = []
        v_targets: List[float] = []
        is_weights: List[float] = []
        lengths: List[int] = []
        hand_labels: List[Optional[_np.ndarray]] = []
        used_uids: List[int] = []
        skip_full = bool(self.config.get('use_full_features') and self.config.get('skip_zero_padded_full_samples', True))
        for sample in batch:
            if not isinstance(sample, dict):
                continue
            v_target = _extract_value(sample)
            if v_target is None:
                continue
            # full feature 空サンプル除外
            if skip_full:
                st_meta = sample.get('state') or {}
                if ('full_compact' not in st_meta) and ('full_input' not in st_meta):
                    continue
            # π 復元
            try:
                if 'pi_q' in sample and sample.get('pi_format') == 'u16_norm65535':
                    pi_q = sample.get('pi_q')
                    if hasattr(pi_q, 'astype'):
                        arr = pi_q.astype(_np.float32, copy=False)
                    else:
                        arr = _np.asarray(list(pi_q), dtype=_np.float32)
                    s_q = float(arr.sum())
                    if s_q > 0:
                        pi_arr = arr / s_q
                    else:
                        if arr.size == 0:
                            continue
                        pi_arr = _np.ones_like(arr, dtype=_np.float32) / arr.size
                else:
                    raw_pi = sample.get('pi')
                    if not raw_pi:
                        continue
                    pi_arr = _np.asarray(list(raw_pi), dtype=_np.float32)
            except Exception:
                raw_pi = sample.get('pi')
                if not raw_pi:
                    continue
                pi_arr = _np.asarray(list(raw_pi), dtype=_np.float32)
            if pi_arr.size == 0:
                continue
            # canonicalize state so stored viewpoint becomes self=0 (best-effort)
            try:
                st_raw = sample.get('state') or {}
                # capture original stored self pid for label rotation
                try:
                    s_pid = int(st_raw.get('self_player_id', st_raw.get('player_id', 0)))
                except Exception:
                    s_pid = int(getattr(self, 'player_id', 0) or 0)
                st_can = self._canonicalize_state(st_raw)  # type: ignore[attr-defined]
            except Exception:
                st_can = sample.get('state')
                try:
                    s_pid = int(sample.get('state', {}).get('self_player_id', sample.get('player_id', 0)))
                except Exception:
                    s_pid = int(getattr(self, 'player_id', 0) or 0)
            # Policy actions are absolute card identifiers (not relative indices),
            # so do not attempt to rotate/roll the policy target by player chunks.
            states.append(st_can)
            pi_arrays.append(pi_arr)
            v_targets.append(float(v_target))
            lengths.append(int(pi_arr.shape[0]))
            # hand labels (存在時のみ) -- use canonicalized state so labels follow viewpoint rotation
            hl = None
            try:
                st_meta = st_can or {}
                if isinstance(st_meta, dict) and ('hand_labels' in st_meta):
                    hl_raw = st_meta.get('hand_labels')
                    if hl_raw is not None:
                        hl = _np.asarray(list(hl_raw), dtype=_np.float32)
            except Exception:
                hl = None
            hand_labels.append(hl)
            # importance weight
            try:
                uid = sample.get('uid')
                iw = float(uid2weight.get(int(uid), 1.0)) if (uid2weight is not None and uid is not None) else 1.0
            except Exception:
                iw = 1.0
            is_weights.append(iw)
            try:
                if uid is not None:
                    used_uids.append(int(uid))
            except Exception:
                pass
        return {
            'states': states,
            'pi_arrays': pi_arrays,
            'v_targets': v_targets,
            'is_weights': is_weights,
            'lengths': lengths,
            'hand_labels': hand_labels,
            'used_uids': used_uids,
            'fast_full_input_np': fast_full_input_np,
        }

    # ---------------- Training ----------------
    def train_step(self, batch_size: int = 64):
        """サンプルの一部で1ステップ学習し各種メトリクスを返す。value=None のものは除外。

        戻り値: dict(loss, policy_loss, value_loss, entropy, 追加統計...)
        代表的失敗理由: torch_not_installed / no_model / no_data / no_valid_samples
        """
        self._logged_inside = False
        try:
            import torch
        except ImportError:
            return {"loss": None, "reason": "torch_not_installed"}
        if self.model is None:
            return {"loss": None, "reason": "no_model"}
        # 共有バッファ: 自プレイヤーの確定サンプルのみ抽出 (学習splitのみ)
        if self._use_shared and hasattr(self.replay_buffer, 'iter_all'):
            my_samples = [s for s in self.replay_buffer.iter_all(owner_pid=self.player_id)
                          if _has_value_label(s) and (s.get('split') != 'val')]
            if not my_samples:
                return {"loss": None, "reason": "no_data"}
            batch_pool = my_samples
        else:  # ローカル
            if not self.replay_buffer:
                return {"loss": None, "reason": "no_data"}
            try:
                batch_pool = [s for s in self.replay_buffer if isinstance(s, dict) and _has_value_label(s) and (s.get('split') != 'val')]
            except Exception:
                batch_pool = self.replay_buffer

        # Optimizer 遅延初期化（Value Head専用の正則化を適用）
        if self._optimizer is None:
            self.ensure_optimizer()

        # --- Prioritized sampling (lightweight, conservative) ---
        uid2weight = None
        if (self._use_shared and hasattr(self.replay_buffer, 'sample_prioritized')
                and bool(self.config.get('prioritized_replay', True))):
            try:
                alpha = float(self.config.get('prioritized_replay_alpha', 0.6) or 0.6)
                eps = float(self.config.get('prioritized_replay_eps', 1e-6) or 1e-6)
                fast_full_input_np = None
                if hasattr(self.replay_buffer, 'sample_prioritized_fast'):
                    try:
                        fast_res = self.replay_buffer.sample_prioritized_fast(
                            batch_size, owner_pid=self.player_id, alpha=alpha, eps=eps,
                            return_weights=True, return_full_input=True)
                    except Exception:
                        fast_res = None
                else:
                    fast_res = None
                if isinstance(fast_res, dict) and fast_res.get('samples') is not None:
                    sampled = fast_res['samples']
                    sampled_uids = fast_res.get('uids', [])
                    is_weights = fast_res.get('is_weights', [])
                    fast_full_input_np = fast_res.get('full_input')
                else:
                    sampled, sampled_uids, is_weights = self.replay_buffer.sample_prioritized(
                        batch_size, owner_pid=self.player_id, alpha=alpha, eps=eps, return_weights=True)
                # map uid -> importance weight (normalized to max=1 by buffer)
                uid2weight = {int(u): float(w) for u, w in zip(sampled_uids, is_weights)}
                batch = sampled
            except Exception:
                batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
                fast_full_input_np = None
        else:
            batch = batch_pool if len(batch_pool) <= batch_size else random.sample(batch_pool, batch_size)
            fast_full_input_np = None
        # フル特徴量モード時に旧フォーマット(feature_version=0)サンプルを除外
        if self.config.get('use_full_features'):
            filtered = [s for s in batch if s.get('feature_version', 0) >= 1]
            if not filtered:
                return {"loss": None, "reason": "no_full_feature_samples"}
            batch = filtered
        # --- クラス別ミックス (陽性率の最低保証) ---
        try:
            import math as _m
            target_pos_rate = float(self.config.get('value_pos_min_rate', 0.2) or 0.2)
        except Exception:
            target_pos_rate = 0.0
        if target_pos_rate > 0.0 and batch:
            def _is_pos(sample):
                v = _extract_value(sample)
                try:
                    return (v is not None) and (float(v) > 0.5)
                except Exception:
                    return False
            cur_pos = sum(1 for s in batch if _is_pos(s))
            B = len(batch)
            max_frac = float(self.config.get('value_pos_topup_max_frac', 0.35) or 0.35)
            target_count = int(_m.ceil(min(max_frac, max(0.0, target_pos_rate)) * B))
            need = max(0, target_count - cur_pos)
            if need > 0:
                # 候補プール: 同一オーナー/学習splitの全サンプルから陽性のみ抽出
                try:
                    # batch_pool は先に作成済みの学習候補プール
                    candidates = [s for s in batch_pool if _is_pos(s)]
                except Exception:
                    candidates = []
                # 過学習対策: 直近使用UIDの除外 / 現バッチのUID除外 / 新し過ぎるUID除外
                in_batch_uids = set()
                for s in batch:
                    u = s.get('uid', None)
                    if u is not None:
                        try:
                            in_batch_uids.add(int(u))
                        except Exception:
                            pass
                recent_set = set(list(self._recent_pos_uids)) if isinstance(self._recent_pos_uids, (list, set)) else set(list(getattr(self, '_recent_pos_uids', [])))
                try:
                    max_uid_all = max(int(s.get('uid', -1)) for s in candidates if s.get('uid') is not None) if candidates else -1
                except Exception:
                    max_uid_all = -1
                uid_age_margin = int(self.config.get('value_pos_uid_age_margin', 1000) or 1000)
                filtered_cands = []
                for s in candidates:
                    uid = s.get('uid', None)
                    try:
                        uid_i = int(uid) if uid is not None else None
                    except Exception:
                        uid_i = None
                    if uid_i is not None:
                        if uid_i in in_batch_uids:
                            continue
                        if uid_i in recent_set:
                            continue
                        if (uid_age_margin > 0) and (max_uid_all >= 0) and (uid_i > (max_uid_all - uid_age_margin)):
                            # 新規すぎるサンプルは避ける（多様性確保）
                            continue
                    filtered_cands.append(s)
                # 最低候補数が小さい場合はスキップ（極端な再利用を避ける）
                min_cands = int(self.config.get('value_pos_topup_min_candidates', 200) or 200)
                if len(filtered_cands) >= max(min_cands, need):
                    # ミックス: 一部は一様ランダム、残りは優先度重み付き
                    import random as _r
                    uni_ratio = float(self.config.get('value_pos_topup_uniform_mix', 0.3) or 0.3)
                    take_uni = int(round(need * max(0.0, min(1.0, uni_ratio))))
                    take_pri = max(0, need - take_uni)
                    _r.shuffle(filtered_cands)
                    top_uni = filtered_cands[:take_uni]
                    remain_cands = filtered_cands[take_uni:]
                    # 重み: priority^alpha（無ければ1.0）
                    try:
                        alpha = float(self.config.get('prioritized_replay_alpha', 0.6) or 0.6)
                    except Exception:
                        alpha = 0.6
                    def _prio_w(s):
                        try:
                            p = float(s.get('priority', 1.0) or 1.0)
                        except Exception:
                            p = 1.0
                        try:
                            return max(1e-8, p) ** alpha
                        except Exception:
                            return 1.0
                    weights = [_prio_w(s) for s in remain_cands]
                    # 重み付きサンプル（置換なし）
                    top_pri = []
                    if take_pri > 0 and remain_cands:
                        # 簡易: サンプリング毎に正規化して選択
                        pool = list(remain_cands)
                        w = list(weights)
                        for _ in range(min(take_pri, len(pool))):
                            s_w = sum(w)
                            if s_w <= 0:
                                idx = _r.randrange(len(pool))
                            else:
                                r = _r.random() * s_w
                                acc = 0.0
                                idx = 0
                                for j, ww in enumerate(w):
                                    acc += ww
                                    if acc >= r:
                                        idx = j
                                        break
                            top_pri.append(pool.pop(idx))
                            _ = w.pop(idx)
                    topups = top_uni + top_pri
                    # 負例を置換して陽性率を引き上げる
                    if topups:
                        neg_indices = [i for i, s in enumerate(batch) if not _is_pos(s)]
                        if neg_indices:
                            # replace up to available slots
                            replace_n = min(len(topups), len(neg_indices))
                            # 置換でfast_full_inputは不整合になるので無効化
                            fast_full_input_np = None
                            for k in range(replace_n):
                                idx = neg_indices[k]
                                batch[idx] = topups[k]
                                # IS重みがある場合は新規UIDを1.0で登録（保守的）
                                try:
                                    if uid2weight is not None:
                                        uid_new = topups[k].get('uid')
                                        if uid_new is not None:
                                            uid2weight[int(uid_new)] = 1.0
                                except Exception:
                                    pass
                                # 最近使用UIDとして登録
                                try:
                                    uid_reg = topups[k].get('uid')
                                    if uid_reg is not None and hasattr(self, '_recent_pos_uids') and hasattr(self._recent_pos_uids, 'append'):
                                        self._recent_pos_uids.append(int(uid_reg))
                                except Exception:
                                    pass
        # --- ここまで: クラス別ミックス ---
        # 損失集計用のコンテナ
        policy_losses = []
        value_losses = []
        hand_losses = []
        entropies = []
        valid = 0
        collected_pi = []
        collected_model = []
        collected_v_pred = []
        collected_v_t = []
        collected_hand_pos_rate = []
        # accumulate raw hand prediction tensors for metrics (per-sample)
        collected_hand_logits = []
        collected_hand_tgts = []
        collected_hand_masks = []
        used_uids = []
        variable = getattr(self.model, 'supports_variable_actions', False) and hasattr(self.model, 'evaluate')

        # Optional DataLoader preparation (indices sampled in main process)
        use_loader = bool(self.config.get('dataloader_num_workers', 0) or 0) > 0 and not variable
        prepared_batch = None
        if use_loader:
            try:
                from torch.utils.data import DataLoader
                # Eligible snapshot
                if self._use_shared and hasattr(self.replay_buffer, 'iter_all'):
                    pool_snapshot = [s for s in self.replay_buffer.iter_all(owner_pid=self.player_id)
                                     if _has_value_label(s) and (s.get('split') != 'val')]
                else:
                    src = self.replay_buffer or []
                    pool_snapshot = [s for s in src if isinstance(s, dict) and _has_value_label(s) and (s.get('split') != 'val')]
                if self.config.get('use_full_features'):
                    pool_snapshot = [s for s in pool_snapshot if s.get('feature_version', 0) >= 1]
                if not pool_snapshot:
                    return {"loss": None, "reason": "no_data"}
                # uid -> idx mapping
                uid2idx: Dict[int, int] = {}
                for i, s in enumerate(pool_snapshot):
                    u = s.get('uid')
                    if u is not None and (u not in uid2idx):
                        try:
                            uid2idx[int(u)] = i
                        except Exception:
                            pass
                # prioritized on buffer
                uid2weight_local: Dict[int, float] = {}
                if (self._use_shared and hasattr(self.replay_buffer, 'sample_prioritized_fast') and bool(self.config.get('prioritized_replay', True))):
                    try:
                        alpha = float(self.config.get('prioritized_replay_alpha', 0.6) or 0.6)
                        eps = float(self.config.get('prioritized_replay_eps', 1e-6) or 1e-6)
                    except Exception:
                        alpha, eps = 0.6, 1e-6
                    fast = self.replay_buffer.sample_prioritized_fast(batch_size, owner_pid=self.player_id, alpha=alpha, eps=eps, return_weights=True, return_full_input=False)
                    sel_uids = fast.get('uids', []) if isinstance(fast, dict) else []
                    is_ws = fast.get('is_weights', []) if isinstance(fast, dict) else []
                    for u, w in zip(sel_uids, is_ws):
                        try:
                            uid2weight_local[int(u)] = float(w)
                        except Exception:
                            continue
                    sel_indices = [uid2idx[u] for u in sel_uids if u in uid2idx]
                else:
                    import random as _r
                    if len(pool_snapshot) <= batch_size:
                        sel_indices = list(range(len(pool_snapshot)))
                    else:
                        sel_indices = _r.sample(range(len(pool_snapshot)), batch_size)
                # --- Positive rate top-up (Sampler level) ---
                try:
                    import math as _m, random as _r
                    target_pos_rate = float(self.config.get('value_pos_min_rate', 0.2) or 0.2)
                except Exception:
                    target_pos_rate = 0.0
                if target_pos_rate > 0.0 and sel_indices:
                    def _is_pos_idx(idx: int) -> bool:
                        s = pool_snapshot[idx]
                        v = _extract_value(s)
                        try:
                            return (v is not None) and (float(v) > 0.5)
                        except Exception:
                            return False
                    cur_pos = sum(1 for i in sel_indices if _is_pos_idx(i))
                    Bsel = len(sel_indices)
                    try:
                        max_frac = float(self.config.get('value_pos_topup_max_frac', 0.35) or 0.35)
                    except Exception:
                        max_frac = 0.35
                    target_count = int(_m.ceil(min(max_frac, max(0.0, target_pos_rate)) * Bsel))
                    need = max(0, target_count - cur_pos)
                    if need > 0:
                        # Candidates: positives not already selected
                        selected_set = set(sel_indices)
                        recent_set = set(list(getattr(self, '_recent_pos_uids', []))) if hasattr(self, '_recent_pos_uids') else set()
                        # Collect indices of positive samples not selected
                        pos_pool = []
                        for idx, s in enumerate(pool_snapshot):
                            if idx in selected_set:
                                continue
                            if not _is_pos_idx(idx):
                                continue
                            uid = s.get('uid')
                            try:
                                if uid is not None and int(uid) in recent_set:
                                    continue
                            except Exception:
                                pass
                            pos_pool.append(idx)
                        try:
                            min_cands = int(self.config.get('value_pos_topup_min_candidates', 200) or 200)
                        except Exception:
                            min_cands = 200
                        if len(pos_pool) >= max(min_cands, need):
                            _r.shuffle(pos_pool)
                            add_indices = pos_pool[:need]
                            # Replace negatives in current selection
                            neg_indices_local = [i for i in sel_indices if not _is_pos_idx(i)]
                            replace_n = min(len(add_indices), len(neg_indices_local))
                            for k in range(replace_n):
                                old_idx = neg_indices_local[k]
                                new_idx = add_indices[k]
                                # Swap old_idx -> new_idx in sel_indices
                                try:
                                    pos_slot = sel_indices.index(old_idx)
                                    sel_indices[pos_slot] = new_idx
                                except Exception:
                                    continue
                                uid_new = pool_snapshot[new_idx].get('uid')
                                if uid_new is not None:
                                    try:
                                        uid2weight_local[int(uid_new)] = 1.0  # conservative IS weight
                                        if hasattr(self, '_recent_pos_uids') and hasattr(self._recent_pos_uids, 'append'):
                                            self._recent_pos_uids.append(int(uid_new))
                                    except Exception:
                                        pass
                # --- end top-up ---
                # Dataset / Loader (raw samples + top-level collate using _prepare_batch via partial)
                from functools import partial
                ds = RawSampleDataset(pool_snapshot)
                sp = PreselectedSampler(sel_indices)
                num_workers = int(self.config.get('dataloader_num_workers', 0) or 0)
                prefetch = int(self.config.get('dataloader_prefetch_factor', 2) or 2)
                import torch as _t
                pin = bool(self.config.get('dataloader_pin_memory', True) and _t.cuda.is_available())
                collate_fn = partial(collate_prepare_batch, agent_ref=self, uid2weight=uid2weight_local, fast_full_input_np=None)
                _dl_kwargs = dict(batch_size=len(sel_indices), sampler=sp, num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn, drop_last=False)
                if num_workers > 0:
                    _dl_kwargs['prefetch_factor'] = prefetch
                dl = DataLoader(ds, **_dl_kwargs)
                prepared_batch = next(iter(dl))
            except Exception:
                prepared_batch = None
                use_loader = False

        vectorized_ok = False
        if not variable and hasattr(self.model, 'forward_batch'):
            try:
                import numpy as _np
                if prepared_batch is not None:
                    states = prepared_batch['states']
                    pi_arrays = prepared_batch['pi_arrays']
                    v_targets_list = prepared_batch['v_targets']
                    is_weights_list = prepared_batch['is_weights']
                    lengths = prepared_batch['lengths']
                    used_uids.extend(prepared_batch.get('used_uids', []))
                else:
                    prep = self._prepare_batch(batch, uid2weight, fast_full_input_np)
                    states = prep['states']
                    pi_arrays = prep['pi_arrays']
                    v_targets_list = prep['v_targets']
                    is_weights_list = prep['is_weights']
                    lengths = prep['lengths']
                    used_uids.extend(prep['used_uids'])
                if states:
                    # モデル一括 forward（belief ヘッド出力も取得）
                    import torch
                    try:
                        if fast_full_input_np is not None:
                            import torch as _t
                            xs = _t.from_numpy(fast_full_input_np).to(self.model.device).float()
                        else:
                            xs = torch.stack([self.model._encode_state(s) for s in states], dim=0)
                    except Exception:
                        xs = None
                    # Optional: log input diversity statistics for debugging
                    try:
                        if xs is not None and bool(self.config.get('debug_log_input_diversity', False)):
                            try:
                                with torch.no_grad():
                                    feat_std = xs.std(dim=0)
                                    mean_std = float(feat_std.mean().detach().cpu().item())
                                    median_std = float(feat_std.median().detach().cpu().item())
                                    zero_frac = float((feat_std.detach().cpu() < 1e-6).float().mean().item())
                                print(f"[INPUT_DIVERSITY] B={xs.size(0)} mean_feat_std={mean_std:.6e} median_feat_std={median_std:.6e} zero_frac={zero_frac:.6f}")
                            except Exception as _e:
                                print(f"[INPUT_DIVERSITY] failed: {_e}")
                    except Exception:
                        pass

                    if xs is not None and hasattr(self.model, 'forward_with_belief'):
                        policy_logits_batch, value_logits_batch, hand_logits_batch = self.model.forward_with_belief(xs)
                    else:
                        policy_logits_batch, value_logits_batch = self.model.forward_batch(states)
                        hand_logits_batch = None
                    device = policy_logits_batch.device
                    max_len = max(lengths)
                    # Cap target/padding length to model's policy output width to avoid
                    # mismatch when stored pi arrays are longer than policy_head output.
                    try:
                        policy_out_dim = int(policy_logits_batch.size(1)) if hasattr(policy_logits_batch, 'size') else None
                    except Exception:
                        policy_out_dim = None
                    if policy_out_dim is not None:
                        if max_len > policy_out_dim:
                            # Warn once to help debugging data/model mismatch
                            if not hasattr(self, '_vec_pi_len_warned'):
                                try:
                                    print(f"[WARN] train_step: pi length {max_len} > model.policy_dim {policy_out_dim}; truncating targets")
                                except Exception:
                                    pass
                                self._vec_pi_len_warned = True  # type: ignore[attr-defined]
                        capped_max_len = min(max_len, policy_out_dim)
                    else:
                        capped_max_len = max_len
                    B = len(states)
                    # Pad π ターゲット
                    pi_pad = _np.zeros((B, capped_max_len), dtype=_np.float32)
                    for i, arr in enumerate(pi_arrays):
                        take = min(int(arr.shape[0]), capped_max_len)
                        if take > 0:
                            pi_pad[i, :take] = arr[:take]
                    pi_pad_t = torch.from_numpy(pi_pad).to(device)
                    # 有効幅を「ターゲット幅」と「モデル出力幅」の最小値に再度そろえる（形状不一致防止）
                    policy_width = int(policy_logits_batch.size(1)) if hasattr(policy_logits_batch, 'size') else capped_max_len
                    eff_width = min(int(pi_pad_t.size(1)), policy_width)
                    if eff_width <= 0:
                        raise RuntimeError("eff_width computed as 0 or negative")
                    if eff_width != pi_pad_t.size(1):
                        pi_pad_t = pi_pad_t[:, :eff_width]
                    # Truncate lengths to eff_width for mask construction
                    lengths_eff = [min(int(l), eff_width) for l in lengths]
                    lengths_t = torch.tensor(lengths_eff, device=device)
                    # logits のパディング処理: 余剰部を -inf 相当でマスク
                    logits_slice = policy_logits_batch[:, :eff_width]
                    # 長さ未満部分だけ使用するためマスクを構築
                    arange = torch.arange(eff_width, device=device).unsqueeze(0).expand(B, -1)
                    mask = (arange < lengths_t.unsqueeze(1)).float()
                    LARGE_NEG = -1e9
                    masked_logits = logits_slice * mask + (1 - mask) * LARGE_NEG
                    log_probs = torch.log_softmax(masked_logits, dim=1)
                    probs = torch.exp(log_probs) * mask  # パディング部ほぼ0
                    # policy loss (各行で sum)
                    policy_loss_all = - (pi_pad_t * log_probs).sum(dim=1)
                    # value ロジット選択: 各サンプルに埋め込まれた視点(self_player_id)に対応する列を選ぶ
                    # (保存時の player_id と学習エージェントの player_id が異なるバッチを許容するため)
                    try:
                        import torch as _t
                        # states は現在バッチの state dict リスト
                        pids_list = []
                        for st in states:
                            try:
                                if isinstance(st, dict) and ('self_player_id' in st):
                                    pids_list.append(int(st.get('self_player_id')))
                                else:
                                    # fallback: use agent.player_id if missing
                                    pids_list.append(int(getattr(self, 'player_id', 0)))
                            except Exception:
                                pids_list.append(int(getattr(self, 'player_id', 0)))
                        pids_t = _t.tensor(pids_list, device=device, dtype=_t.long)
                        # value_logits_batch: [B, num_players] (or variants)
                        if hasattr(value_logits_batch, 'dim') and value_logits_batch.dim() == 2:
                            num_cols = value_logits_batch.size(1)
                            # avoid out-of-range indices
                            max_pid = int(pids_t.max().item()) if pids_t.numel() > 0 else 0
                            # fast-path: if all pids are 0 (canonicalized), just take column 0
                            if pids_t.numel() > 0 and int(pids_t.max().item()) == 0:
                                try:
                                    v_logits = value_logits_batch[:, 0]
                                except Exception:
                                    v_logits = value_logits_batch.gather(1, pids_t.unsqueeze(1).clamp(max=num_cols - 1)).squeeze(1)
                            else:
                                if max_pid >= num_cols:
                                    # clamp indices to available columns (fallback)
                                    pids_norm = pids_t.clamp(max=num_cols - 1)
                                else:
                                    pids_norm = pids_t
                                v_logits = value_logits_batch.gather(1, pids_norm.unsqueeze(1)).squeeze(1)
                        elif hasattr(value_logits_batch, 'dim') and value_logits_batch.dim() == 1:
                            # rare: model returned (num_players,) for whole batch
                            sel = int(pids_list[0]) if pids_list else int(getattr(self, 'player_id', 0))
                            sel = max(0, min(sel, value_logits_batch.numel() - 1))
                            v_logits = value_logits_batch[sel].unsqueeze(0)
                        else:
                            # fallback
                            v_logits = value_logits_batch[:, 0]
                    except Exception:
                        try:
                            import torch as _t
                            if hasattr(value_logits_batch, 'shape') and len(value_logits_batch.shape) >= 2:
                                v_logits = value_logits_batch[:, 0]
                            elif hasattr(value_logits_batch, '__len__'):
                                v0 = value_logits_batch[0]
                                import torch as __t
                                try:
                                    v_logits = __t.tensor([float(v0)], device=device)
                                except Exception:
                                    v_logits = __t.tensor([float(v0)])
                            else:
                                import torch as _t
                                v_logits = _t.zeros(1)
                        except Exception:
                            import torch as _t
                            v_logits = _t.zeros(1)
                    # Optionally boost terminal samples via config 'terminal_sample_boost'
                    try:
                        term_boost = float(self.config.get('terminal_sample_boost', 3.0) or 3.0)
                    except Exception:
                        term_boost = 1.0
                    # Convert importance weights to a tensor on the model device
                    is_weights_t = torch.tensor(list(map(float, is_weights_list)), dtype=torch.float32, device=device).view(-1)

                    # Apply terminal-sample boost using a mask with identical length/device
                    if term_boost != 1.0:
                        try:
                            term_mask_list = []
                            for st in states:
                                try:
                                    is_term = False
                                    if isinstance(st, dict):
                                        is_term = bool(st.get('is_terminal', False)) or (int(st.get('hand_size', -1)) == 0)
                                    term_mask_list.append(1.0 if is_term else 0.0)
                                except Exception:
                                    term_mask_list.append(0.0)
                            term_mask_t = torch.tensor(term_mask_list, dtype=is_weights_t.dtype, device=is_weights_t.device).view(-1)
                            boost_factor = float(term_boost)
                            boost_vec = 1.0 + (boost_factor - 1.0) * term_mask_t
                            is_weights_t = is_weights_t * boost_vec
                        except Exception:
                            # If anything goes wrong, fall back to safe per-index multiply
                            for i_st, st in enumerate(states):
                                try:
                                    is_term = False
                                    if isinstance(st, dict):
                                        is_term = bool(st.get('is_terminal', False)) or (int(st.get('hand_size', -1)) == 0)
                                    if is_term and i_st < len(is_weights_list):
                                        is_weights_list[i_st] = float(is_weights_list[i_st]) * float(term_boost)
                                except Exception:
                                    pass
                            is_weights_t = torch.tensor(list(map(float, is_weights_list)), dtype=torch.float32, device=device).view(-1)

                    v_targets_t = torch.tensor(v_targets_list, dtype=torch.float32, device=device).view(-1)
                    # Debug: print value logits/targets distribution when requested
                    try:
                        if bool(self.config.get('debug_log_value_stats', False)):
                            try:
                                import numpy as _np
                                with torch.no_grad():
                                    vl = v_logits.detach().cpu().numpy() if hasattr(v_logits, 'detach') else _np.asarray(v_logits)
                                    vt = v_targets_t.detach().cpu().numpy() if hasattr(v_targets_t, 'detach') else _np.asarray(v_targets_t)
                                mean_vl = float(_np.mean(vl)) if vl.size else None
                                std_vl = float(_np.std(vl)) if vl.size else None
                                mean_vt = float(_np.mean(vt)) if vt.size else None
                                min_vt = float(_np.min(vt)) if vt.size else None
                                max_vt = float(_np.max(vt)) if vt.size else None
                                count = int(vt.size) if hasattr(vt, 'size') else int(len(vt))
                                try:
                                    count_ones = int(_np.sum(_np.isclose(vt, 1.0))) if vt.size else 0
                                except Exception:
                                    count_ones = int(sum(1 for x in vt if float(x) == 1.0)) if vt.size else 0
                                # print a small sample of pairs
                                pairs = []
                                for i in range(min(10, vl.size)):
                                    pairs.append(f"{vl[i]:.4f}->{vt[i]:.4f}")
                                print(f"[DEBUG_VALUE_STATS] mean_logit={mean_vl} std_logit={std_vl} mean_target={mean_vt} min_target={min_vt} max_target={max_vt} count={count} ones={count_ones} samples=[{', '.join(pairs)}]")
                            except Exception as _e:
                                print(f"[DEBUG_VALUE_STATS] failed: {_e}")
                    except Exception:
                        pass
                    # Monitor terminal predictions: warn if terminal samples predicted low
                    try:
                        if bool(self.config.get('monitor_terminal_predictions', True)):
                            import torch as _t
                            with torch.no_grad():
                                v_probs = _t.sigmoid(v_logits)
                                th = float(self.config.get('monitor_terminal_threshold', 0.6) or 0.6)
                                msgs = []
                                for i_st, st in enumerate(states):
                                    try:
                                        is_term = False
                                        # state from dataset may include is_terminal/hand_size
                                        if isinstance(st, dict):
                                            is_term = bool(st.get('is_terminal', False)) or (int(st.get('hand_size', -1)) == 0)
                                        if is_term:
                                            p = float(v_probs[i_st].cpu().item()) if hasattr(v_probs[i_st], 'cpu') else float(v_probs[i_st].item())
                                            vt = float(v_targets_t[i_st].cpu().item()) if hasattr(v_targets_t[i_st],'cpu') else float(v_targets_t[i_st])
                                            if p < th:
                                                uid = st.get('uid', None) if isinstance(st, dict) else None
                                                msgs.append(f"idx={i_st} uid={uid} hand_size={st.get('hand_size') if isinstance(st, dict) else 'NA'} pred={p:.4f} target={vt:.4f}")
                                    except Exception:
                                        continue
                                if msgs:
                                    txt = "; ".join(msgs[:20])
                                    if hasattr(self, 'logger') and getattr(self, 'logger', None):
                                        try:
                                            self.logger.log_text(f"[WARN][TERM_PRED_LOW] {txt}")
                                        except Exception:
                                            print(f"[WARN][TERM_PRED_LOW] {txt}")
                                    else:
                                        print(f"[WARN][TERM_PRED_LOW] {txt}")
                    except Exception:
                        pass
                    if 'bce_logits_loss_fn' not in self.__dict__:
                        import torch.nn as _nn
                        try:
                            pw = float(self.pos_weight)
                        except Exception:
                            pw = 1.0
                        pos_w_tensor = None
                        if pw != 1.0:
                            pos_w_tensor = torch.tensor([pw], dtype=torch.float32, device=device)
                        self.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()
                    # Compute per-sample BCE (no reduction) then apply IS weights conservatively to value loss only
                    try:
                        import torch.nn as _nn
                        # create a reduction='none' BCE with same pos_weight behavior
                        pos_w = None
                        try:
                            pos_w = float(self.pos_weight)
                        except Exception:
                            pos_w = 1.0
                        pos_w_tensor = None
                        if pos_w != 1.0:
                            pos_w_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
                        bce_none = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor, reduction='none') if pos_w_tensor is not None else _nn.BCEWithLogitsLoss(reduction='none')
                        value_loss_per = bce_none(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1)).squeeze(1)
                        # apply importance weights and average
                        value_loss_all = (value_loss_per * is_weights_t).mean()
                    except Exception:
                        # fallback to previous mean if anything goes wrong
                        value_loss_all = self.bce_logits_loss_fn(v_logits.unsqueeze(1), v_targets_t.unsqueeze(1))
                    # Entropy
                    entropy_all = - (probs * log_probs).sum(dim=1)
                    # 集約
                    policy_losses.append(policy_loss_all.mean())
                    value_losses.append(value_loss_all)  # already mean
                    entropies.append(entropy_all.mean())
                    valid = len(states)
                    # hand 予測損失（存在時のみ）
                    try:
                        hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
                    except Exception:
                        hand_coef = 0.0
                    if hand_coef > 0.0 and (hand_logits_batch is not None):
                        # 修正: 正解は「相手の全手札」(hand_labels)。マスクは「未知カード=1, 観測済み=0」。
                        # 目的: 行動決定直前の状態で、まだ出していない相手手札を適切に学習する。
                        # ロジットは各サンプルの入力順(自分を除いた相手順)に整列している前提。
                        D = int(hand_logits_batch.shape[1])
                        K = max(1, D // 53)  # 相手人数 (N-1)
                        Bsel = hand_logits_batch.shape[0]
                        tgt = torch.zeros(Bsel, D, dtype=torch.float32, device=device)
                        mask = torch.zeros(Bsel, D, dtype=torch.float32, device=device)
                        for bi, st in enumerate(states):
                            try:
                                # 1) 正解ラベル: hand_labels をそのまま使用 (相手の全手札)
                                hl = st.get('hand_labels')
                                if hl is not None:
                                    # 長さ整合: 必要なら切り詰め/ゼロ埋め
                                    import torch as _t
                                    if isinstance(hl, (list, tuple)):
                                        v = _t.tensor(list(hl), dtype=_t.float32, device=device)
                                    else:
                                        # numpy等のケース
                                        try:
                                            import numpy as _np
                                            # Ensure we copy non-writable arrays to avoid PyTorch warnings
                                            arr = _np.asarray(hl, dtype=_np.float32)
                                            if not arr.flags.writeable:
                                                arr = arr.copy()
                                            v = _t.tensor(arr, dtype=_t.float32, device=device)
                                        except Exception:
                                            try:
                                                v = _t.tensor(list(hl), dtype=_t.float32, device=device)
                                            except Exception:
                                                v = _t.zeros(D, dtype=_t.float32, device=device)
                                    if v.numel() < D:
                                        pad = _t.zeros(D - v.numel(), dtype=_t.float32, device=device)
                                        v = _t.cat([v, pad], dim=0)
                                    elif v.numel() > D:
                                        v = v[:D]
                                    tgt[bi] = v
                                # 2) マスク: 未知カード(= 自手札/場/捨て札の和集合 以外)のみ 1
                                self_hand_idx = st.get('self_hand_indices') or []
                                field_idx = st.get('field_card_indices') or []
                                discard_union_idx = st.get('discard_union_indices') or []
                                seen_set = set(self_hand_idx) | set(field_idx) | set(discard_union_idx)
                                unknown_idx = [ci for ci in range(53) if ci not in seen_set]
                                if unknown_idx:
                                    # 各相手スロット(連続53)に展開
                                    pos = []
                                    base = 0
                                    for _k in range(K):
                                        pos.extend([base + ci for ci in unknown_idx])
                                        base += 53
                                    if pos:
                                        import torch as _t
                                        idx_t = _t.tensor([p for p in pos if 0 <= p < D], device=device, dtype=_t.long)
                                        mask[bi, idx_t] = 1.0
                                        # last_action が存在する場合、直近に相手が出したカードを正例としてマーク
                                        try:
                                            last_player = st.get('last_action_player')
                                            last_cards_idx = st.get('last_action_card_indices') or []
                                            if last_player is not None and last_player != pid and last_cards_idx:
                                                # opponent slot のオフセットを算出して対応位置を求める
                                                N = K + 1
                                                opponents_order = [i for i in range(N) if i != pid]
                                                opp_offset = {opp_id: idx * 53 for idx, opp_id in enumerate(opponents_order)}
                                                off = opp_offset.get(last_player)
                                                if off is not None:
                                                    pos_last = [off + ci for ci in last_cards_idx if 0 <= off + ci < D]
                                                    if pos_last:
                                                        idx_last = _t.tensor(pos_last, device=device, dtype=_t.long)
                                                        mask[bi, idx_last] = 1.0
                                                        tgt[bi, idx_last] = 1.0
                                        except Exception:
                                            pass
                            except Exception:
                                # サンプル単体の失敗は無視 (tgt=0, mask=0)
                                continue
                        # 損失計算: マスク有効領域のみBCE、Focal、動的pos_weight
                        active = mask.sum().detach()
                        if active.item() > 0:
                            try:
                                eps = 1e-6
                                pos_area = (tgt * mask).sum()
                                p = (pos_area / (active + eps)).clamp(min=eps, max=1.0)
                                pos_w = ((1.0 - p) / (p + eps)).clamp(1.0, 20.0)
                                import torch.nn.functional as F
                                base_loss = F.binary_cross_entropy_with_logits(hand_logits_batch, tgt, reduction='none')
                                # Focal項
                                try:
                                    gamma = float(self.config.get('hand_focal_gamma', 2.0))
                                except Exception:
                                    gamma = 2.0
                                if gamma and gamma > 0.0:
                                    p_sig = torch.sigmoid(hand_logits_batch)
                                    pt = p_sig * tgt + (1.0 - p_sig) * (1.0 - tgt)
                                    focal = (1.0 - pt).clamp(min=1e-4).pow(gamma)
                                else:
                                    focal = 1.0
                                w_class = 1.0 + (pos_w - 1.0) * tgt
                                final_loss = (base_loss * focal * w_class * mask).sum() / (active + eps)
                            except Exception:
                                import torch.nn as _nn
                                final_loss = _nn.BCEWithLogitsLoss(reduction='mean')(hand_logits_batch, tgt)
                            hand_losses.append(final_loss)
                            # collect per-sample raw tensors for hand metrics
                            try:
                                for bi in range(Bsel):
                                    collected_hand_logits.append(hand_logits_batch[bi].detach().cpu())
                                    collected_hand_tgts.append(tgt[bi].detach().cpu())
                                    collected_hand_masks.append(mask[bi].detach().cpu())
                            except Exception:
                                pass
                            try:
                                pos_rate_h = float((tgt[mask > 0.5].mean().detach().cpu().item())) if (mask > 0.5).any() else None
                            except Exception:
                                pos_rate_h = None
                            collected_hand_pos_rate.append(pos_rate_h)
                    # メトリクス用個別保存
                    for i in range(len(states)):
                        n = lengths[i]
                        collected_pi.append(pi_pad_t[i, :n].detach())
                        collected_model.append(probs[i, :n].detach())
                        v_prob = torch.sigmoid(v_logits[i])
                        collected_v_pred.append(v_prob.detach())
                        collected_v_t.append(v_targets_t[i].detach())
                    vectorized_ok = True
            except Exception as _vec_e:
                # 一度だけ警告してフォールバック
                if not hasattr(self, '_vec_warned'):
                    print(f"[WARN] vectorized train_step fallback: {_vec_e}")
                    self._vec_warned = True  # type: ignore[attr-defined]
                vectorized_ok = False

        if not vectorized_ok:
            # 従来 per-sample ループ (variable モデル含む)
            for sample in batch:
                legal_actions = sample.get("legal_actions")
                # 可変長モデル使用時のみ、id_v1 からの復元を試みる（固定ヘッドでは不要）
                if variable and legal_actions is None and sample.get('actions_format') == 'id_v1' and 'legal_ids' in sample:
                    try:
                        # ACTION_ID_LIST を使わない方針のため、固定ヘッドでは復元スキップ
                        # 互換: 可変長モデル時のみ別経路での復元を想定（現状未使用）
                        legal_actions = None
                    except Exception:
                        legal_actions = None
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
                v_target = _extract_value(sample)
                # 固定ヘッドでは legal_actions は不要。可変長のみ必須とする。
                if (variable and not legal_actions) or (not pi_target) or (v_target is None):
                    continue
                if self.config.get('use_full_features') and self.config.get('skip_zero_padded_full_samples', True):
                    try:
                        st = sample.get('state') or {}
                        if ('full_compact' not in st) and ('full_input' not in st):
                            continue
                    except Exception:
                        pass
                # 可変長: 合法手数、固定ヘッド: πの長さを使用
                n = (len(legal_actions) if variable else len(pi_target))
                if variable:
                    logits_raw, v_pred_raw = self.model.evaluate(sample['state'], legal_actions)
                else:
                    logits_raw, v_out_logits = self.model.forward(sample['state'])
                    if hasattr(v_out_logits, 'shape'):
                        # select logit corresponding to the sample's stored viewpoint
                        try:
                            st = sample.get('state') or {}
                            sample_pid = int(st.get('self_player_id')) if (isinstance(st, dict) and ('self_player_id' in st)) else int(sample.get('player_id', getattr(self, 'player_id', 0)))
                        except Exception:
                            sample_pid = int(getattr(self, 'player_id', 0))
                        if 0 <= sample_pid < v_out_logits.shape[0]:
                            v_pred_raw = v_out_logits[sample_pid]
                        else:
                            v_pred_raw = v_out_logits[0]
                    else:
                        v_pred_raw = v_out_logits
                # Determine model/device to ensure newly created tensors live on same device
                try:
                    import torch as _torch
                    _model_dev = getattr(self.model, 'device', None)
                    if _model_dev is None:
                        try:
                            _model_dev = next(self.model.parameters()).device
                        except Exception:
                            _model_dev = _torch.device('cpu')
                except Exception:
                    _model_dev = 'cpu'

                if hasattr(logits_raw, 'shape'):
                    logits_t = logits_raw
                    if logits_t.shape[0] < n:
                        pad = torch.zeros(n - logits_t.shape[0], device=logits_t.device)
                        logits_t = torch.cat([logits_t, pad], dim=0)
                    else:
                        logits_t = logits_t[:n]
                else:
                    logits_list = list(logits_raw)
                    if len(logits_list) < n:
                        logits_list += [0.0] * (n - len(logits_list))
                    logits_t = torch.tensor(logits_list[:n], dtype=torch.float32, device=_model_dev)
                log_probs = logits_t.log_softmax(dim=0)
                probs = log_probs.exp()
                pi_t = torch.tensor(pi_target, dtype=torch.float32, device=log_probs.device)
                if pi_t.shape[0] != log_probs.shape[0]:
                    m = min(pi_t.shape[0], log_probs.shape[0])
                    pi_t = pi_t[:m]
                    log_probs = log_probs[:m]
                    probs = probs[:m]
                policy_loss = -(pi_t * log_probs).sum()
                if isinstance(v_pred_raw, float):
                    # ensure scalar logits are created on same device as model
                    try:
                        v_logit = torch.tensor(v_pred_raw, dtype=torch.float32, device=_model_dev)
                    except Exception:
                        v_logit = torch.tensor(v_pred_raw, dtype=torch.float32)
                else:
                    v_logit = v_pred_raw.float()
                v_t = torch.tensor(float(v_target), dtype=torch.float32, device=v_logit.device)
                if 'bce_logits_loss_fn' not in self.__dict__:
                    import torch.nn as _nn
                    try:
                        pw = float(self.pos_weight)
                    except Exception:
                        pw = 1.0
                    pos_w_tensor = None
                    if pw != 1.0:
                        import torch as _t
                        pos_w_tensor = _t.tensor([pw], dtype=_t.float32, device=v_logit.device)
                    self.bce_logits_loss_fn = _nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor) if pos_w_tensor is not None else _nn.BCEWithLogitsLoss()
                # apply importance-sampling weight (if prioritized sampling was used)
                try:
                    uid = sample.get('uid')
                    iw = float(uid2weight.get(int(uid), 1.0)) if uid2weight is not None else 1.0
                except Exception:
                    iw = 1.0
                # Per-sample terminal boost (fallback for non-vectorized path)
                try:
                    term_boost = float(self.config.get('terminal_sample_boost', 1.0) or 1.0)
                except Exception:
                    term_boost = 1.0
                if term_boost != 1.0:
                    try:
                        st = sample.get('state') or {}
                        is_term = False
                        if isinstance(st, dict):
                            is_term = bool(st.get('is_terminal', False)) or (int(st.get('hand_size', -1)) == 0)
                        if is_term:
                            iw = float(iw) * float(term_boost)
                    except Exception:
                        pass
                value_loss = self.bce_logits_loss_fn(v_logit.unsqueeze(0), v_t.unsqueeze(0)) * float(iw)
                v_prob = torch.sigmoid(v_logit)
                entropy = -(probs * log_probs).sum()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                # hand ヘッド（単体; 可能なら）簡素化: 例外は一括捕捉
                hand_coef = 0.0
                try:
                    hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
                except Exception:
                    hand_coef = 0.0
                if hand_coef > 0.0 and hasattr(self.model, 'forward_with_belief'):
                    # Fallback strict masking per-sample
                    try:
                        hand_logits = None
                        _, _, hand_logits = self.model.forward_with_belief(sample['state'])
                    except Exception:
                        hand_logits = None
                    if hand_logits is not None:
                        try:
                            N = int(getattr(self.model, 'num_players', int(self.config.get('num_players', 4))))
                        except Exception:
                            N = int(self.config.get('num_players', 4) or 4)
                        D = 53 * (N - 1)
                        tgt = torch.zeros(D, dtype=torch.float32, device=hand_logits.device)
                        mask_sample = torch.zeros(D, dtype=torch.float32, device=hand_logits.device)
                        st = sample.get('state') or {}
                        pid_self = getattr(self, 'player_id', 0)
                        opponents_order = [i for i in range(N) if i != pid_self]
                        opp_offset = {opp_id: idx * 53 for idx, opp_id in enumerate(opponents_order)}
                        # Build mask marking UNKNOWN cards = 1 (consistent with vectorized path).
                        # Prefer explicit metadata; if missing, attempt to reconstruct self indices
                        # from `full_input` (first 53 bits) as a fallback.
                        self_hand_idx = st.get('self_hand_indices') or []
                        if not self_hand_idx:
                            try:
                                fi = st.get('full_input')
                                if isinstance(fi, (list, tuple)) and len(fi) >= 53:
                                    self_hand_idx = [i for i, v in enumerate(fi[:53]) if float(v) > 0.5]
                            except Exception:
                                self_hand_idx = []
                        field_idx = st.get('field_card_indices') or []
                        discard_union_idx = st.get('discard_union_indices') or []
                        last_player = st.get('last_action_player')
                        last_cards_idx = st.get('last_action_card_indices') or []
                        # seen = observed cards (self + field + discards). unknown = complement
                        seen_set = set(self_hand_idx) | set(field_idx) | set(discard_union_idx)
                        unknown_idx = [ci for ci in range(53) if ci not in seen_set]
                        # For each opponent slot, mark unknown card positions as active (mask=1)
                        for opp_id in opponents_order:
                            off = opp_offset.get(opp_id)
                            if off is None:
                                continue
                            for ci in unknown_idx:
                                pos = off + ci
                                if 0 <= pos < D:
                                    mask_sample[pos] = 1.0
                                    tgt[pos] = 0.0
                        # If the last action by an opponent played specific cards, include
                        # those positions in the mask and mark them positive (tgt=1).
                        if last_player is not None and last_player != pid_self and last_cards_idx:
                            off = opp_offset.get(last_player)
                            if off is not None:
                                for ci in last_cards_idx:
                                    pos = off + ci
                                    if 0 <= pos < D:
                                        mask_sample[pos] = 1.0
                                        tgt[pos] = 1.0
                        active = mask_sample.sum().detach()
                        if active.item() > 0:
                            try:
                                import torch.nn.functional as F
                                eps = 1e-6
                                pos_area = (tgt * mask_sample).sum()
                                p = (pos_area / (active + eps)).clamp(min=eps, max=1.0)
                                pos_w = ((1.0 - p) / (p + eps)).clamp(1.0, 20.0)
                                base_loss = F.binary_cross_entropy_with_logits(hand_logits.float(), tgt, reduction='none')
                                # Focal term (1 - pt)^gamma applied only to hand prediction
                                try:
                                    gamma = float(self.config.get('hand_focal_gamma', 2.0))
                                except Exception:
                                    gamma = 2.0
                                if gamma and gamma > 0.0:
                                    p_hat = torch.sigmoid(hand_logits.float())
                                    pt = p_hat * tgt + (1.0 - p_hat) * (1.0 - tgt)
                                    focal = (1.0 - pt).clamp(min=1e-4).pow(gamma)
                                else:
                                    focal = 1.0
                                w_class = 1.0 + (pos_w - 1.0) * tgt
                                hloss = (base_loss * focal * w_class * mask_sample).sum() / (active + eps)
                            except Exception:
                                import torch.nn as _nn
                                hloss = _nn.BCEWithLogitsLoss(reduction='mean')(hand_logits.float(), tgt)
                            hand_losses.append(hloss)
                            # collect per-sample raw tensors for hand metrics (fallback per-sample path)
                            try:
                                if hand_logits is not None:
                                    collected_hand_logits.append(hand_logits.detach().cpu())
                                    collected_hand_tgts.append(tgt.detach().cpu())
                                    collected_hand_masks.append(mask_sample.detach().cpu())
                            except Exception:
                                pass
                            try:
                                collected_hand_pos_rate.append(float((tgt[mask_sample > 0.5].mean().detach().cpu().item())) if (mask_sample > 0.5).any() else None)
                            except Exception:
                                pass

                valid += 1
                collected_pi.append(pi_t.detach())
                collected_model.append(probs.detach())
                collected_v_pred.append(v_prob.detach())
                collected_v_t.append(v_t.detach())
                try:
                    uid = sample.get('uid')
                    if uid is not None:
                        used_uids.append(int(uid))
                except Exception:
                    pass

        if valid == 0:
            return {"loss": None, "reason": "no_valid_samples"}

        policy_loss_mean = torch.stack(policy_losses).mean()
        value_loss_mean = torch.stack(value_losses).mean()
        hand_loss_mean = (torch.stack(hand_losses).mean() if hand_losses else None)
        entropy_mean = torch.stack(entropies).mean()
        total_loss = (self.policy_loss_coef * policy_loss_mean +
                      self.value_loss_coef * value_loss_mean -
                      self.entropy_coef * entropy_mean)
        try:
            hand_coef = float(self.config.get('hand_pred_loss_coef', 0.0) or 0.0)
        except Exception:
            hand_coef = 0.0
        if hand_coef > 0.0 and hand_loss_mean is not None:
            total_loss = total_loss + hand_coef * hand_loss_mean
        # --- Mixed precision training (GradScaler) ---
        try:
            import torch as _t
        except Exception:
            _t = None

        # decide whether to use AMP: require CUDA device on the model
        _use_amp = False
        try:
            _dev = getattr(self.model, 'device', None)
            _use_amp = bool(getattr(_dev, 'type', None) == 'cuda') and (_t is not None and _t.cuda.is_available())
        except Exception:
            _use_amp = False

        # delayed GradScaler init (store on agent)
        if _use_amp:
            try:
                if not hasattr(self, '_amp_scaler') or self._amp_scaler is None:
                    # torch.cuda.amp.GradScaler は非推奨: 新 API torch.amp.GradScaler を使用
                    self._amp_scaler = _t.amp.GradScaler('cuda')
            except Exception:
                # fallback to no AMP
                try:
                    self._amp_scaler = None
                except Exception:
                    pass

        self._optimizer.zero_grad()
        did_update = False
        # placeholders for gradient/weight norms
        grad_norm = None
        # value_head specific grad norm (computed when grads exist)
        value_head_grad_norm = None
        weight_norm = None
        if _use_amp and getattr(self, '_amp_scaler', None) is not None:
            try:
                # scale the loss, backward, then unscale for clipping
                self._amp_scaler.scale(total_loss).backward()
                try:
                    # unscale before clip
                    self._amp_scaler.unscale_(self._optimizer)
                except Exception:
                    pass
                # compute gradient norm (unscaled grads should be available after unscale_)
                try:
                    import math as _m
                    total_sq = 0.0
                    v_total_sq = 0.0
                    for p in self.model.parameters():
                        if getattr(p, 'grad', None) is not None:
                            try:
                                gnorm = float(p.grad.detach().cpu().norm().item())
                                total_sq += (gnorm * gnorm)
                            except Exception:
                                continue
                    # value_head grad norm
                    try:
                        for p in getattr(self.model, 'value_head', []).parameters():
                            if getattr(p, 'grad', None) is not None:
                                try:
                                    gnorm = float(p.grad.detach().cpu().norm().item())
                                    v_total_sq += (gnorm * gnorm)
                                except Exception:
                                    continue
                        value_head_grad_norm = float(_m.sqrt(v_total_sq)) if v_total_sq >= 0.0 else None
                    except Exception:
                        value_head_grad_norm = None
                    grad_norm = float(_m.sqrt(total_sq)) if total_sq >= 0.0 else None
                except Exception:
                    grad_norm = None
                    value_head_grad_norm = None
                if self.grad_clip and self.grad_clip > 0:
                    try:
                        _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    except Exception:
                        pass
                try:
                    self._amp_scaler.step(self._optimizer)
                    self._amp_scaler.update()
                    did_update = True
                except Exception:
                    # If step failed (e.g. inf/overflow), update scaler and skip this step
                    try:
                        self._amp_scaler.update()
                    except Exception:
                        pass
                    did_update = False
            except Exception:
                # fallback to FP32 path on unexpected errors
                try:
                    total_loss.backward()
                    # compute grad norm before clipping/step
                    try:
                            import math as _m
                            total_sq = 0.0
                            v_total_sq = 0.0
                            for p in self.model.parameters():
                                if getattr(p, 'grad', None) is not None:
                                    try:
                                        gnorm = float(p.grad.detach().cpu().norm().item())
                                        total_sq += (gnorm * gnorm)
                                    except Exception:
                                        continue
                            try:
                                for p in getattr(self.model, 'value_head', []).parameters():
                                    if getattr(p, 'grad', None) is not None:
                                        try:
                                            gnorm = float(p.grad.detach().cpu().norm().item())
                                            v_total_sq += (gnorm * gnorm)
                                        except Exception:
                                            continue
                                value_head_grad_norm = float(_m.sqrt(v_total_sq)) if v_total_sq >= 0.0 else None
                            except Exception:
                                value_head_grad_norm = None
                            grad_norm = float(_m.sqrt(total_sq)) if total_sq >= 0.0 else None
                    except Exception:
                        grad_norm = None
                    if self.grad_clip and self.grad_clip > 0:
                        try:
                            _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        except Exception:
                            pass
                    self._optimizer.step()
                    did_update = True
                except Exception:
                    did_update = False
        else:
            # FP32 update path
            try:
                total_loss.backward()
                # compute grad norm (FP32 path)
                try:
                    import math as _m
                    total_sq = 0.0
                    v_total_sq = 0.0
                    for p in self.model.parameters():
                        if getattr(p, 'grad', None) is not None:
                            try:
                                gnorm = float(p.grad.detach().cpu().norm().item())
                                total_sq += (gnorm * gnorm)
                            except Exception:
                                continue
                    try:
                        for p in getattr(self.model, 'value_head', []).parameters():
                            if getattr(p, 'grad', None) is not None:
                                try:
                                    gnorm = float(p.grad.detach().cpu().norm().item())
                                    v_total_sq += (gnorm * gnorm)
                                except Exception:
                                    continue
                        value_head_grad_norm = float(_m.sqrt(v_total_sq)) if v_total_sq >= 0.0 else None
                    except Exception:
                        value_head_grad_norm = None
                    grad_norm = float(_m.sqrt(total_sq)) if total_sq >= 0.0 else None
                except Exception:
                    grad_norm = None
                    value_head_grad_norm = None
                if self.grad_clip and self.grad_clip > 0:
                    try:
                        _t.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    except Exception:
                        pass
                self._optimizer.step()
                did_update = True
            except Exception:
                did_update = False
        # Scheduler step (warmup+cosine) — optimizer 実ステップがあった時のみ実行
        try:
            if did_update:
                if self._scheduler is None:
                    self.ensure_scheduler()
                if self._scheduler is not None:
                    # 一部の実装では optimizer._step_count に基づき順序チェックが行われ、
                    # さらに PyTorch は optimizer.step() より前の scheduler.step() を警告する。
                    # そのため、確実に optimizer の内部ステップが進んだと判定できる場合のみ scheduler を進める。
                    step_cnt = getattr(self._optimizer, "_step_count", None)
                    has_stepped = False
                    try:
                        if isinstance(step_cnt, int):
                            has_stepped = step_cnt > 0
                        else:
                            # Fallback: 任意の param state から 'step' を参照 (Adam系は per-param で保持)
                            opt_state = getattr(self._optimizer, "state", {})
                            if isinstance(opt_state, dict) and opt_state:
                                for st in opt_state.values():
                                    s = st.get("step") if isinstance(st, dict) else None
                                    if isinstance(s, int) and s > 0:
                                        has_stepped = True
                                        break
                    except Exception:
                        has_stepped = True  # 最悪でも進める
                    if has_stepped:
                        # 自前カウンタも同期してから scheduler を進める
                        self._update_step += 1
                        self._scheduler.step()
        except Exception:
            pass

        # compute weight norm after update (reflects current parameter magnitudes)
        try:
            import math as _m
            total_sq_w = 0.0
            for p in self.model.parameters():
                try:
                    wnorm = float(p.detach().cpu().norm().item())
                    total_sq_w += (wnorm * wnorm)
                except Exception:
                    continue
            weight_norm = float(_m.sqrt(total_sq_w)) if total_sq_w >= 0.0 else None
        except Exception:
            weight_norm = None

        # 追加メトリクス計算（共通ユーティリティに委譲, pos_rate はローカルで算出）
        policy_kl = None
        policy_top1 = None
        value_acc = None
        value_brier = None
        pos_rate = None
        if collected_pi:
            try:
                from utils.metrics import compute_policy_value_metrics
                extra = compute_policy_value_metrics(collected_pi, collected_model, collected_v_pred, collected_v_t)
                policy_kl = extra.get('policy_kl')
                policy_top1 = extra.get('policy_top1_match')
                value_acc = extra.get('value_acc')
                value_brier = extra.get('value_brier')
                # pos_rate は学習時のみ必要なためここで算出
                v_label_list = [float(v_lab.item()) for v_lab in collected_v_t]
                if v_label_list:
                    pos_rate = float(sum(1.0 if v > 0.5 else 0.0 for v in v_label_list) / len(v_label_list))
            except Exception as e:
                if not hasattr(self, '_metric_warned'):
                    print(f"[WARN] metric calc failed: {e}")
                    self._metric_warned = True

        metrics = {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss_mean.item()),
            "value_loss": float(value_loss_mean.item()),
            "entropy": float(entropy_mean.item()),
            "samples": valid,
            "policy_kl": policy_kl,
            "policy_top1_match": policy_top1,
            "value_acc": value_acc,
            "value_brier": value_brier,
            "pos_rate": pos_rate,
            "cum_pos_rate": (self.total_positive / self.total_value_samples) if self.total_value_samples > 0 else None,
            "tt_hit_rate": (self.tt_hits / max(1, (self.tt_hits + self.tt_misses))) if (self.tt_hits + self.tt_misses) > 0 else None,
        }
        # attach gradient/weight norms if available
        try:
            metrics['grad_norm'] = grad_norm
            metrics['weight_norm'] = weight_norm
            metrics['value_head_grad_norm'] = value_head_grad_norm
        except Exception:
            pass
        # compute hand-pred metrics (top-K accuracy, Brier) if we collected predictions
        try:
            if collected_hand_logits:
                import numpy as _np
                import math as _m
                top1_cnt = 0
                top3_cnt = 0
                brier_sum = 0.0
                valid_s = 0
                for logits_t, tgt_t, mask_t in zip(collected_hand_logits, collected_hand_tgts, collected_hand_masks):
                    try:
                        lg = _np.asarray(logits_t)
                        tg = _np.asarray(tgt_t)
                        mk = _np.asarray(mask_t)
                        sel = _np.where(mk > 0.5)[0]
                        if sel.size == 0:
                            continue
                        probs = 1.0 / (1.0 + _np.exp(-lg[sel]))
                        # order descending
                        order = _np.argsort(-probs)
                        k1 = 1
                        k3 = min(3, sel.size)
                        topk_idx = sel[order[:k3]]
                        pos_idx = _np.where(tg > 0.5)[0]
                        # check intersection
                        if pos_idx.size > 0 and _np.intersect1d(topk_idx, pos_idx).size > 0:
                            top3_cnt += 1
                        # top1
                        top1_idx = sel[order[0]]
                        if pos_idx.size > 0 and (top1_idx in pos_idx):
                            top1_cnt += 1
                        # brier over masked positions
                        pred_probs = 1.0 / (1.0 + _np.exp(-lg[sel]))
                        tgt_vals = tg[sel]
                        brier = float(((pred_probs - tgt_vals) ** 2).mean())
                        brier_sum += brier
                        valid_s += 1
                    except Exception:
                        continue
                    # accumulate global TP / actual positives for recall
                    try:
                        # predicted positives at threshold 0.5
                        pred_pos = _np.where(pred_probs > 0.5)[0]
                        # map sel indices to global indices: sel[pred_pos]
                        pred_global = sel[pred_pos]
                        # actual positives indices (global)
                        actual_pos = pos_idx
                        # intersection count
                        tp = _np.intersect1d(pred_global, actual_pos).size
                        total_actual = actual_pos.size
                    except Exception:
                        tp = 0
                        total_actual = 0
                    try:
                        metrics.setdefault('_hand_recall_accum_tp', 0)
                        metrics.setdefault('_hand_recall_accum_actual', 0)
                        metrics['_hand_recall_accum_tp'] += int(tp)
                        metrics['_hand_recall_accum_actual'] += int(total_actual)
                    except Exception:
                        pass
                if valid_s > 0:
                    metrics['hand_top1'] = float(top1_cnt / valid_s)
                    metrics['hand_top3'] = float(top3_cnt / valid_s)
                    metrics['hand_brier'] = float(brier_sum / valid_s)
                else:
                    metrics['hand_top1'] = None
                    metrics['hand_top3'] = None
                    metrics['hand_brier'] = None
            # compute overall recall if accumulated
            try:
                tp_tot = int(metrics.pop('_hand_recall_accum_tp', 0))
                act_tot = int(metrics.pop('_hand_recall_accum_actual', 0))
                metrics['hand_recall'] = float(tp_tot / act_tot) if act_tot > 0 else None
            except Exception:
                metrics['hand_recall'] = None
        except Exception:
            pass
        # 追加: hand ヘッド関連メトリクス
        try:
            if hand_coef > 0.0:
                metrics["hand_pred_loss"] = (float(hand_loss_mean.item()) if hand_loss_mean is not None else None)
                # ラベルの正例率（監視用）
                if collected_hand_pos_rate:
                    # None を除外して平均
                    vals = [v for v in collected_hand_pos_rate if v is not None]
                    metrics["hand_label_pos_rate"] = (sum(vals) / len(vals)) if vals else None
                else:
                    metrics["hand_label_pos_rate"] = None
        except Exception:
            pass
        # pos_rate 警告
        try:
            warn_th = float(self.config.get('pos_rate_warn_threshold', 0.02))
            if pos_rate is not None and pos_rate < warn_th:
                print(f"[WARN] value positive sample rate low ({pos_rate:.2%}) < {warn_th:.2%}")
        except Exception:
            pass

        # --- Update priorities for prioritized replay (conservative: abs(pred-target) + eps) ---
        try:
            if bool(self.config.get('prioritized_replay', False)) and hasattr(self.replay_buffer, 'update_priorities') and used_uids:
                uid_to_p = {}
                eps = float(self.config.get('prioritized_replay_eps', 1e-6) or 1e-6)
                for i, uid in enumerate(used_uids):
                    try:
                        pred = collected_v_pred[i]
                        targ = collected_v_t[i]
                        # convert tensors to float safely
                        try:
                            pred_f = float(pred.detach().cpu().item()) if hasattr(pred, 'detach') else float(pred)
                        except Exception:
                            try:
                                pred_f = float(pred.item())
                            except Exception:
                                pred_f = float(pred)
                        try:
                            targ_f = float(targ.detach().cpu().item()) if hasattr(targ, 'detach') else float(targ)
                        except Exception:
                            try:
                                targ_f = float(targ.item())
                            except Exception:
                                targ_f = float(targ)
                        p = abs(pred_f - targ_f) + eps
                        uid_to_p[int(uid)] = float(p)
                    except Exception:
                        continue
                if uid_to_p:
                    try:
                        self.replay_buffer.update_priorities(uid_to_p)
                    except Exception:
                        pass
        except Exception:
            pass
        if self.logger:
            self.logger.log_train(metrics)
            self._logged_inside = True
            # メモリスナップショット (学習プレイヤーのみ想定: config.learning_player_id)
            try:
                lp = int(self.config.get('learning_player_id', 0))
                if self.player_id == lp and hasattr(self.logger, 'log_memory_snapshot'):
                    # 共有リプレイ形式かローカルかでサンプル数を取得
                    sample_count = None
                    try:
                        if self.replay_buffer is None:
                            sample_count = 0  # worker_zero_buffer モードではサンプル数は0
                        elif self._use_shared and hasattr(self.replay_buffer, '__len__'):
                            sample_count = len(self.replay_buffer)
                        elif isinstance(self.replay_buffer, list):
                            sample_count = len(self.replay_buffer)
                    except Exception:
                        pass
                    self.logger.log_memory_snapshot(sample_count=sample_count, force=False)
            except Exception:
                pass
            # events.log へ周期的に TRAIN 指標を一行テキストとして追記 (軽量モニタ用)
            try:
                freq = int(self.config.get('events_log_train_every', 0) or 0)
                if freq > 0:
                    step = getattr(self.logger, 'update_step', None)
                    # log_train 呼び出し後なので update_step は 1 インクリメント済み
                    if step and (step % freq == 0):
                        parts = [
                            f"loss={metrics.get('loss'):.4f}" if metrics.get('loss') is not None else None,
                            f"pl={metrics.get('policy_loss'):.4f}" if metrics.get('policy_loss') is not None else None,
                            f"vl={metrics.get('value_loss'):.4f}" if metrics.get('value_loss') is not None else None,
                            f"hand={metrics.get('hand_pred_loss'):.4f}" if metrics.get('hand_pred_loss') is not None else None,
                            f"ent={metrics.get('entropy'):.3f}" if metrics.get('entropy') is not None else None,
                            f"kl={metrics.get('policy_kl'):.4f}" if metrics.get('policy_kl') is not None else None,
                            f"top1={metrics.get('policy_top1_match'):.3f}" if metrics.get('policy_top1_match') is not None else None,
                            f"v_acc={metrics.get('value_acc'):.3f}" if metrics.get('value_acc') is not None else None,
                            f"v_brier={metrics.get('value_brier'):.4f}" if metrics.get('value_brier') is not None else None,
                            f"pos={metrics.get('pos_rate'):.3f}" if metrics.get('pos_rate') is not None else None,
                            f"hand_pos={metrics.get('hand_label_pos_rate'):.3f}" if metrics.get('hand_label_pos_rate') is not None else None,
                            f"cum_pos={metrics.get('cum_pos_rate'):.3f}" if metrics.get('cum_pos_rate') is not None else None,
                            f"samples={metrics.get('samples')}" if metrics.get('samples') is not None else None,
                        ]
                        line = " ".join(p for p in parts if p is not None)
                        self.logger.log_text(f"[TRAIN] step={step} {line}", also_print=False)
            except Exception:
                pass
        return metrics


__all__ = [
    'VALUE_U8_NONE',
    '_encode_value_u8',
    '_decode_value_u8',
    '_extract_value',
    '_has_value_label',
    'ReplaySnapshotDataset',
    'RawSampleDataset',
    'PreselectedSampler',
    'collate_prepared',
    'collate_prepare_batch',
    'TrainStepMixin',
    'InferenceClient',
    'LocalInferenceClient',
    'RemoteInferenceClient',
]
