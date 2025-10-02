from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import joblib

class ReplayBuffer:
    """共有リプレイバッファ (全エージェント共通)。

    特徴:
      - deque(maxlen) による O(1) 古いサンプル破棄
      - サンプルには uid / player_id を付与
      - owner_pid フィルタ付きサンプリング
    """
    def __init__(self, maxlen: int, path: str | None = None):
        self._data: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._next_id: int = 0
        self.maxlen = maxlen
        self.default_path = path

    def append(self, sample: Dict[str, Any]) -> int:
        # 非 dict / 必須キー不足は無視 (異常混入防止)
        if not isinstance(sample, dict):
            return -1
        # 追い出し対象を先に取得 (maxlen到達時 deque は自動で左端を捨てるが、Pythonでは直接検出できないため
        # 事前に長さを見て手動popしフックする)
        evicted = None
        if len(self._data) == self.maxlen:
            evicted = self._data.popleft()
            if evicted is not None:
                try:
                    evicted["in_buffer"] = False
                except Exception:
                    pass
        sample["uid"] = self._next_id
        self._next_id += 1
        sample["in_buffer"] = True
        self._data.append(sample)
        return sample["uid"]

    def __len__(self) -> int:
        return len(self._data)

    def iter_all(self, owner_pid: Optional[int] = None):
        if owner_pid is None:
            for s in self._data:
                yield s
        else:
            for s in self._data:
                if s.get("player_id") == owner_pid:
                    yield s

    def sample(self, n: int, owner_pid: Optional[int] = None) -> List[Dict[str, Any]]:
        import random
        pool = list(self.iter_all(owner_pid=owner_pid))
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)
        return random.sample(pool, n)

    # ---------------- Persistence ----------------
    def save(self, path: str):
        import sys, traceback
        # 破損/再帰エラー防止のため、安全にシリアライズ可能な最小サブセットへ整形
        allow_keys = {
            "player_id","state","pi","value","model_version","feature_version","legal_actions","value_pred","uid",
            # 量子化 / 圧縮フィールド
            "pi_q","pi_format","legal_ids","actions_format","value_u8","value_pred_u8"
        }

        def _make_state_safe(st):
            if not isinstance(st, dict):
                return None
            st_safe = {}
            if "full_input" in st:
                fi = st["full_input"]
                orig_len = None
                try:
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
                    if getattr(arr16, 'shape', [0])[0] > 5000:
                        arr16 = arr16[:5000]
                    st_safe["full_input"] = arr16
                    if orig_len is not None:
                        st_safe["full_input_len"] = int(orig_len)
                    st_safe["full_input_dtype"] = "float16"
            for mk in ("hand_size","field_size","turn","full_input_dim"):
                if mk in st:
                    st_safe[mk] = st[mk]
            return st_safe

        safe_list = []
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
                # 問題サンプル特定用ログ
                print(f"[WARN] recursion while sanitizing sample idx={idx}: {e}")
                print("[WARN] sample keys=", list(s.keys()))
                continue
            except Exception:
                continue

        payload = {"maxlen": self.maxlen, "next_id": self._next_id, "data": safe_list}

        # 一時的に再帰制限を引き上げ (深い入れ子による失敗緩和)
        orig_limit = sys.getrecursionlimit()
        if orig_limit < 5000:
            try:
                sys.setrecursionlimit(5000)
            except Exception:
                pass
        try:
            joblib.dump(payload, path, compress=3)
            return
        except RecursionError as e:
            # 詳細スタック出力 (最初の一回のみフル)
            if not hasattr(self, '_recursion_first'):  # type: ignore[attr-defined]
                print(f"[WARN] replay save recursion error (compress=3): {e}")
                tb = ''.join(traceback.format_exc()[-2000:])
                print(f"[WARN] traceback tail:\n{tb}")
                self._recursion_first = True  # type: ignore[attr-defined]
            # 圧縮無しで再挑戦
            try:
                joblib.dump(payload, path, compress=0)
                return
            except RecursionError:
                # バイナリサーチで問題サンプル特定 (最大 10 試行)
                def _can_dump(sub):
                    try:
                        joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": sub}, path + '.probe', compress=0)
                        return True
                    except RecursionError:
                        return False
                    except Exception:
                        return True  # 他エラーは無視
                lo, hi = 0, len(safe_list)
                bad_idx = None
                attempts = 0
                while lo < hi and attempts < 10:
                    mid = (lo + hi) // 2
                    if _can_dump(safe_list[:mid]):
                        lo = mid + 1
                    else:
                        hi = mid
                    attempts += 1
                if lo <= len(safe_list):
                    bad_idx = lo - 1
                if bad_idx is not None and 0 <= bad_idx < len(safe_list):
                    print(f"[WARN] suspect sample causing recursion idx={bad_idx} (will exclude & fallback)")
                    try:
                        del safe_list[bad_idx]
                    except Exception:
                        pass
                # 最小フォールバック: 末尾 1000 サンプルのみ保存
                fallback = safe_list[-1000:] if len(safe_list) > 1000 else safe_list
                try:
                    joblib.dump({"maxlen": self.maxlen, "next_id": self._next_id, "data": fallback}, path, compress=0)
                    print(f"[WARN] fallback replay saved with {len(fallback)}/{len(safe_list)} samples")
                    return
                except Exception as ee2:
                    print(f"[ERROR] replay save ultimate fallback failed: {ee2}")
        except Exception as e:
            if not hasattr(self, '_warned_save'):  # type: ignore[attr-defined]
                print(f"[WARN] replay save failed once: {e}")
                self._warned_save = True  # type: ignore[attr-defined]
        finally:
            # 復元
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
        # 後方互換: 旧形式 (list of samples) の場合
        if data_list is None and isinstance(obj, list):
            data_list = obj
        if not data_list:
            return rb
        for s in data_list:
            # uid がない旧サンプルには再割り当て
            if "uid" not in s:
                s["uid"] = rb._next_id
            rb._next_id = max(rb._next_id, s["uid"] + 1)
            rb._data.append(s)
        return rb

__all__ = ["ReplayBuffer"]
