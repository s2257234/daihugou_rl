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
        joblib.dump({
            "maxlen": self.maxlen,
            "next_id": self._next_id,
            "data": list(self._data),  # deque -> list
        }, path, compress=3)

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
