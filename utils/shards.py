from __future__ import annotations
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import joblib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_write_joblib(obj: Any, path: str, compress: int = 3) -> None:
    tmp = path + ".part"
    joblib.dump(obj, tmp, compress=compress)
    os.replace(tmp, path)


def list_shards(dir_path: str, ext: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(ext) and '.tmp.' not in f]
    # sort by mtime then name for stability
    files.sort(key=lambda p: (os.path.getmtime(p), p))
    return files


def read_shard(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def move_file(src: str, dst_dir: Optional[str]) -> None:
    try:
        if dst_dir:
            ensure_dir(dst_dir)
            base = os.path.basename(src)
            dst = os.path.join(dst_dir, base)
            if os.path.exists(dst):
                # de-duplicate
                name, ext = os.path.splitext(base)
                dst = os.path.join(dst_dir, f"{name}.ingested{ext}")
            os.replace(src, dst)
        else:
            os.remove(src)
    except Exception:
        # last resort: try remove
        try:
            os.remove(src)
        except Exception:
            pass


def make_shard_path(dir_path: str, ext: str, prefix: str = "sp") -> str:
    ts = int(time.time() * 1000)
    pid = os.getpid()
    rnd = uuid.uuid4().hex[:6]
    name = f"{prefix}-{ts}-{pid}-{rnd}{ext}"
    return os.path.join(dir_path, name)


def write_samples_as_shard(samples: List[Dict[str, Any]], dir_path: str, ext: str, *, episodes: Optional[int] = None) -> Optional[str]:
    """サンプルリストを 1 つのシャード(joblib)に書き出す。

    episodes: このシャードに含まれる完了エピソード数 (エピソード境界がサンプル列と一致しない場合は概算可)。
              None の場合はメタデータに含めない (後方互換)。
    """
    if not samples:
        return None
    ensure_dir(dir_path)
    path = make_shard_path(dir_path, ext)
    payload: Dict[str, Any] = {"version": 1, "count": len(samples), "samples": samples}
    if episodes is not None:
        payload["episodes"] = int(episodes)
    atomic_write_joblib(payload, path, compress=3)
    return path
