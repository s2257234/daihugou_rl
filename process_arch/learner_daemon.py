from __future__ import annotations
"""Learner daemon process: ingest sample shards and perform training updates.

Usage:
  python -m process_arch.learner_daemon

This process polls a shard directory and ingests samples into the trainer's
shared replay buffer, then runs periodic training bursts.
"""
import os
import sys
import time
import psutil
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from trainer.trainer import Trainer
from utils.shards import ensure_dir, list_shards, read_shard, move_file
from agents.replay_buffer import ReplayBuffer

# グローバル: 累積エピソード数 (シャードメタデータ episodes より加算)
_TOTAL_EPISODES_INGESTED = 0

# Learner 用 RSS 監視状態
class _RssRestartState:
    def __init__(self):
        self.last_check = 0.0
        self.consecutive_high = 0
        self.last_restart_time = 0.0

_rss_state = _RssRestartState()


def ingest_shards(tr: Trainer, cfg: Dict[str, Any], rb: ReplayBuffer) -> int:
    """Shard を読み込み一時的にメモリへ積む (file-only モード時は直後に保存して purge)。
    シャード内 episodes メタがあればグローバル加算。
    """
    dir_path = cfg.get("sample_shard_dir", "sample_shards")
    ingested_dir = cfg.get("sample_shard_ingested_dir", None)
    ext = cfg.get("sample_shard_ext", ".shard.joblib")
    files = list_shards(dir_path, ext)
    if not files:
        return 0
    # 一度に取り込む最大シャード数 (負荷平準化 / メモリ瞬間使用量抑制)。0/未設定で無制限。
    max_per_poll = int(cfg.get("learner_max_shards_per_poll", 0) or 0)
    truncated = False
    if max_per_poll > 0 and len(files) > max_per_poll:
        files = files[:max_per_poll]
        truncated = True
    ingested = 0
    for p in files:
        try:
            payload = read_shard(p)
            samples = payload.get("samples") or []
            eps_meta = payload.get("episodes")
            for s in samples:
                if isinstance(s, dict):
                    try:
                        rb.append(s)
                        ingested += 1
                    except Exception:
                        pass
            if isinstance(eps_meta, int) and eps_meta > 0:
                global _TOTAL_EPISODES_INGESTED
                _TOTAL_EPISODES_INGESTED += eps_meta
        except Exception as e:
            _log_event(cfg, f"shard read failed path='{p}' err={e}")
        finally:
            try:
                move_file(p, ingested_dir)
            except Exception:
                pass
    if truncated:
        _log_event(cfg, f"ingest shard cap applied max_per_poll={max_per_poll} remaining_queued={len(list_shards(dir_path, ext))}")
    return ingested


def _merge_into_snapshot(cfg: Dict[str, Any], rb: ReplayBuffer) -> int:
    """メモリ rb の内容をディスクのスナップショットへマージして保存し、保存件数を返す。
    既存ファイルがあればロードして追記、なければ新規作成。保存後は rb は変更しない (呼び出し側で clear)。"""
    snapshot_path = cfg.get("replay_path", "replay_buffer.joblib")
    try:
        if os.path.exists(snapshot_path):
            base = ReplayBuffer.load(snapshot_path)
        else:
            base = ReplayBuffer(maxlen=int(cfg.get("learner_buffer_size", cfg.get("buffer_size", 250000))))
        # append all from rb
        cnt_before = len(base)
        for s in rb.iter_all(owner_pid=None):
            try:
                base.append(s)
            except Exception:
                pass
        base.save(snapshot_path, purge=False)
        return len(base) - cnt_before
    except Exception as e:
        _log_event(cfg, f"merge snapshot failed: {e}")
        return 0


def _log_event(cfg: Dict[str, Any], msg: str) -> None:
    try:
        import datetime
        log_dir = cfg.get('log_dir', 'logs')
        ensure_dir(log_dir)
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] [LEARNER] {msg}\n")
    except Exception:
        pass


def _save_candidate(tr: Trainer, cfg: Dict[str, Any]) -> str | None:
    model = getattr(tr, 'model', None)
    if model is None:
        return None
    cand_dir = cfg.get('candidate_model_dir', os.path.join(cfg.get('checkpoint_dir','checkpoints'), 'candidates'))
    ensure_dir(cand_dir)
    ts = int(time.time())
    path = os.path.join(cand_dir, f"policy_value_cand_{ts}.pt")
    try:
        # force_sync=True で非同期I/O経由せず原子保存を保証
        model.save(path, force_sync=True)
        return path
    except Exception as e:
        _log_event(cfg, f"candidate save failed: {e}")
        return None


def run_learner():
    cfg = dict(ALPHA_ZERO_CONFIG)
    poll = float(cfg.get("learner_poll_interval_sec", 5.0) or 5.0)
    burst = int(cfg.get("updates_per_iter", 50) or 50)
    min_new = int(cfg.get("learner_min_new_samples_before_train", cfg.get("concurrent_min_new_samples_before_train", 2000)) or 2000)
    snapshot_sec = int(cfg.get("learner_replay_snapshot_interval_sec", 900) or 900)
    file_only = bool(cfg.get("learner_file_only_replay", False))
    _log_event(cfg, "starting learner")
    tr = Trainer(config=cfg)
    tr.setup()
    # メモリ上の共有リプレイは最小限保持。file_only モードでは ingest → 保存 → purge で常時空に近い状態。
    rb = ReplayBuffer(maxlen=int(cfg.get("learner_buffer_size", cfg.get("buffer_size", 250000))))
    pending = 0
    last_snapshot = time.time()
    snapshot_path = cfg.get("replay_path", "replay_buffer.joblib")
    # チェックポイント間隔 (エピソード基準) ※分離モード用: シャードメタ episodes を利用
    ckpt_interval_eps = int(cfg.get("checkpoint_interval_episodes", 0) or 0)
    next_ckpt_at = ckpt_interval_eps if ckpt_interval_eps > 0 else None
    while True:
        n = ingest_shards(tr, cfg, rb)
        if n > 0:
            pending += n
            if file_only:
                # 直ちにスナップショットへマージしてメモリ解放
                added = _merge_into_snapshot(cfg, rb)
                try:
                    rb.clear()
                except Exception:
                    pass
                # 既定では詳細ログを抑制（単一プロセス互換）。必要なら concurrent_debug_logging=True で出す。
                if cfg.get('concurrent_debug_logging'):
                    _log_event(cfg, f"merged into snapshot added={added} (file_only purge)")
        # スナップショットタイミング (学習の有無に関わらず)
        now = time.time()
        if snapshot_sec > 0 and (now - last_snapshot) >= snapshot_sec:
            try:
                if not file_only and len(rb) > 0:
                    rb.save(snapshot_path, purge=False)
                    if cfg.get('concurrent_debug_logging'):
                        _log_event(cfg, f"snapshot saved size={len(rb)} path='{os.path.basename(snapshot_path)}'")
                else:
                    # file_only: 既に都度マージしているため、存在確認のみ
                    if os.path.exists(snapshot_path):
                        if cfg.get('concurrent_debug_logging'):
                            sz = os.path.getsize(snapshot_path)
                            _log_event(cfg, f"snapshot touched bytes={sz}")
                last_snapshot = now
            except Exception as e:
                _log_event(cfg, f"snapshot save failed: {e}")
        # チェックポイント (エピソード間隔) 発火判定
        if next_ckpt_at is not None and _TOTAL_EPISODES_INGESTED >= next_ckpt_at:
            try:
                # version_tag に ep{番号} を付与 (単一プロセス互換)
                tr._save_checkpoint(version_tag=f"ep{_TOTAL_EPISODES_INGESTED}")  # type: ignore[attr-defined]
                _log_event(cfg, f"checkpoint saved at episodes={_TOTAL_EPISODES_INGESTED}")
            except Exception as e:
                _log_event(cfg, f"checkpoint save failed episodes={_TOTAL_EPISODES_INGESTED} err={e}")
            next_ckpt_at += ckpt_interval_eps  # 次の目標へ加算

        # 周期ステータスログ (Trainer 互換簡易版)
        status_interval = float(cfg.get("concurrent_status_log_sec", 0.0) or 0.0)
        include_mem = bool(cfg.get("status_log_include_memory", False))
        if status_interval > 0:
            if not hasattr(run_learner, "_last_status_ts"):
                run_learner._last_status_ts = time.time()  # type: ignore[attr-defined]
            if (time.time() - run_learner._last_status_ts) >= status_interval:  # type: ignore[attr-defined]
                rss_part = ""
                if include_mem:
                    try:
                        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
                        rss_part = f" rss={rss_mb:.1f}MB"
                    except Exception:
                        pass
                _log_event(cfg, f"status episodes={_TOTAL_EPISODES_INGESTED} pending_samples={pending} min_new={min_new}{rss_part}")
                run_learner._last_status_ts = time.time()  # type: ignore[attr-defined]

        # 学習トリガ
        if pending >= min_new:
            # file_only モード: 学習直前に再ロード (最新スナップショット + まだ purge されていないメモリ分)
            # 単一プロセス互換ログ: ここでまとめて行う学習を告知
            _log_event(cfg, f"ingested {pending} samples -> train {burst} updates")
            try:
                # 直前までのメモリ分を統合 (file_only の場合は空のはずだが冪等に)
                if file_only and len(rb) > 0:
                    _ = _merge_into_snapshot(cfg, rb)
                    rb.clear()
                # スナップショットからロードしてメモリに展開 (学習中のみ保持)
                try:
                    loaded = ReplayBuffer.load(snapshot_path)
                except Exception as e:
                    _log_event(cfg, f"snapshot load failed: {e}")
                    loaded = rb  # フォールバック: 直近 rb
                tr.shared_replay = loaded
                tr.train_updates(num_updates=burst)
                cand = _save_candidate(tr, cfg)
                if cand:
                    _log_event(cfg, f"saved candidate '{os.path.basename(cand)}'")
                # 学習用にロードしたメモリを解放
                tr.shared_replay = None
                pending = 0
            except Exception as e:
                _log_event(cfg, f"train_updates error: {e}")
        # 閾値未満の進捗ログは既定で抑制（単一プロセス互換）。必要なら debug フラグで出す。
        else:
            if pending > 0 and cfg.get('concurrent_debug_logging'):
                _log_event(cfg, f"pending below threshold pending={pending} (<{min_new})")
        time.sleep(poll)
        # RSS チェック（Learner 詳細版）
        try:
            if bool(cfg.get("worker_restart_enable", False)):
                now_t = time.time()
                check_interval = 30.0
                if (now_t - _rss_state.last_check) >= check_interval:
                    _rss_state.last_check = now_t
                    rss_high_mb = float(cfg.get("worker_restart_rss_high_mb", 0) or 0)
                    rss_low_mb = float(cfg.get("worker_restart_rss_low_mb", 0) or 0)
                    consec_req = int(cfg.get("worker_restart_consecutive_required", 3) or 3)
                    min_interval = float(cfg.get("worker_restart_min_interval_sec", 1200) or 1200)
                    jitter_sec = float(cfg.get("worker_restart_jitter_sec", 0) or 0)
                    emergency_total_mb = float(cfg.get("worker_restart_emergency_total_mb", 0) or 0)
                    include_children = bool(cfg.get("replay_memory_include_children", False))
                    p = psutil.Process(os.getpid())
                    rss_mb = p.memory_info().rss / (1024*1024)
                    total_mb = rss_mb
                    if include_children:
                        for c in p.children(recursive=True):
                            try:
                                total_mb += c.memory_info().rss / (1024*1024)
                            except Exception:
                                pass
                    emergency = emergency_total_mb > 0 and total_mb >= emergency_total_mb
                    do_restart = False
                    if emergency:
                        do_restart = True
                    elif rss_high_mb > 0 and rss_mb >= rss_high_mb:
                        _rss_state.consecutive_high += 1
                        if _rss_state.consecutive_high >= consec_req:
                            do_restart = True
                    else:
                        # ヒステリシス (低水位に戻ればカウンタリセット)
                        if rss_low_mb <= 0 or rss_mb <= rss_low_mb:
                            _rss_state.consecutive_high = 0
                    if do_restart:
                        if (now_t - _rss_state.last_restart_time) < min_interval:
                            _log_event(cfg, f"[restart] skipped(min_interval) rss={rss_mb:.1f}MB total={total_mb:.1f}MB")
                        else:
                            _rss_state.last_restart_time = now_t
                            # jitter
                            if jitter_sec > 0:
                                import random as _r
                                time.sleep(_r.uniform(0, jitter_sec))
                            _log_event(cfg, f"[restart] learner exit rss={rss_mb:.1f}MB total={total_mb:.1f}MB high={rss_high_mb}MB consec={_rss_state.consecutive_high}")
                            break
        except Exception:
            pass


if __name__ == '__main__':
    run_learner()
