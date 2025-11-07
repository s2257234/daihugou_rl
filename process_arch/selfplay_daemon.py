from __future__ import annotations
"""Self-Play daemon process: generates samples and writes shard files.

Minimal decoupled producer.

Usage (PowerShell):
  python -m process_arch.selfplay_daemon --config agents/config.py --episodes 0
    (episodes=0 => infinite loop)

Stops when selfplay_max_episodes>0 and reached or a stop flag file exists: sample_shards/STOP
"""
import os
import sys
import time
import argparse
import psutil
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from agents.drl_agent import AlphaZeroAgent
from evaluation.gating import _override_eval_config  # reuse noise disable if needed
from utils.shards import write_samples_as_shard, ensure_dir

try:
    from game.environment import DaifugoSimpleEnv
except Exception as e:  # pragma: no cover
    raise RuntimeError("Environment import failed") from e


def build_agents(model, cfg: Dict[str, Any]) -> List[AlphaZeroAgent]:
    agents: List[AlphaZeroAgent] = []
    for pid in range(int(cfg.get("num_players", 4))):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=cfg)
        # queue/shard モード: 共有リプレイではなく確定サンプルを一時保持
        try:
            ag.replay_buffer = None  # worker_zero_buffer 相当
        except Exception:
            pass
        agents.append(ag)
    return agents


def load_model(path: str, cfg: Dict[str, Any]):
    from agents.models import PolicyValueNet
    if not os.path.exists(path):
        # fresh init
        n = int(cfg.get("num_players", 4) or 4)
        full_dim = 59 * n + 125  # v4 layout: 59N + 125
        model = PolicyValueNet(
            max_policy_size=int(cfg.get("max_policy_size", 128) or 128),
            hidden_size=int(cfg.get("hidden_size", 128) or 128),
            num_players=n,
            device="cpu",
            use_full_features=True,
            full_feature_dim=full_dim,
        )
        return model
    return PolicyValueNet.load(path, map_location="cpu")

def _log_event(cfg: Dict[str, Any], msg: str) -> None:
    try:
        import datetime
        log_dir = cfg.get('log_dir', 'logs')
        ensure_dir(log_dir)
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] [SELFPLAY] {msg}\n")
    except Exception:
        pass


def run_selfplay(cfg: Dict[str, Any]):
    shard_dir = cfg.get("sample_shard_dir", "sample_shards")
    stop_flag = os.path.join(shard_dir, "STOP")
    best_model_path = cfg.get("best_model_path", cfg.get("checkpoint_path"))
    model_reload_interval = float(cfg.get("selfplay_model_reload_interval_sec", 30.0) or 30.0)
    max_eps = int(cfg.get("selfplay_max_episodes", 0) or 0)
    ext = cfg.get("sample_shard_ext", ".shard.joblib")
    shard_max = int(cfg.get("sample_shard_max_samples", 2000) or 2000)
    refresh_flag = cfg.get("model_refresh_flag_path", os.path.join(cfg.get("checkpoint_dir","checkpoints"), "_refresh.flag"))

    model = load_model(best_model_path, cfg)
    agents = build_agents(model, cfg)
    env = DaifugoSimpleEnv(num_players=cfg.get("num_players",4), agent_classes=None)
    env.agents = agents
    for ag in agents:
        ag.set_env_ref(env)

    last_reload = time.time()
    last_model_mtime = None
    episodes_done = 0
    pending_samples: List[Dict[str, Any]] = []
    pending_episodes = 0  # 現在メモリ上の pending_samples に含まれている完了エピソード数
    status_interval = float(cfg.get("concurrent_status_log_sec", 0.0) or 0.0)
    include_mem = bool(cfg.get("status_log_include_memory", False))
    last_status_ts = time.time()
    _log_event(cfg, f"start best_model='{best_model_path}' shard_dir='{shard_dir}'")
    # RSS 再起動関連設定
    restart_enable = bool(cfg.get("worker_restart_enable", False))
    rss_high_mb = float(cfg.get("worker_restart_rss_high_mb", 0) or 0)
    rss_low_mb = float(cfg.get("worker_restart_rss_low_mb", 0) or 0)
    rss_consecutive_req = int(cfg.get("worker_restart_consecutive_required", 3) or 3)
    restart_min_interval = float(cfg.get("worker_restart_min_interval_sec", 1200) or 1200)
    restart_jitter = float(cfg.get("worker_restart_jitter_sec", 0) or 0)
    emergency_total_mb = float(cfg.get("worker_restart_emergency_total_mb", 0) or 0)
    grace_timeout = int(cfg.get("worker_restart_grace_timeout_sec", 180) or 180)
    force_kill_sec = int(cfg.get("worker_restart_force_kill_sec", 240) or 240)
    flush_timeout = int(cfg.get("worker_restart_flush_timeout_sec", 20) or 20)
    log_objects_on_exit = bool(cfg.get("worker_restart_log_object_types_on_exit", False))

    rss_check_interval = 30.0  # 固定の低頻度チェック
    last_rss_check = 0.0
    consecutive_high = 0
    last_restart_time = 0.0

    def _current_rss_mb(include_children: bool = False) -> float:
        try:
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss
            if include_children and cfg.get("replay_memory_include_children", False):
                for c in p.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
            return rss / (1024*1024)
        except Exception:
            return 0.0

    def _log_mem(msg: str):
        _log_event(cfg, msg)

    while True:
        if max_eps > 0 and episodes_done >= max_eps:
            print("[SELFPLAY] reached max episodes; exit")
            break
        if os.path.exists(stop_flag):
            print("[SELFPLAY] STOP flag detected; exit")
            break
        # reload model periodically if file mtime changed
        flag_reload = False
        try:
            if os.path.exists(refresh_flag):
                flag_reload = True
        except Exception:
            flag_reload = False
        # reload only if interval elapsed AND model file actually changed OR refresh flag set
        need_interval = (time.time() - last_reload >= model_reload_interval)
        file_changed = False
        try:
            if os.path.exists(best_model_path):
                cur_mtime = os.path.getmtime(best_model_path)
                if last_model_mtime is None or cur_mtime > last_model_mtime + 1e-6:
                    file_changed = True
        except Exception:
            file_changed = False
        if flag_reload or (need_interval and file_changed):
            try:
                # naive reload (always) when interval or flag
                model = load_model(best_model_path, cfg)
                for ag in agents:
                    ag.set_model(model)
                _log_event(cfg, "reloaded model")
                if flag_reload:
                    try:
                        os.remove(refresh_flag)
                    except Exception:
                        pass
                try:
                    if os.path.exists(best_model_path):
                        last_model_mtime = os.path.getmtime(best_model_path)
                except Exception:
                    pass
            except Exception as e:
                _log_event(cfg, f"model reload failed: {e}")
            last_reload = time.time()
        # episode
        env.reset()
        # agent episode reset if available
        for ag in agents:
            if hasattr(ag, 'reset_episode'):
                try:
                    ag.reset_episode()
                except Exception:
                    pass
        steps = 0
        max_steps = int(cfg.get("max_episode_steps", 1000) or 1000)
        prev_rankings = list(getattr(env.game, 'rankings', []))
        while not getattr(env.game, 'done', False):
            if steps >= max_steps:
                break
            pid = env.game.turn
            ag = env.agents[pid]
            act = ag.select_action(env, training=True)
            try:
                env.step(external_action=act)
            except TypeError:
                env.step(act)
            steps += 1
            # フェーズ確定検知（誰かが上がったら）
            cur_rankings = list(getattr(env.game, 'rankings', []))
            if len(cur_rankings) > len(prev_rankings):
                new_winners = cur_rankings[len(prev_rankings):]
                for w in new_winners:
                    for az in agents:
                        was_active = az.player_id not in prev_rankings
                        try:
                            az.finalize_phase(winner_player_id=w, was_active=was_active)
                        except Exception:
                            pass
                prev_rankings = cur_rankings
        # finalize phase/value labels inside agents (similar to trainer._finalize_episode_rewards logic)
        for ag in agents:
            try:
                ag.flush_unfinished_phase()
                ag.finalize_game()
            except Exception:
                pass
            # flush confirmed samples from agent internal buffer if any
            ep_samples = getattr(ag, '_episode_confirmed_samples', [])
            if ep_samples:
                pending_samples.extend(ep_samples)
                try:
                    ag._episode_confirmed_samples = []
                except Exception:
                    pass
        episodes_done += 1
        pending_episodes += 1
        # shard flush condition
        if len(pending_samples) >= shard_max:
            path = write_samples_as_shard(pending_samples, shard_dir, ext, episodes=pending_episodes)
            if path:
                _log_event(cfg, f"shard written path='{os.path.basename(path)}' samples={len(pending_samples)} episodes={pending_episodes}")
            pending_samples.clear()
            pending_episodes = 0

        # 周期ステータスログ
        if status_interval > 0 and (time.time() - last_status_ts) >= status_interval:
            rss_part = ""
            if include_mem:
                try:
                    import psutil as _ps
                    rss_mb = _ps.Process(os.getpid()).memory_info().rss / (1024*1024)
                    rss_part = f" rss={rss_mb:.1f}MB"
                except Exception:
                    rss_part = ""
            shard_count = 0
            try:
                if os.path.isdir(shard_dir):
                    shard_count = sum(1 for f in os.listdir(shard_dir) if f.endswith(ext))
            except Exception:
                shard_count = 0
            _log_event(cfg, f"status ep_done={episodes_done} pending_samples={len(pending_samples)} shard_files={shard_count}{rss_part}")
            last_status_ts = time.time()

        # RSS チェック & 再起動判定 (自己対局ワーカー単体版)
        now_t = time.time()
        if restart_enable and (now_t - last_rss_check) >= rss_check_interval:
            last_rss_check = now_t
            rss_mb = _current_rss_mb(include_children=True)
            emergency_triggered = False
            if emergency_total_mb > 0:
                try:
                    total_mb = rss_mb  # 子含めて計算済み
                    if total_mb >= emergency_total_mb:
                        emergency_triggered = True
                except Exception:
                    emergency_triggered = False
            if emergency_triggered:
                _log_mem(f"[restart] emergency total_rss={rss_mb:.1f}MB >= {emergency_total_mb}MB -> graceful restart")
                do_restart = True
            else:
                do_restart = False
                if rss_high_mb > 0 and rss_mb >= rss_high_mb:
                    consecutive_high += 1
                    if consecutive_high >= rss_consecutive_req:
                        do_restart = True
                else:
                    consecutive_high = 0
            if do_restart:
                # 最低間隔
                if (now_t - last_restart_time) < restart_min_interval:
                    _log_mem(f"[restart] skipped (min_interval) rss={rss_mb:.1f}MB")
                else:
                    last_restart_time = now_t
                    # graceful: 現在の pending サンプルを flush してから終了
                    try:
                        if pending_samples:
                            pth = write_samples_as_shard(pending_samples, shard_dir, ext, episodes=pending_episodes)
                            if pth:
                                _log_event(cfg, f"graceful flush before restart path='{os.path.basename(pth)}' samples={len(pending_samples)} episodes={pending_episodes}")
                            pending_samples.clear()
                            pending_episodes = 0
                    except Exception:
                        pass
                    # オブジェクト種類ログ (軽量): agent 内部属性数など
                    if log_objects_on_exit:
                        try:
                            ag = agents[0]
                            keys = []
                            for k in dir(ag):
                                if k.startswith('_'):
                                    continue
                                try:
                                    v = getattr(ag, k)
                                    keys.append(k)
                                except Exception:
                                    pass
                            _log_mem(f"[restart] object_keys_count={len(keys)}")
                        except Exception:
                            pass
                    _log_mem(f"[restart] exiting for RSS rss={rss_mb:.1f}MB high={rss_high_mb}MB consecutive={consecutive_high}")
                    break
    # final flush
    if pending_samples:
        path = write_samples_as_shard(pending_samples, shard_dir, ext, episodes=pending_episodes)
        if path:
            _log_event(cfg, f"final shard path='{os.path.basename(path)}' samples={len(pending_samples)} episodes={pending_episodes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=0, help='最大エピソード数 (0=無限)')
    ap.add_argument('--config', type=str, default=None, help='未使用: config.py を直接編集してください')
    args = ap.parse_args()
    cfg = dict(ALPHA_ZERO_CONFIG)
    if args.episodes:
        cfg['selfplay_max_episodes'] = args.episodes
    run_selfplay(cfg)

if __name__ == '__main__':
    main()
