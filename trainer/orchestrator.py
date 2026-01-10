from __future__ import annotations

import os
import time
import random
import multiprocessing as mp
from typing import Any, Dict, List

from agents.drl_agent import AlphaZeroAgent
from utils.process_monitor import ProcessMonitor, auto_replay_water_purge
from evaluation.gating import start_async_gate, finalize_async_gate_if_ready

# Worker entry points defined in trainer.trainer (spawn-safe top-level)
from trainer.trainer import _selfplay_worker_entry, _selfplay_daemon_worker  # type: ignore


def self_play(trainer, num_episodes: int = 1):
    # 並列数で分岐
    workers = int(trainer.config.get("selfplay_workers", 0) or 0)
    if workers and workers > 1:
        return _self_play_parallel(trainer, num_episodes=num_episodes, workers=workers)

    start_time = time.time()
    # 実行開始直後に 0% 進捗を表示して無音時間を減らす
    if trainer.minimal_progress and trainer.use_progress_bar and num_episodes > 0:
        bar, _pct = trainer._make_progress_bar(0, num_episodes)
        line = f"[SELFPLAY] {bar} 0/{num_episodes}"
        print(line, end='\r', flush=True)
        trainer._last_progress_len = len(line)
    for ep in range(num_episodes):
        ep_start = time.time()
        if not trainer.minimal_progress and not trainer.use_progress_bar:
            print(f"[EPOCH] {ep+1}/{num_episodes}")
        # ウォームアップ期間中は対戦相手をランダム/ルールベースに
        if trainer._episodes_total_run < trainer.warmup_episodes:
            trainer._apply_warmup_opponents()
        elif trainer._episodes_total_run == trainer.warmup_episodes:
            trainer._restore_learning_agents()
        trainer._play_one_episode(ep)
        trainer._episodes_total_run += 1
        # 周期チェックポイント保存
        if trainer.ckpt_interval > 0 and (trainer._episodes_total_run % trainer.ckpt_interval == 0):
            trainer._save_checkpoint(version_tag=f"ep{trainer._episodes_total_run}")
            if trainer.keep_prev_model:
                trainer._snapshot_current_model()
            if trainer.keep_prev_model and trainer.prev_model_mix_players > 0:
                if trainer.past_models:
                    trainer._assign_past_models_to_opponents()
                elif trainer._previous_model is not None:
                    trainer._mix_previous_model_opponents()
            trainer.model_version += 1
            for ag in trainer.agents:
                if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                    ag.model_version = trainer.model_version
        if trainer.opponent_mix_interval > 0 and trainer.keep_prev_model and trainer.prev_model_mix_players > 0:
            if (trainer._episodes_total_run % trainer.opponent_mix_interval == 0) and trainer.past_models:
                trainer._assign_past_models_to_opponents()

        # High/Low water purge (single-process self_play)
        try:
            rb = trainer.shared_replay or (trainer.agents and getattr(trainer.agents[0], 'replay_buffer', None))
            trainer._mem_purge_state = auto_replay_water_purge(trainer.config, rb, trainer.logger, state=getattr(trainer, '_mem_purge_state', None))
        except Exception:
            pass

        # 進捗表示 (エピソード終了後に確定時間で ETA 推定)
        if trainer.minimal_progress and trainer.use_progress_bar:
            done = ep + 1
            ep_dur = time.time() - ep_start
            if trainer._eta_smooth is None:
                trainer._eta_smooth = ep_dur
            else:
                a = max(0.0, min(1.0, trainer.eta_alpha))
                trainer._eta_smooth = a * ep_dur + (1 - a) * trainer._eta_smooth
            elapsed = time.time() - start_time
            estimated_total = (trainer._eta_smooth or 0.0) * num_episodes
            remaining = max(0.0, estimated_total - elapsed)
            if trainer.monotonic_eta and trainer._eta_prev_remaining is not None and remaining > trainer._eta_prev_remaining:
                remaining = trainer._eta_prev_remaining
            trainer._eta_prev_remaining = remaining
            def _fmt(t: float):
                m, s = divmod(int(t), 60)
                h, m = divmod(m, 60)
                return f"{h:d}:{m:02d}:{s:02d}"
            bar, _pct = trainer._make_progress_bar(done, num_episodes)
            line = f"[SELFPLAY] {bar} {done}/{num_episodes} eta={_fmt(remaining)}"
            pad = max(0, trainer._last_progress_len - len(line))
            print(line + ' ' * pad, end='\r' if done < num_episodes else '\n', flush=True)
            trainer._last_progress_len = len(line)
    try:
        if trainer.logger and getattr(trainer.logger, 'csv_summary_only', False):
            trainer.logger.write_csv_summaries()
    except Exception as e:
        print(f"[WARN] write_csv_summaries(self_play) failed: {e}")
    return


def train_concurrent(
    trainer,
    *,
    total_episodes: int,
    workers: int | None = None,
    updates_per_iter: int = 50,
    queue_maxsize: int = 15000,
    progress_print_every: int = 50,
):
    assert total_episodes > 0
    workers = int(workers if workers is not None else (trainer.config.get("selfplay_workers", 0) or 0))
    workers = max(1, workers)

    os.makedirs(trainer.config["checkpoint_dir"], exist_ok=True)
    model_blob_path = trainer.config.get("checkpoint_path", os.path.join(trainer.config["checkpoint_dir"], "policy_value_latest.pt"))
    if (trainer.model is not None) and (not os.path.exists(model_blob_path)):
        try:
            trainer.model.save(model_blob_path)
        except Exception:
            pass

    dst_buffer = trainer.shared_replay if trainer.shared_replay is not None else None
    learner = trainer.agents[trainer.learning_player_id]
    if dst_buffer is None and isinstance(learner, AlphaZeroAgent):
        dst_buffer = learner.replay_buffer

    ctx = mp.get_context("spawn")
    sample_queue = ctx.Queue(maxsize=max(1000, queue_maxsize))
    event_queue = ctx.Queue(maxsize=10000)
    stop_event = ctx.Event()
    request_q = None
    response_queues: List[Any] = [None for _ in range(workers)]
    control_queues: List[Any] = []

    _master_thread = None

    procs: List[mp.Process] = []
    for wid in range(workers):
        cq = ctx.Queue(maxsize=5)
        control_queues.append(cq)
        p = ctx.Process(
            target=_selfplay_daemon_worker,
            args=(wid, trainer.config, model_blob_path, sample_queue, event_queue, stop_event, cq, request_q, response_queues[wid]),
            daemon=True,
        )
        p.start()
        procs.append(p)

    if trainer.minimal_progress and trainer.use_progress_bar:
        bar, _pct = trainer._make_progress_bar(0, total_episodes)
        line = f"[SELFPLAY~] {bar} 0/{total_episodes} (workers={workers})"
        print(line, end='\r', flush=True)
        last_len_sp = len(line)
    else:
        last_len_sp = 0

    ep_done = 0
    train_it = 0
    ckpt_interval = int(trainer.ckpt_interval or 0)
    min_new_samples_before_train = int(trainer.config.get("concurrent_min_new_samples_before_train", 2000) or 2000)
    new_samples_since_train = 0
    last_train_ts = time.time()
    latest_ckpt_interval_sec = float(trainer.config.get("concurrent_latest_save_every_sec", 30.0) or 0.0)
    blob_save_interval_sec = float(trainer.config.get("concurrent_blob_save_every_sec", 30.0) or 0.0)
    last_latest_ckpt_ts = 0.0
    last_blob_save_ts = 0.0
    status_log_sec = float(trainer.config.get("concurrent_status_log_sec", 0) or 0)
    status_log_include_mem = bool(trainer.config.get("status_log_include_memory", False))
    debug_flag = bool(trainer.config.get("concurrent_debug_logging", False))
    last_status_ts = time.time()

    def _drain_samples(max_items: int | None = None):
        nonlocal dst_buffer
        consumed = 0
        skipped = 0
        import random as _r
        while True:
            if max_items is not None and consumed >= max_items:
                break
            try:
                s = sample_queue.get_nowait()
            except Exception:
                break
            if not isinstance(s, dict):
                skipped += 1
                continue
            if 'state' not in s or not ('pi' in s or 'pi_q' in s):
                skipped += 1
                continue
            try:
                if s.get('split') is None:
                    ratio = float(trainer.config.get('val_split_ratio', 0.0) or 0.0)
                    s['split'] = 'val' if (_r.random() < ratio) else 'train'
            except Exception:
                pass
            if trainer.shared_replay is not None:
                try:
                    if bool(trainer.config.get('replay_async_enabled', False)) and hasattr(trainer.shared_replay, 'append_async'):
                        drop_oldest = bool(trainer.config.get('replay_async_drop_oldest', True))
                        trainer.shared_replay.append_async(s, drop_oldest=drop_oldest)
                    else:
                        trainer.shared_replay.append(s)
                except Exception:
                    pass
            else:
                dst_buffer.append(s)  # type: ignore[attr-defined]
            consumed += 1
        if debug_flag and skipped > 0:
            print(f"[DEBUG] drain skipped={skipped} accepted={consumed} (reason: missing state or pi/pi_q)")
        return consumed

    restart_cfg = {
        'enable': bool(trainer.config.get('worker_restart_enable', False)),
        'high_mb': int(trainer.config.get('worker_restart_rss_high_mb', 0) or 0),
        'consecutive': int(trainer.config.get('worker_restart_consecutive_required', 2) or 2),
        'min_interval': int(trainer.config.get('worker_restart_min_interval_sec', 600) or 600),
        'jitter': int(trainer.config.get('worker_restart_jitter_sec', 0) or 0),
        'emergency_total_mb': int(trainer.config.get('worker_restart_emergency_total_mb', 0) or 0),
        'grace_timeout': int(trainer.config.get('worker_restart_grace_timeout_sec', 120) or 120),
        'force_kill_sec': int(trainer.config.get('worker_restart_force_kill_sec', 150) or 150),
        'flush_timeout': int(trainer.config.get('worker_restart_flush_timeout_sec', 20) or 20),
        'log_obj_on_exit': bool(trainer.config.get('worker_restart_log_object_types_on_exit', True)),
    }
    worker_stats = {}
    for idx, p in enumerate(procs):
        worker_stats[idx] = {
            'rss_high_count': 0,
            'last_restart': 0.0,
            'generation': 0,
            'pending': False,
            'grace_start': None,
        }
    def _clear_gate_handles():
        try:
            trainer._gate_proc = None
            trainer._gate_queue = None
            trainer._gate_candidate_path = None
            trainer._gate_baseline_path = None
        except Exception:
            pass
    monitor = ProcessMonitor(
        restart_cfg,
        logger=trainer.logger,
        get_gate_proc=lambda: getattr(trainer, "_gate_proc", None),
        clear_gate_handles=_clear_gate_handles,
    )

    last_worker_rss_check = 0.0
    worker_rss_check_interval = 15.0

    try:
        while ep_done < total_episodes:
            try:
                evt, val = event_queue.get(timeout=0.1)
            except Exception:
                evt = None
            if evt == "ep_done":
                ep_done += int(val)
                trainer._episodes_total_run += int(val)
                if ckpt_interval > 0 and (trainer._episodes_total_run % ckpt_interval == 0):
                    try:
                        trainer._save_checkpoint(version_tag=f"ep{trainer._episodes_total_run}")
                        if trainer.keep_prev_model:
                            trainer._snapshot_current_model()
                        if trainer.keep_prev_model and trainer.prev_model_mix_players > 0:
                            if trainer.past_models:
                                trainer._assign_past_models_to_opponents()
                            elif trainer._previous_model is not None:
                                trainer._mix_previous_model_opponents()
                        trainer.model_version += 1
                        for ag in trainer.agents:
                            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                                ag.model_version = trainer.model_version
                    except Exception:
                        pass
            elif evt == "play_stats":
                try:
                    st = val or {}
                    wid = st.get('wid')
                    w_ep = st.get('ep')
                    def _fmt(x):
                        return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None else (str(x) if x is not None else 'n/a')
                    line = (
                        f"[playstats] wid={wid} ep={w_ep} "
                        f"moves_avg={_fmt(st.get('moves_avg'))} moves_p95={_fmt(st.get('moves_p95'))} "
                        f"sims_avg={_fmt(st.get('sims_avg'))} sims_p50={_fmt(st.get('sims_p50'))} sims_p95={_fmt(st.get('sims_p95'))} "
                        f"early_stop_rate={_fmt(st.get('early_stop_rate'))} n={int(st.get('sample_size') or 0)}"
                    )
                    if trainer.logger:
                        trainer.logger.log_text(line)
                    else:
                        print(line)
                except Exception:
                    pass
                if trainer.minimal_progress and trainer.use_progress_bar:
                    bar, _pct = trainer._make_progress_bar(ep_done, total_episodes)
                    l2 = f"[SELFPLAY~] {bar} {ep_done}/{total_episodes} (workers={workers})"
                    pad = max(0, last_len_sp - len(l2))
                    print(l2 + ' ' * pad, end='\r' if ep_done < total_episodes else '\n', flush=True)
                    last_len_sp = len(l2)
            elif evt == "perf_ep":
                try:
                    if not (bool(trainer.config.get('measure_forward_time', False)) and bool(trainer.config.get('measure_game_time', False))):
                        raise RuntimeError("perf_ep_disabled")
                    st = val or {}
                    wid = st.get('wid')
                    w_ep = st.get('ep')
                    tfwd = st.get('t_forward_avg_ms')
                    tg = st.get('t_game_sec')
                    mv = st.get('moves')
                    line = f"[perf-ep] wid={wid} ep={w_ep} t_forward_ms={tfwd} t_game_sec={tg} moves={mv}"
                    if trainer.logger:
                        trainer.logger.log_text(line)
                    else:
                        print(line)
                except Exception:
                    pass
                if trainer.minimal_progress and trainer.use_progress_bar:
                    bar, _pct = trainer._make_progress_bar(ep_done, total_episodes)
                    l2 = f"[SELFPLAY~] {bar} {ep_done}/{total_episodes} (workers={workers})"
                    pad = max(0, last_len_sp - len(l2))
                    print(l2 + ' ' * pad, end='\r' if ep_done < total_episodes else '\n', flush=True)
                    last_len_sp = len(l2)
            elif evt == "hb":
                try:
                    wid, steps = val
                    if trainer.logger:
                        trainer.logger.log_text(f"[worker-hb] wid={wid} steps={steps}")
                except Exception:
                    pass

            consumed_now = _drain_samples(max_items=500)
            new_samples_since_train += int(consumed_now)
            if consumed_now > 0:
                try:
                    rb = trainer.shared_replay or (trainer.agents and getattr(trainer.agents[0], 'replay_buffer', None))
                    trainer._mem_purge_state = auto_replay_water_purge(trainer.config, rb, trainer.logger, state=getattr(trainer, '_mem_purge_state', None))
                except Exception:
                    pass
            if debug_flag and consumed_now>0:
                print(f"[DEBUG] drained={consumed_now} total_new={new_samples_since_train} replay_size={len(trainer.shared_replay) if trainer.shared_replay else 'n/a'}")
            if debug_flag and (ep_done % max(1, int(trainer.config.get('debug_status_interval_eps', 25))) == 0):
                try:
                    if trainer.shared_replay is not None:
                        total_rb = len(trainer.shared_replay)
                        labeled_rb = 0
                        try:
                            for rec in trainer.shared_replay.iter_all():
                                if isinstance(rec, dict) and rec.get('value') is not None:
                                    labeled_rb += 1
                        except Exception:
                            pass
                        print(f"[DEBUG][parent] ep_done={ep_done} shared_replay_total={total_rb} labeled={labeled_rb}")
                except Exception:
                    pass

            now = time.time()
            force_interval_sec = float(trainer.config.get("concurrent_force_train_interval_sec", 0) or 0)
            force_due = False
            if force_interval_sec > 0 and (time.time() - last_train_ts) >= force_interval_sec and new_samples_since_train > 0:
                force_due = True
                if debug_flag:
                    print(f"[DEBUG] force-train interval reached ({force_interval_sec}s) new_since_train={new_samples_since_train} (< threshold {min_new_samples_before_train})")
            if new_samples_since_train >= min_new_samples_before_train or force_due:
                burst_start_ts = time.time()
                last_loss_val = None
                for _ in range(max(1, int(updates_per_iter))):
                    loss_info = trainer.agents[0].train_step(batch_size=trainer.config.get("batch_size", 256))
                    if isinstance(loss_info, dict) and loss_info.get("loss") is None:
                        samples_cnt = None
                        try:
                            if trainer.shared_replay is not None and hasattr(trainer.shared_replay, '__len__'):
                                samples_cnt = len(trainer.shared_replay)
                        except Exception:
                            samples_cnt = None
                        loss_info.setdefault("policy_loss", 0.0)
                        loss_info.setdefault("value_loss", 0.0)
                        loss_info.setdefault("hand_pred_loss", 0.0)
                        loss_info.setdefault("entropy", 0.0)
                        loss_info.setdefault("policy_kl", 0.0)
                        loss_info.setdefault("policy_top1_match", 0.0)
                        loss_info.setdefault("value_acc", 0.0)
                        loss_info.setdefault("value_brier", 0.0)
                        loss_info.setdefault("pos_rate", 0.0)
                        loss_info.setdefault("cum_pos_rate", 0.0)
                        if samples_cnt is not None:
                            loss_info.setdefault("samples", samples_cnt)
                        loss_info["loss"] = float(loss_info.get("policy_loss", 0.0)) + float(loss_info.get("value_loss", 0.0))
                    if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                        last_loss_val = float(loss_info.get("loss"))
                    if trainer.config.get("purge_replay_after_each_update"):
                        try:
                            trainer._save_checkpoint()
                            new_samples_since_train = 0
                            if trainer.logger:
                                trainer.logger.log_text("[replay] immediate_purge_after_update")
                        except Exception as _e:
                            print(f"[WARN] immediate save after update failed: {_e}")
                    train_it += 1
                    if trainer.minimal_progress and trainer.use_progress_bar:
                        if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                            loss_part = f"loss={loss_info['loss']:.4f}"
                        else:
                            loss_part = "loss=----"
                        lr_val = None
                        try:
                            opt = getattr(trainer.agents[0], '_optimizer', None)
                            if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                                lr_val = opt.param_groups[0].get('lr', None)
                        except Exception:
                            lr_val = None
                        lr_part = f"lr={lr_val:.2e}" if lr_val is not None else "lr=----"
                        line = f"[TRAIN~] it={train_it} {loss_part} {lr_part}"
                        print(line, end='\r', flush=True)
                    if trainer.logger and isinstance(loss_info, dict) and loss_info.get("loss") is not None and not getattr(trainer.agents[0], '_logged_inside', False):
                        trainer.logger.log_train(loss_info)
                    # 検証: 毎回実行してログ出力（検証ロスは毎回計算し、train_updates.csvに記録する）
                    if trainer.logger:
                        try:
                            vinfo = trainer.agents[0].validate_step(batch_size=trainer.config.get("val_batch_size") or trainer.config.get("batch_size", 256))
                        except Exception:
                            vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                        # 検証結果をログに記録（None値でも記録して、CSVの列を埋める）
                        if isinstance(vinfo, dict):
                            trainer.logger.log_validation(vinfo)

                last_train_ts = time.time()
                new_samples_since_train = 0

                def _resolve_device_str(dev_str: str | None) -> str:
                    if not dev_str or dev_str == "auto":
                        try:
                            import torch as _t
                            return "cuda" if _t.cuda.is_available() else "cpu"
                        except Exception:
                            return "cpu"
                    return dev_str

                def _maybe_gate_before_save():
                    gate_enable = bool(trainer.config.get("eval_gate_enable", False))
                    if not gate_enable or trainer.model is None:
                        return None
                    try:
                        _train_it = int(train_it)
                    except Exception:
                        _train_it = 0
                    try:
                        start_after = int(trainer.config.get("eval_gate_start_after_updates", 0) or 0)
                    except Exception:
                        start_after = 0
                    try:
                        every = int(trainer.config.get("eval_gate_every_updates", 0) or 0)
                    except Exception:
                        every = 0
                    if _train_it < start_after:
                        return None
                    if every > 0 and getattr(trainer, "_last_gate_start_it", -1) >= 0 and (_train_it - trainer._last_gate_start_it) < every:
                        return None
                    baseline_model = None
                    try:
                        from agents.models import PolicyValueNet as _PVN
                        base_path = trainer.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                        if os.path.exists(base_path):
                            dev_str = _resolve_device_str(trainer.config.get("device", None))
                            baseline_model = _PVN.load(base_path, map_location=dev_str)
                            try:
                                dev = _resolve_device_str(trainer.config.get("device", None))
                                if dev:
                                    baseline_model.to(dev)  # type: ignore[arg-type]
                            except Exception:
                                pass
                    except Exception as _e:
                        if trainer.logger:
                            try:
                                trainer.logger.log_text(f"[WARN] eval-gate baseline load failed (concurrent): {_e}")
                            except Exception:
                                pass
                        baseline_model = None
                    if baseline_model is None:
                        return None
                    try:
                        from evaluation.gating import evaluate_candidate
                        games = int(trainer.config.get("eval_gate_games", 20) or 20)
                        thr = float(trainer.config.get("eval_gate_threshold", 0.6) or 0.6)
                        seed = trainer.config.get("eval_gate_seed", None)
                        try:
                            if trainer.logger:
                                trainer.logger.log_text(f"[gate] start sync games={games} thr={thr:.0%} seed={seed}")
                            else:
                                print(f"[gate] start sync games={games} thr={thr:.0%} seed={seed}")
                        except Exception:
                            pass
                        trainer._last_gate_start_it = int(_train_it)
                        gate_result = evaluate_candidate(trainer.model, baseline_model, trainer.config, games=games, seed=seed)
                        gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                        trainer._last_gate_threshold = float(thr)
                        trainer._last_gate_result = dict(gate_result)
                        try:
                            wins = gate_result.get('wins')
                            total = gate_result.get('games')
                            decision = 'promote' if gated_pass else 'reject'
                            msg = f"[gate] result win_rate={gate_result.get('win_rate',0.0):.2%}"
                            if wins is not None and total is not None:
                                msg += f" ({int(wins)}/{int(total)})"
                            msg += f" decision={decision} thr={thr:.0%}"
                            if trainer.logger:
                                trainer.logger.log_text(msg)
                            else:
                                print(msg)
                        except Exception:
                            pass
                        if not gated_pass:
                            trainer.model = baseline_model
                            learner = trainer.agents[trainer.learning_player_id]
                            if isinstance(learner, AlphaZeroAgent):
                                learner.set_model(trainer.model)
                        return gate_result
                    except Exception as e:
                        if trainer.logger:
                            try:
                                trainer.logger.log_text(f"[WARN] eval-gate failed in concurrent save: {e}")
                            except Exception:
                                pass
                        return None

                async_gate = bool(trainer.config.get("eval_gate_async", False))
                gate_enable_flag = bool(trainer.config.get("eval_gate_enable", False))
                if latest_ckpt_interval_sec > 0.0:
                    if (now - last_latest_ckpt_ts) >= latest_ckpt_interval_sec:
                        if async_gate and gate_enable_flag:
                            if not trainer._should_start_gate(train_it):
                                pass
                            else:
                                try:
                                    cand_path = os.path.join(trainer.config.get("checkpoint_dir", "checkpoints"), "_candidate_eval.pt")
                                    base_path = trainer.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                                    if trainer.model is not None:
                                        try:
                                            trainer.model.save(cand_path, force_sync=True)
                                        except TypeError:
                                            trainer.model.save(cand_path)
                                    if getattr(trainer, "_gate_proc", None) is None or (trainer._gate_proc is not None and not trainer._gate_proc.is_alive()):
                                        games = int(trainer.config.get("eval_gate_games", 20) or 20)
                                        seed = trainer.config.get("eval_gate_seed", None)
                                        dev_str = "cpu"
                                        trainer._gate_candidate_path = cand_path
                                        trainer._gate_baseline_path = base_path
                                        try:
                                            _ctx = mp.get_context("spawn")
                                        except Exception:
                                            _ctx = mp
                                        trainer._gate_proc, trainer._gate_queue = start_async_gate(
                                            cand_path,
                                            base_path,
                                            dict(trainer.config),
                                            games=games,
                                            seed=seed,
                                            device=dev_str,
                                            ctx=_ctx,
                                        )
                                        trainer._last_gate_start_it = int(train_it)
                                        try:
                                            thr = float(trainer.config.get("eval_gate_threshold", 0.6) or 0.6)
                                            if trainer.logger:
                                                import os as _os
                                                trainer.logger.log_text(
                                                    f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={_os.path.basename(cand_path)} base={_os.path.basename(base_path)}"
                                                )
                                            else:
                                                print(f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={cand_path} base={base_path}")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            trainer._save_checkpoint(skip_model_save=True)
                        else:
                            _ = _maybe_gate_before_save()
                            trainer._save_checkpoint()
                            if trainer.keep_prev_model:
                                trainer._snapshot_current_model()
                        last_latest_ckpt_ts = now
                else:
                    if async_gate and gate_enable_flag:
                        if not trainer._should_start_gate(train_it):
                            pass
                        else:
                            try:
                                cand_path = os.path.join(trainer.config.get("checkpoint_dir", "checkpoints"), "_candidate_eval.pt")
                                base_path = trainer.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                                if trainer.model is not None:
                                    try:
                                        trainer.model.save(cand_path, force_sync=True)
                                    except TypeError:
                                        trainer.model.save(cand_path)
                                if getattr(trainer, "_gate_proc", None) is None or (trainer._gate_proc is not None and not trainer._gate_proc.is_alive()):
                                    games = int(trainer.config.get("eval_gate_games", 20) or 20)
                                    seed = trainer.config.get("eval_gate_seed", None)
                                    dev_str = "cpu"
                                    trainer._gate_candidate_path = cand_path
                                    trainer._gate_baseline_path = base_path
                                    try:
                                        _ctx = mp.get_context("spawn")
                                    except Exception:
                                        _ctx = mp
                                    trainer._gate_proc, trainer._gate_queue = start_async_gate(
                                        cand_path,
                                        base_path,
                                        dict(trainer.config),
                                        games=games,
                                        seed=seed,
                                        device=dev_str,
                                        ctx=_ctx,
                                    )
                                    trainer._last_gate_start_it = int(train_it)
                                    try:
                                        thr = float(trainer.config.get("eval_gate_threshold", 0.6) or 0.6)
                                        if trainer.logger:
                                            import os as _os
                                            trainer.logger.log_text(
                                                f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={_os.path.basename(cand_path)} base={_os.path.basename(base_path)}"
                                            )
                                        else:
                                            print(f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={cand_path} base={base_path}")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        trainer._save_checkpoint(skip_model_save=True)
                    else:
                        _ = _maybe_gate_before_save()
                        trainer._save_checkpoint()
                        if trainer.keep_prev_model:
                            trainer._snapshot_current_model()

                try:
                    if bool(trainer.config.get("measure_train_during_training", True)):
                        learner = trainer.agents[trainer.learning_player_id]
                        if hasattr(learner, 'measure_train_step_time'):
                            try:
                                iters = int(trainer.config.get('measure_iterations', 30) or 10)
                                warm = int(trainer.config.get('measure_warmup', 2) or 2)
                                bsz = int(trainer.config.get('batch_size', 256) or 256)
                                learner.measure_train_step_time(batch_size=bsz, iterations=iters, warmup=warm)
                            except Exception:
                                pass
                except Exception:
                    pass
                new_samples_since_train = 0

            now2 = time.time()
            if status_log_sec > 0 and (now2 - last_status_ts) >= status_log_sec:
                last_status_ts = now2
                res_gate = finalize_async_gate_if_ready(
                    getattr(trainer, "_gate_proc", None),
                    getattr(trainer, "_gate_queue", None),
                    cand_path=getattr(trainer, "_gate_candidate_path", None),
                    base_path=getattr(trainer, "_gate_baseline_path", None),
                    cfg=trainer.config,
                    logger=trainer.logger,
                    keep_prev_model=bool(trainer.keep_prev_model),
                    snapshot_cb=(lambda: trainer._snapshot_current_model()) if trainer.keep_prev_model else None,
                )
                if res_gate is not None:
                    try:
                        trainer._last_gate_threshold = float(res_gate.get("threshold", trainer._last_gate_threshold))
                    except Exception:
                        pass
                    if isinstance(res_gate.get("result"), dict):
                        trainer._last_gate_result = dict(res_gate["result"])  # type: ignore
                    trainer._gate_proc = None
                    trainer._gate_queue = None
                    trainer._gate_candidate_path = None
                    trainer._gate_baseline_path = None
                try:
                    replay_size = len(trainer.shared_replay) if trainer.shared_replay is not None else (len(dst_buffer) if dst_buffer is not None and hasattr(dst_buffer, '__len__') else None)
                except Exception:
                    replay_size = None
                io_parts = []
                try:
                    log_dir = trainer.config.get('log_dir', 'logs')
                    ev_path = os.path.join(log_dir, 'events.log')
                    mcts_path = os.path.join(log_dir, 'mcts_samples.jsonl')
                    ckpt_path = trainer.config.get('checkpoint_path', 'checkpoints/policy_value_latest.pt')
                    def _mb(p):
                        try:
                            return os.path.getsize(p) / (1024*1024)
                        except Exception:
                            return None
                    ev_mb = _mb(ev_path)
                    mcts_mb = _mb(mcts_path)
                    ckpt_mb = _mb(ckpt_path)
                    if ev_mb is not None:
                        io_parts.append(f"events:{ev_mb:.1f}MB")
                    if mcts_mb is not None:
                        io_parts.append(f"mcts:{mcts_mb:.1f}MB")
                    if ckpt_mb is not None:
                        io_parts.append(f"ckpt:{ckpt_mb:.1f}MB")
                except Exception:
                    pass
                io_sizes_str = (" io_sizes=" + ",".join(io_parts)) if io_parts else ""
                try:
                    procs_state = ",".join([f"{i}:{'A' if p.is_alive() else 'X'}@{p.pid}" for i,p in enumerate(procs)])
                except Exception:
                    procs_state = ""
                try:
                    gp = getattr(trainer, "_gate_proc", None)
                    if gp is not None:
                        procs_state = (procs_state + ("," if procs_state else "")) + f"gate:{'A' if gp.is_alive() else 'X'}@{gp.pid}"
                except Exception:
                    pass
                msg = f"[status] ep_done={ep_done} train_it={train_it} new_since_train={new_samples_since_train} replay_size={replay_size} workers={workers}{io_sizes_str} procs=[{procs_state}]"
                if trainer.logger:
                    trainer.logger.log_text(msg)
                    if status_log_include_mem:
                        try:
                            sample_cnt = replay_size
                            trainer.logger.log_memory_snapshot(sample_count=sample_cnt, force=False)
                        except Exception:
                            pass
                else:
                    print(msg)
                    if bool(trainer.config.get('measure_select_during_training', True)):
                        learner = trainer.agents[trainer.learning_player_id]
                        if hasattr(learner, 'measure_select_action_time'):
                            env_for_measure = getattr(trainer, 'env', None)
                            if env_for_measure is not None:
                                iters = int(trainer.config.get('measure_select_iterations', 5) or 5)
                                warm = int(trainer.config.get('measure_select_warmup', 1) or 1)
                                learner.measure_select_action_time(env_for_measure, iterations=iters, warmup=warm, training=True)

            if (time.time() - last_worker_rss_check) >= worker_rss_check_interval:
                last_worker_rss_check = time.time()
                try:
                    marked = monitor.check_and_mark_restarts(procs, worker_stats)
                    if marked:
                        monitor.issue_graceful_restart(
                            marked,
                            procs,
                            control_queues,
                            make_control_queue=lambda: ctx.Queue(maxsize=5),
                            spawn_worker=lambda wid, cq: ctx.Process(
                                target=_selfplay_daemon_worker,
                                args=(wid, trainer.config, model_blob_path, sample_queue, event_queue, stop_event, cq, request_q, response_queues[wid]),
                                daemon=True,
                            ),
                            worker_stats=worker_stats,
                        )
                except Exception:
                    pass

        stop_event.set()
    finally:
        if new_samples_since_train > 0:
            for _ in range(max(1, int(updates_per_iter))):
                _ = trainer.agents[0].train_step(batch_size=trainer.config.get("batch_size", 256))
                try:
                    val_every = int(trainer.config.get("val_eval_every_updates", 0) or 0)
                except Exception:
                    val_every = 0
                if trainer.logger and val_every > 0:
                    try:
                        vinfo = trainer.agents[0].validate_step(batch_size=trainer.config.get("val_batch_size") or trainer.config.get("batch_size", 256))
                    except Exception:
                        vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                    if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None):
                        trainer.logger.log_validation(vinfo)
                if trainer.config.get("purge_replay_after_each_update"):
                    try:
                        trainer._save_checkpoint()
                        if trainer.logger:
                            trainer.logger.log_text("[replay] immediate_purge_after_update(final)")
                    except Exception as _e:
                        print(f"[WARN] immediate save after update (final) failed: {_e}")
        try:
            gate_enable = bool(trainer.config.get("eval_gate_enable", False))
        except Exception:
            gate_enable = False
        if gate_enable and trainer.model is not None:
            baseline_model = None
            try:
                from agents.models import PolicyValueNet as _PVN
                base_path = trainer.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                if os.path.exists(base_path):
                    def _resolve_device_str(dev_str: str | None) -> str:
                        if not dev_str or dev_str == "auto":
                            try:
                                import torch as _t
                                return "cuda" if _t.cuda.is_available() else "cpu"
                            except Exception:
                                return "cpu"
                        return dev_str
                    dev_str = _resolve_device_str(trainer.config.get("device", None))
                    baseline_model = _PVN.load(base_path, map_location=dev_str)
                    dev = dev_str
                    if dev:
                        baseline_model.to(dev)  # type: ignore[arg-type]
            except Exception as _e:
                baseline_model = None
            if baseline_model is not None:
                try:
                    from evaluation.gating import evaluate_candidate
                    games = int(trainer.config.get("eval_gate_games", 20) or 20)
                    thr = float(trainer.config.get("eval_gate_threshold", 0.6) or 0.6)
                    seed = trainer.config.get("eval_gate_seed", None)
                    gate_result = evaluate_candidate(trainer.model, baseline_model, trainer.config, games=games, seed=seed)
                    gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                    msg = f"[GATE] (final) win_rate={gate_result.get('win_rate'):.2%} threshold={thr:.0%} result={'ACCEPT' if gated_pass else 'REJECT'}"
                    trainer._last_gate_threshold = float(thr)
                    trainer._last_gate_result = dict(gate_result)
                    if trainer.logger:
                        trainer.logger.log_text(msg)
                    else:
                        print(msg)
                    if not gated_pass:
                        trainer.model = baseline_model
                        lrn = trainer.agents[trainer.learning_player_id]
                        if isinstance(lrn, AlphaZeroAgent):
                            lrn.set_model(trainer.model)
                        if trainer.logger:
                            trainer.logger.log_text("[GATE] (final) reverted to baseline model")
                except Exception:
                    pass
        trainer._save_checkpoint()
        if trainer.keep_prev_model and trainer.model is not None:
            trainer._snapshot_current_model()
        _ = _drain_samples(max_items=None)
        for p in procs:
            p.join(timeout=5)
        trainer._save_checkpoint()
        try:
            if trainer.logger and getattr(trainer.logger, 'csv_summary_only', False):
                trainer.logger.write_csv_summaries()
        except Exception as e:
            print(f"[WARN] write_csv_summaries(train_concurrent) failed: {e}")
    try:
        if trainer.logger:
            trainer.logger.log_text(f"[summary] total_episodes={ep_done} total_train_updates={train_it}")
            if hasattr(trainer.logger, 'flush_buffers'):
                trainer.logger.flush_buffers(force=True)
    except Exception:
        pass
    return {"episodes": ep_done, "train_updates": train_it}


def _self_play_parallel(trainer, num_episodes: int, workers: int):
    start_time = time.time()
    total_done = 0
    last_progress_len = 0

    os.makedirs(trainer.config["checkpoint_dir"], exist_ok=True)
    model_blob_path = trainer.config.get("checkpoint_path", os.path.join(trainer.config["checkpoint_dir"], "policy_value_latest.pt"))
    if (trainer.model is not None) and (not os.path.exists(model_blob_path)):
        try:
            trainer.model.save(model_blob_path)
        except Exception:
            pass

    remaining = int(num_episodes)
    ckpt_interval = int(trainer.ckpt_interval or 0)

    if trainer.minimal_progress and trainer.use_progress_bar and num_episodes > 0:
        bar, _pct = trainer._make_progress_bar(0, num_episodes)
        line = f"[SELFPLAY*] {bar} 0/{num_episodes} (workers={workers})"
        print(line, end='\r', flush=True)
        last_progress_len = len(line)

    while remaining > 0:
        if ckpt_interval > 0:
            to_next = ckpt_interval - (trainer._episodes_total_run % ckpt_interval)
            if to_next <= 0:
                to_next = ckpt_interval
            chunk = min(remaining, to_next)
        else:
            chunk = remaining

        base = chunk // workers
        rem = chunk % workers
        ep_splits = [base + (1 if i < rem else 0) for i in range(workers)]
        tasks = []
        for wid, n_ep in enumerate(ep_splits):
            if n_ep <= 0:
                continue
            tasks.append((wid, n_ep, trainer.config, model_blob_path))

        results = []
        if tasks:
            with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
                for wid, n_ep, cfg, model_path in tasks:
                    results.append(pool.apply_async(_selfplay_worker_entry, (wid, n_ep, cfg, model_path)))
                for res in results:
                    worker_out = res.get()
                    samples = worker_out.get("samples", [])
                    if trainer.shared_replay is None:
                        learner = trainer.agents[trainer.learning_player_id]
                        if isinstance(learner, AlphaZeroAgent):
                            for s in samples:
                                try:
                                    if bool(trainer.config.get('replay_async_enabled', False)) and hasattr(learner.replay_buffer, 'append_async'):
                                        drop_oldest = bool(trainer.config.get('replay_async_drop_oldest', True))
                                        learner.replay_buffer.append_async(s, drop_oldest=drop_oldest)  # type: ignore[attr-defined]
                                    else:
                                        learner.replay_buffer.append(s)  # type: ignore[attr-defined]
                                except Exception:
                                    try:
                                        learner.replay_buffer.append(s)  # type: ignore[attr-defined]
                                    except Exception:
                                        pass
                    else:
                        for s in samples:
                            try:
                                if bool(trainer.config.get('replay_async_enabled', False)) and hasattr(trainer.shared_replay, 'append_async'):
                                    drop_oldest = bool(trainer.config.get('replay_async_drop_oldest', True))
                                    trainer.shared_replay.append_async(s, drop_oldest=drop_oldest)
                                else:
                                    trainer.shared_replay.append(s)
                            except Exception:
                                try:
                                    trainer.shared_replay.append(s)
                                except Exception:
                                    pass
                    total_done += worker_out.get("episodes", 0)
                    if trainer.minimal_progress and trainer.use_progress_bar:
                        bar, _pct = trainer._make_progress_bar(total_done, num_episodes)
                        line = f"[SELFPLAY*] {bar} {total_done}/{num_episodes} (workers={workers})"
                        pad = max(0, last_progress_len - len(line))
                        print(line + ' ' * pad, end='\r' if total_done < num_episodes else '\n', flush=True)
                        last_progress_len = len(line)

        for _ in range(chunk):
            trainer._episodes_total_run += 1
            if trainer.ckpt_interval > 0 and (trainer._episodes_total_run % trainer.ckpt_interval == 0):
                trainer._save_checkpoint(version_tag=f"ep{trainer._episodes_total_run}")
                if trainer.keep_prev_model:
                    trainer._snapshot_current_model()
                if trainer.keep_prev_model and trainer.prev_model_mix_players > 0:
                    if trainer.past_models:
                        trainer._assign_past_models_to_opponents()
                    elif trainer._previous_model is not None:
                        trainer._mix_previous_model_opponents()
                trainer.model_version += 1
                for ag in trainer.agents:
                    if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                        ag.model_version = trainer.model_version
            if trainer.opponent_mix_interval > 0 and trainer.keep_prev_model and trainer.prev_model_mix_players > 0:
                if (trainer._episodes_total_run % trainer.opponent_mix_interval == 0) and trainer.past_models:
                    trainer._assign_past_models_to_opponents()

        remaining -= chunk

    elapsed = time.time() - start_time
    if not trainer.minimal_progress:
        print(f"[SELFPLAY*] finished {num_episodes} episodes in {elapsed:.1f}s using {workers} workers")
    return
