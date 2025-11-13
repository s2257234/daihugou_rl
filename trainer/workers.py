from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List, Optional

from agents.factory import create_env_and_agents
from agents.drl_agent import AlphaZeroAgent


class SelfplayDaemonWorker:
    def __init__(
        self,
        worker_id: int,
        config: Dict[str, Any],
        model_path: str,
        sample_queue,
        event_queue,
        stop_event,
        control_queue=None,
        request_q=None,
        response_q=None,
    ) -> None:
        self.worker_id = int(worker_id)
        self.config = dict(config)
        self.model_path = model_path
        self.sample_queue = sample_queue
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.control_queue = control_queue
        self.request_q = request_q
        self.response_q = response_q

        self.model = None
        self.agents: List[AlphaZeroAgent] = []
        self.env = None
        self.device = "cpu"

        self.learning_pid = int(self.config.get("learning_player_id", 0) or 0)
        self.pool_dir = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "_pool")
        self.enable_mix = bool(self.config.get("keep_previous_model_opponent", False)) and int(self.config.get("previous_model_mix_players", 0) or 0) > 0
        self.mix_players = int(self.config.get("previous_model_mix_players", 0) or 0)
        self.mix_interval = int(self.config.get("opponent_mix_interval_episodes", 0) or 0)
        self.ep_since_mix = 0

        self.last_mtime = 0.0
        self.check_interval_episodes = int(self.config.get("concurrent_model_check_every", 5) or 5)
        self.ep_since_check = 0
        self.max_steps = int(self.config.get("max_episode_steps", 1000))

    # ---------- setup utilities ----------
    def _setup_threads_env(self):
        try:
            tn = int(self.config.get("torch_num_threads_workers", 0) or 0)
            if tn > 0:
                os.environ["OMP_NUM_THREADS"] = str(tn)
                os.environ["MKL_NUM_THREADS"] = str(tn)
                os.environ["OPENBLAS_NUM_THREADS"] = str(tn)
                os.environ["NUMEXPR_NUM_THREADS"] = str(tn)
        except Exception:
            pass

    def _setup_rng(self):
        try:
            seed = int(self.config.get("seed", 42)) + 10000 * int(self.worker_id)
        except Exception:
            seed = 42 + 10000 * int(self.worker_id)
        random.seed(seed)
        try:
            import numpy as _np  # type: ignore
            _np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass

    def _setup_torch_threads(self):
        try:
            tn = int(self.config.get("torch_num_threads_workers", 0) or 0)
            itn = int(self.config.get("torch_num_interop_threads_workers", 0) or 0)
            if tn > 0 or itn > 0:
                import torch as _t  # type: ignore
                if tn > 0:
                    _t.set_num_threads(tn)
                if itn > 0 and hasattr(_t, "set_num_interop_threads"):
                    try:
                        _t.set_num_interop_threads(itn)
                    except Exception:
                        pass
                try:
                    _intra = _t.get_num_threads()
                    _interop = _t.get_num_interop_threads() if hasattr(_t, "get_num_interop_threads") else -1
                    print(f"[threads][daemon {self.worker_id}] intra={_intra} interop={_interop} OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}")
                except Exception:
                    pass
        except Exception:
            pass

    def _get_mtime(self, p: str) -> float:
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0

    def _list_pool_files(self) -> List[str]:
        try:
            if not os.path.isdir(self.pool_dir):
                return []
            files = [os.path.join(self.pool_dir, f) for f in os.listdir(self.pool_dir) if f.endswith('.pt')]
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files
        except Exception:
            return []

    def _assign_opponents_from_pool(self):
        if not self.enable_mix or self.mix_players <= 0:
            return
        try:
            from agents.models import PolicyValueNet as _PVN  # lazy import
            # 学習プレイヤーは常に最新を保持
            try:
                ag_learner = self.agents[self.learning_pid]
                if hasattr(ag_learner, 'set_model'):
                    ag_learner.set_model(self.model)
            except Exception:
                pass
            pool_files = self._list_pool_files()
            if not pool_files:
                return
            cand = [i for i in range(len(self.agents)) if i != self.learning_pid]
            if not cand:
                return
            random.shuffle(cand)
            selected = cand[: self.mix_players]
            # 一旦全員を最新に戻す
            for i in cand:
                try:
                    if hasattr(self.agents[i], 'set_model'):
                        self.agents[i].set_model(self.model)
                except Exception:
                    pass
            # 選出相手へ過去モデル割当
            for idx in selected:
                try:
                    path = random.choice(pool_files)
                    opp_model = _PVN.load(path, map_location=self.device)
                    if hasattr(self.agents[idx], 'set_model'):
                        self.agents[idx].set_model(opp_model)
                except Exception:
                    pass
        except Exception:
            pass

    def _flush_local_buffers_to_queue(self):
        try:
            for az in self.agents:
                buf = getattr(az, 'replay_buffer', [] if False else [])
                if buf is None:
                    episode_samples = getattr(az, '_episode_confirmed_samples', [])
                    for s in episode_samples:
                        if isinstance(s, dict) and s.get('value') is not None:
                            try:
                                self.sample_queue.put(s, block=True)
                            except Exception:
                                break
                    if hasattr(az, '_episode_confirmed_samples'):
                        try:
                            az._episode_confirmed_samples.clear()
                        except Exception:
                            pass
                    continue
                if not buf:
                    continue
                for s in list(buf):
                    if isinstance(s, dict) and s.get('value') is not None:
                        try:
                            self.sample_queue.put(s, block=True)
                        except Exception:
                            break
                try:
                    buf.clear()
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- main lifecycle ----------
    def setup(self):
        self._setup_threads_env()
        self._setup_rng()
        self._setup_torch_threads()
        # create model/agents/env via factory
        worker_zero_buffer = bool(self.config.get("worker_zero_buffer", False))
        bundle = create_env_and_agents(
            self.config,
            context="worker",
            model_path=self.model_path,
            worker_id=self.worker_id,
            request_q=self.request_q,
            response_q=self.response_q,
            worker_zero_buffer=worker_zero_buffer,
        )
        self.model = bundle.model
        self.agents = bundle.agents
        self.env = bundle.env
        self.device = bundle.device
        # 初期対戦相手の割当（プールがあれば活用）
        if self.enable_mix and self.mix_players > 0:
            self._assign_opponents_from_pool()
        self.last_mtime = self._get_mtime(self.model_path)

    def maybe_reload_model(self):
        self.ep_since_check += 1
        if self.ep_since_check < self.check_interval_episodes:
            return
        self.ep_since_check = 0
        cur_mtime = self._get_mtime(self.model_path)
        if cur_mtime <= self.last_mtime:
            return
        try:
            from agents.models import PolicyValueNet as _PVN
            new_model = _PVN.load(self.model_path, map_location=self.device)
            self.model = new_model
            # 学習プレイヤーのみ最新適用
            try:
                ag_learner = self.agents[self.learning_pid]
                if hasattr(ag_learner, 'set_model'):
                    ag_learner.set_model(self.model)
            except Exception:
                pass
        except Exception:
            pass
        self.last_mtime = cur_mtime

    def play_one_episode(self) -> int:
        if hasattr(self.env, 'reset'):
            self.env.reset()
        for ag in self.agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        t_game_start = time.time()
        step_count = 0
        prev_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
        while not getattr(self.env.game, "done", False):
            if step_count >= self.max_steps:
                break
            cur_pid = self.env.game.turn
            ag = self.agents[cur_pid]
            action = ag.select_action(self.env, training=True)
            try:
                self.env.step(external_action=action)
            except TypeError:
                self.env.step(action)
            step_count += 1
            if (step_count % 200) == 0:
                try:
                    self.event_queue.put(("hb", (self.worker_id, step_count)), block=False)
                except Exception:
                    pass
            current_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for az in self.agents:
                        was_active = az.player_id not in prev_rankings
                        az.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                prev_rankings = current_rankings
        for az in self.agents:
            az.flush_unfinished_phase()
            az.finalize_game()
        # 送信
        self._flush_local_buffers_to_queue()
        try:
            self.event_queue.put(("ep_done", 1), block=True)
        except Exception:
            pass
        # パフォーマンスイベント（簡略）
        try:
            payload = {
                'wid': int(self.worker_id),
                'ep': 0,  # 呼び出し側で通番管理する場合は上書き
                'moves': int(step_count),
                't_game_sec': float(time.time() - t_game_start),
            }
            self.event_queue.put(("perf_ep", payload), block=False)
        except Exception:
            pass
        return step_count

    def loop(self):
        worker_ep_count = 0
        while not self.stop_event.is_set():
            # control
            if self.control_queue is not None:
                try:
                    ctrl = self.control_queue.get_nowait()
                except Exception:
                    ctrl = None
                if ctrl == 'FLUSH_AND_EXIT':
                    self._flush_local_buffers_to_queue()
                    try:
                        self.event_queue.put(('worker_exit', self.worker_id), block=False)
                    except Exception:
                        pass
                    return
            # model reload check
            self.maybe_reload_model()
            # play
            _ = self.play_one_episode()
            worker_ep_count += 1
            # 過去モデルミックスの周期適用
            if self.enable_mix and self.mix_interval > 0:
                self.ep_since_mix += 1
                if self.ep_since_mix >= self.mix_interval:
                    self._assign_opponents_from_pool()
                    self.ep_since_mix = 0
        try:
            self.event_queue.put(("worker_exit", self.worker_id), block=False)
        except Exception:
            pass

    def run(self):
        self.setup()
        self.loop()


def selfplay_daemon_worker_entry(
    worker_id: int,
    config: Dict[str, Any],
    model_path: str,
    sample_queue,
    event_queue,
    stop_event,
    control_queue=None,
    request_q=None,
    response_q=None,
):
    SelfplayDaemonWorker(
        worker_id=worker_id,
        config=config,
        model_path=model_path,
        sample_queue=sample_queue,
        event_queue=event_queue,
        stop_event=stop_event,
        control_queue=control_queue,
        request_q=request_q,
        response_q=response_q,
    ).run()


__all__ = ["SelfplayDaemonWorker", "selfplay_daemon_worker_entry"]
