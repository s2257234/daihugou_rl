from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List, Optional

from agents.factory import create_env_and_agents
from agents.drl_agent import AlphaZeroAgent
from agents.rule_based_agent import RuleBasedAgent


VALUE_U8_NONE = 0xFF  # Legacy constant for backwards compatibility


def _has_value_label(sample: Dict[str, Any]) -> bool:
    """Returns True when the sample includes a valid value label.
    
    Primary check is for raw float 'value' field.
    For backwards compatibility, also accepts legacy 'value_u8' field.
    """
    if not isinstance(sample, dict):
        return False
    # Primary: raw float value
    if sample.get('value') is not None:
        return True
    # Backwards compatibility: check legacy value_u8
    value_u8 = sample.get('value_u8')
    return isinstance(value_u8, int) and 0 <= value_u8 < VALUE_U8_NONE


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
        # ロガー（Trainer からローカル呼び出し時に注入される場合あり）
        self.logger = None
        # ローカル通番 (metrics interval 判定用)
        self._local_episode_counter = 0
        # 累積統計
        self._cum_learning_wins = 0
        self._cum_episodes = 0

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
            files = [os.path.join(self.pool_dir, f) for f in os.listdir(self.pool_dir) if f.endswith('.pt') and '.tmp.' not in f]
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
            # Debug hook: when enabled, probe agent._extract_state(self.env)
            # and emit a concise log about the returned full_input/self indices.
            do_debug = bool(self.config.get('debug_log_extract_state', False))
            if do_debug:
                try:
                    # Env-level summary: per-player hand counts (avoid heavy prints)
                    try:
                        g = getattr(self.env, 'game', None)
                        if g is not None:
                            hand_counts = []
                            hand_samples = []
                            for p in getattr(g, 'players', []) or []:
                                try:
                                    h = getattr(p, 'hand', []) or []
                                    hand_counts.append(len(h))
                                    # sample up to 3 cards as str
                                    sample_cards = [str(c) for c in (h[:3] if len(h) > 0 else [])]
                                    hand_samples.append(sample_cards)
                                except Exception:
                                    hand_counts.append(None)
                                    hand_samples.append([])
                            print(f"[DEBUG-env] worker={self.worker_id} player_hand_counts={hand_counts} player_hand_samples={hand_samples}")
                    except Exception:
                        pass
                    for ia, az in enumerate(self.agents):
                        if hasattr(az, '_extract_state') and self.env is not None:
                            try:
                                probe = az._extract_state(self.env)
                                fi = probe.get('full_input') if isinstance(probe, dict) else None
                                fc = probe.get('full_compact') if isinstance(probe, dict) else None
                                sidx = probe.get('self_hand_indices') if isinstance(probe, dict) else None
                                # compute quick summaries
                                fi_sum = None
                                fi_len = None
                                if fi is not None:
                                    try:
                                        fi_sum = float(sum(fi))
                                        fi_len = len(fi)
                                    except Exception:
                                        fi_sum = None
                                # full_compact packed bits quick inspect
                                packed_info = None
                                try:
                                    if isinstance(fc, dict) and isinstance(fc.get('packed_bits'), (bytes, bytearray)):
                                        pb = fc.get('packed_bits')
                                        packed_info = f"len={len(pb)} first_bytes={[b for b in pb[:4]]}"
                                except Exception:
                                    packed_info = None
                                turn_v = probe.get('turn') if isinstance(probe, dict) else None
                                hand_size_v = probe.get('hand_size') if isinstance(probe, dict) else None
                                self_pid_v = probe.get('self_player_id') if isinstance(probe, dict) else None
                                print(f"[DEBUG-extract_state] worker={self.worker_id} agent={ia} view_turn={turn_v} view_self_player_id={self_pid_v} hand_size={hand_size_v} full_input_len={fi_len} full_input_sum={fi_sum} full_compact={packed_info} self_hand_indices={'present' if sidx else 'absent'}")
                            except Exception as _e:
                                print(f"[DEBUG-extract_state] worker={self.worker_id} agent={ia} probe failed: {_e}")
                except Exception:
                    pass
            def _try_put(s) -> bool:
                try:
                    # avoid blocking indefinitely when parent isn't draining
                    self.sample_queue.put(s, block=False)
                    return True
                except Exception:
                    return False
            for az in self.agents:
                buf = getattr(az, 'replay_buffer', [] if False else [])
                if buf is None:
                    episode_samples = getattr(az, '_episode_confirmed_samples', [])
                    for s in episode_samples:
                        if _has_value_label(s):
                            if not _try_put(s):
                                # queue full: stop flushing to avoid deadlock
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
                    if _has_value_label(s):
                        if not _try_put(s):
                            # queue full: stop flushing to avoid deadlock
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

    def play_one_episode(self, *, flush: bool = True) -> int:
        # 通番更新
        self._local_episode_counter += 1
        ep_index = self._local_episode_counter - 1
        
        # エピソードごとにシードを再設定（カードシャッフルの多様性を確保）
        try:
            base_seed = int(self.config.get("seed", 42))
            episode_seed = base_seed + 10000 * int(self.worker_id) + ep_index
            random.seed(episode_seed)
            try:
                import numpy as _np
                _np.random.seed(episode_seed % (2**32 - 1))
            except Exception:
                pass
        except Exception:
            pass  # シード設定に失敗しても続行
        
        if hasattr(self.env, 'reset'):
            self.env.reset()
        for ag in self.agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        t_game_start = time.time()
        # エピソード開始時点でのアリーナ割当カウント
        try:
            arena_latest_count = int(sum(1 for ag in self.agents if bool(getattr(ag, 'is_using_latest_model', False))))
            arena_rule_count = int(sum(1 for ag in self.agents if isinstance(ag, RuleBasedAgent)))
            arena_past_count = max(0, len(self.agents) - arena_latest_count - arena_rule_count)
        except Exception:
            arena_latest_count = arena_rule_count = arena_past_count = 0
        step_count = 0
        prev_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
        while not getattr(self.env.game, "done", False):
            if step_count >= self.max_steps:
                break
            cur_pid = self.env.game.turn
            ag = self.agents[cur_pid]
            action = ag.select_action(self.env, training=True)
            try:
                # Use return_info=True so the environment can report the actually played action
                # (useful when env clamps illegal external_action).
                _, _, _, info = self.env.step(return_info=True, external_action=action)
                # Strict mode: do not allow env-side correction of agent actions.
                try:
                    strict = bool(self.config.get('strict_no_action_correction', True))
                except Exception:
                    strict = True
                if strict and isinstance(info, dict) and info.get('corrected_external_action'):
                    raise RuntimeError(
                        f"external_action was corrected by env (wid={self.worker_id} pid={cur_pid}) "
                        f"before={info.get('external_action_before')} after={info.get('played_cards')}"
                    )
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
        # メトリクス算出・ロギング
        try:
            rankings = list(getattr(self.env.game, 'rankings', []))
            avg_rank = None
            first_rate = 0.0
            if rankings:
                try:
                    if self.learning_pid in rankings:
                        avg_rank = rankings.index(self.learning_pid) + 1
                    first_rate = 1.0 if rankings and rankings[0] == self.learning_pid else 0.0
                except Exception:
                    pass
            learner_agent = self.agents[self.learning_pid]
            cum_phase_rate = None
            if isinstance(learner_agent, AlphaZeroAgent) and getattr(learner_agent, 'total_value_samples', 0) > 0:
                try:
                    cum_phase_rate = learner_agent.total_positive / max(1, learner_agent.total_value_samples)
                except Exception:
                    cum_phase_rate = None
            # フェーズ勝率 (episode 内集計) と精度
            phase_wins = getattr(learner_agent, 'episode_phase_correct', None)
            phase_attempts = getattr(learner_agent, 'episode_phase_total', None)
            phase_win_rate = None
            if isinstance(phase_wins, int) and isinstance(phase_attempts, int) and phase_attempts > 0:
                phase_win_rate = phase_wins / phase_attempts
            phase_acc = phase_win_rate  # 互換: 正答率を win_rate と同一扱い
            ep_metrics: Dict[str, Any] = {
                'avg_rank': avg_rank,
                'first_rate': first_rate,
                'episode_len': step_count,
                'phase_acc': phase_acc,
                'phase_win_rate': phase_win_rate,
                'phase_wins': phase_wins,
                'phase_attempts': phase_attempts,
                'cum_phase_win_rate': cum_phase_rate,
                'avg_moves_per_game': step_count,
            }
            # --- しばり（shibari）統計をエピソード指標に追加 ---
            try:
                ep_metrics['shibari_triggered'] = int(getattr(self.env.game, 'shibari_triggered_count', 0) or 0)
                ep_metrics['shibari_passes'] = int(getattr(self.env.game, 'shibari_pass_count', 0) or 0)
                ep_metrics['shibari_turns'] = int(getattr(self.env.game, 'turn_count_in_shibari', 0) or 0)
            except Exception:
                ep_metrics['shibari_triggered'] = 0
                ep_metrics['shibari_passes'] = 0
                ep_metrics['shibari_turns'] = 0
            # エピソード単位のアリーナ割当カウントと学習プレイヤー勝敗
            try:
                learning_win = 1 if (rankings and rankings[0] == self.learning_pid) else 0
            except Exception:
                learning_win = 0
            try:
                # 累積更新
                self._cum_episodes = int(getattr(self, '_cum_episodes', 0)) + 1
                self._cum_learning_wins = int(getattr(self, '_cum_learning_wins', 0)) + int(learning_win)
                cum_learning_win_rate = float(self._cum_learning_wins) / max(1, int(self._cum_episodes))
            except Exception:
                cum_learning_win_rate = None
            ep_metrics.update({
                'arena_latest_count': arena_latest_count,
                'arena_past_count': arena_past_count,
                'arena_rule_count': arena_rule_count,
                'learning_player_win': learning_win,
                'cum_learning_win_rate': cum_learning_win_rate,
            })
            # 簡易テキストログにも出力
            try:
                if self.logger:
                    self.logger.log_text(f"[arena] ep={ep_index+1} latest={arena_latest_count} past={arena_past_count} rule={arena_rule_count} learning_win={learning_win} cum_win_rate={cum_learning_win_rate}")
            except Exception:
                pass
            # 追加計測: forward / game time
            if bool(self.config.get('measure_forward_time', False)):
                try:
                    if hasattr(learner_agent, '_perf_infer_calls') and learner_agent._perf_infer_calls > 0:
                        avg_fwd_ms = learner_agent._perf_infer_ms_accum / max(1, learner_agent._perf_infer_calls)
                        ep_metrics['t_forward_avg_ms'] = float(avg_fwd_ms)
                        if getattr(learner_agent, '_forward_time_ms_ema', None) is not None:
                            ep_metrics['t_forward_ema_ms'] = float(learner_agent._forward_time_ms_ema)
                except Exception:
                    pass
            if bool(self.config.get('measure_game_time', False)):
                try:
                    ep_metrics['t_game_sec'] = float(time.time() - t_game_start)
                except Exception:
                    pass
            # measure_log_every_episodes 間隔で簡易ログ (forward+game 両方有効時)
            if self.logger and bool(self.config.get('measure_forward_time', False)) and bool(self.config.get('measure_game_time', False)):
                try:
                    interval = int(self.config.get('measure_log_every_episodes', 20) or 20)
                except Exception:
                    interval = 20
                if interval > 0 and ((ep_index + 1) % interval == 0):
                    try:
                        self.logger.log_text(f"[perf-ep] ep={ep_index+1} t_forward_ms={ep_metrics.get('t_forward_avg_ms')} t_game_sec={ep_metrics.get('t_game_sec')} moves={step_count}")
                    except Exception:
                        pass
            try:
                if self.logger:
                    try:
                        self.logger.log_episode(ep_metrics)
                    except Exception:
                        pass
                else:
                    try:
                        # Send episode metrics to parent process for central logging
                        self.event_queue.put(("ep_metrics", ep_metrics), block=False)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        # 送信（通常は flush するが、呼び出し側で制御できるようにフラグ化）
        if flush:
            self._flush_local_buffers_to_queue()
            try:
                # do not block on event queue either
                self.event_queue.put(("ep_done", 1), block=False)
            except Exception:
                pass
        # パフォーマンスイベント（簡略）
        try:
            payload = {
                'wid': int(self.worker_id),
                'ep': int(ep_index),
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
    config: Dict[str, Any] | str,
    model_path: str,
    sample_queue,
    event_queue,
    stop_event,
    control_queue=None,
    request_q=None,
    response_q=None,
):
    # 受け取った設定が JSON 文字列なら辞書へ復元（Windows spawn のpickle安定化）
    if isinstance(config, str):
        try:
            import json as _json
            config = _json.loads(config)
        except Exception:
            config = {}
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
