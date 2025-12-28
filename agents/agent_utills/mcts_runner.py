from __future__ import annotations

from typing import Any

from agents.mcts import run_puct_mcts


def run_mcts(agent: Any, env: Any, *, training: bool) -> Any:
    """Run PUCT MCTS for AlphaZeroAgent.

    Responsibility: MCTS orchestration.

    Notes:
    - Still calls back into `agent` for policy/value, legal move generation, determinization hooks,
      and TT support. Determinization extraction is phase-2 per spec.
    - Returns root node with additional _env_copy attribute for debugging.
    """
    import time as _tperf

    _t_clone0 = _tperf.time()
    env_copy = agent._copy_env(env)
    _clone_ms = (_tperf.time() - _t_clone0) * 1000.0
    try:
        agent._last_perf_clone_ms = float(_clone_ms)
    except Exception:
        pass
    try:
        agent._perf_infer_ms_accum = 0.0
    except Exception:
        pass
    
    # デバッグ用: MCTS開始時の環境状態を記録
    _debug_env_copy_revolution = None
    _debug_env_copy_field = None
    try:
        _debug_env_copy_revolution = bool(getattr(getattr(env_copy.game, 'rule_checker', None), 'revolution', False))
        _debug_env_copy_field = [str(c) for c in (getattr(env_copy.game, 'current_field', []) or [])]
    except Exception:
        pass

    # determinization pool lazy start (phase-2 responsibility split later)
    try:
        if agent.config.get('enable_parallel_determinization') and bool(agent.config.get('enable_determinization', True)):
            _det_mode_now = (
                agent.config.get('determinization_mode_train', 'fixed_once') if training
                else agent.config.get('determinization_mode_eval', 'stochastic')
            )
            if _det_mode_now != 'none':
                agent._maybe_start_det_pool(env)
    except Exception:
        pass

    def policy_value_fn(e):
        return agent._policy_value(e)

    def legal_fn(e):
        return agent._get_legal_actions(e)

    TT = None
    if getattr(agent, 'enable_mcts_tt', False):
        if getattr(agent, '_mcts_tt', None) is None:
            agent._mcts_tt = {}
            agent._mcts_tt_tick = 0
        TT = agent._get_tt_view()

    policy_value_batch_fn = agent._get_policy_value_batch_fn(training=training)

    det_enable = bool(agent.config.get('enable_determinization', True))
    root_pid = getattr(env.game, 'turn', 0)
    det_mode = (agent.config.get('determinization_mode_train', 'fixed_once') if training
                else agent.config.get('determinization_mode_eval', 'stochastic'))

    template_assignment = None
    if det_enable and det_mode == 'fixed_once':
        try:
            ok, assign = agent._build_single_determinization(env_copy, env, root_pid, apply_direct=False)
            if ok:
                template_assignment = assign
        except Exception:
            template_assignment = None

    def _apply_assignment_fixed(e_clone, assignment, root_pid_):
        try:
            g_new = e_clone.game
            from game.card import Card
            for pid, cards_str in assignment.get('hands', {}).items():
                if pid == root_pid_:
                    continue
                try:
                    g_new.players[pid].hand = [Card.from_string(s) if hasattr(Card, 'from_string') else Card(s) for s in cards_str]
                except Exception:
                    g_new.players[pid].hand = list(cards_str)
        except Exception:
            pass

    def _determinize(e_clone, original_env, root_pid_):
        if not det_enable or det_mode == 'none':
            return
        if det_mode == 'fixed_once' and template_assignment is not None:
            _apply_assignment_fixed(e_clone, template_assignment, root_pid_)
            return
        if agent.config.get('enable_parallel_determinization', True):
            used = agent._apply_from_det_pool(e_clone, original_env, root_pid_)
            if used:
                return
        try:
            agent._det_stats['pool_fallback_inline'] = agent._det_stats.get('pool_fallback_inline', 0) + 1
        except Exception:
            pass
        agent._inline_determinize(e_clone, original_env, root_pid_)

    add_dirichlet_flag = True if training else bool(agent.config.get('inference_dirichlet', False))

    sims_to_run = int(getattr(agent, 'num_simulations', 0) or agent.config.get('num_simulations', 64) or 64)
    try:
        if training:
            learn_pid = int(agent.config.get('learning_player_id', 0) or 0)
            if int(agent.player_id) != learn_pid:
                opp_abs = int(agent.config.get('opponent_num_simulations', 0) or 0)
                if opp_abs > 0:
                    sims_to_run = max(1, opp_abs)
                else:
                    scale = float(agent.config.get('opponent_sim_scale', 0.125) or 0.125)
                    min_sim = int(agent.config.get('opponent_sim_min', 1) or 1)
                    sims_to_run = max(min_sim, int(round(sims_to_run * max(0.0, scale))))
    except Exception:
        sims_to_run = int(sims_to_run)

    root = run_puct_mcts(
        root_env_copy=env_copy,
        num_simulations=sims_to_run,
        policy_value_fn=policy_value_fn,
        policy_value_batch_fn=policy_value_batch_fn,
        get_legal_actions_fn=legal_fn,
        c_puct=agent.puct_c,
        add_dirichlet=add_dirichlet_flag,
        dirichlet_alpha=agent.dirichlet_alpha,
        dirichlet_epsilon=agent.dirichlet_epsilon,
        root_player_id=root_pid,
        batch_eval_size=getattr(agent, 'mcts_batch_eval_size', 1),
        transposition_table=TT,
        determinize_fn=_determinize if det_enable else None,
        early_stop_enable=bool(agent.config.get('mcts_early_stop_enable', False)),
        early_stop_min_sims=int(agent.config.get('mcts_early_stop_min_sims', 16)),
        early_stop_visit_ratio=float(agent.config.get('mcts_early_stop_visit_ratio', 0.75)),
        early_stop_gap_ratio=float(agent.config.get('mcts_early_stop_gap_ratio', 0.10)),
        early_stop_log_sample_rate=float(agent.config.get('mcts_early_stop_log_sample_rate', 0.0)),
        early_stop_post_min_batch=int(agent.config.get('mcts_early_stop_post_min_batch', 0) or 0),
        early_stop_debug=bool(agent.config.get('mcts_early_stop_debug', False)),
        early_stop_logger=getattr(agent, 'logger', None),
        stage_terminal_debug=bool(agent.config.get('mcts_stage_terminal_debug', False)),
        stage_terminal_log_sample_rate=float(agent.config.get('mcts_stage_terminal_log_sample_rate', 0.1)),
        stage_terminal_logger=getattr(agent, 'logger', None),
    )

    _det_mode_now = (
        agent.config.get('determinization_mode_train', 'fixed_once') if training
        else agent.config.get('determinization_mode_eval', 'stochastic')
    )
    try:
        setattr(root, '_perf_clone_ms', float(_clone_ms))
    except Exception:
        pass
    try:
        setattr(root, '_perf_infer_ms', float(getattr(agent, '_perf_infer_ms_accum', 0.0)))
    except Exception:
        pass
    try:
        setattr(root, '_perf_det_mode', _det_mode_now)
    except Exception:
        pass
    try:
        agent._last_perf_det_mode = str(_det_mode_now)
    except Exception:
        pass
    
    # デバッグ用: MCTS開始時の環境状態をルートノードに保存
    try:
        setattr(root, '_debug_env_copy_revolution', _debug_env_copy_revolution)
        setattr(root, '_debug_env_copy_field', _debug_env_copy_field)
    except Exception:
        pass

    return root
