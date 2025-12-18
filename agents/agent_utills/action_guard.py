from __future__ import annotations

from typing import Any, List, Optional


def _as_sorted_key(action: Any) -> tuple:
    if action is None:
        return (None,)
    try:
        return tuple(sorted(str(c) for c in action))
    except Exception:
        return (None,)


def clamp_action_to_env(
    env: Any,
    action_env: Any,
    *,
    legal_actions_hint: Optional[List[Any]] = None,
) -> Any:
    """Clamp `action_env` so it is guaranteed to be legal for `env.step`.

    This module owns the *ActionGuard* responsibility:
    - Prefer env-derived legal actions (env._generate_legal_actions)
    - Handle pass-only states
    - Match using env._action_key when available
    - Last-resort validation via rule_checker.is_valid_move

    Returns:
        action_env in env-facing format: None or list[str]
    """
    # Normalize tuple->list for consistency
    if isinstance(action_env, tuple):
        action_env = list(action_env)

    # 1) Get "real" legal actions from env if possible
    legal = None
    try:
        g = env.game
        cur = g.players[g.turn]
        hand = cur.hand
        field = (g.current_field or [])[:]
        if hasattr(env, "_generate_legal_actions"):
            legal = env._generate_legal_actions(hand, field)
    except Exception:
        legal = None

    if legal is None:
        legal = legal_actions_hint

    # 2) Pass-only: always pass
    if legal is not None:
        try:
            if all(a is None for a in legal):
                return None
        except Exception:
            pass

    # 3) If action is pass, ensure pass is allowed
    if action_env is None:
        if legal is None:
            return None
        if any(a is None for a in legal):
            return None
        # pass not allowed -> choose first non-pass
        for cand in legal:
            if cand is None:
                continue
            try:
                return [str(c) for c in cand]
            except Exception:
                return cand
        return None

    # 4) Build key mapping (use env._action_key only if it works for BOTH sides)
    use_env_action_key = False
    try:
        if hasattr(env, "_action_key") and legal is not None:
            # find any non-pass legal action as representative
            rep = None
            for cand in legal:
                if cand is not None:
                    rep = cand
                    break
            if rep is not None:
                _ = env._action_key(rep)
                _ = env._action_key(action_env)
                use_env_action_key = True
    except Exception:
        use_env_action_key = False

    def _env_key(act: Any) -> tuple:
        if use_env_action_key:
            try:
                return env._action_key(act)
            except Exception:
                return _as_sorted_key(act)
        return _as_sorted_key(act)

    if legal is not None:
        legal_map = {}
        has_pass = False
        for cand in legal:
            if cand is None:
                has_pass = True
                continue
            k = _env_key(cand)
            if k not in legal_map:
                try:
                    legal_map[k] = [str(c) for c in cand]
                except Exception:
                    legal_map[k] = None

        k_act = _env_key(action_env)
        if k_act in legal_map:
            mapped = legal_map.get(k_act)
            return mapped if mapped is not None else action_env

        # fallback: first non-pass legal
        # デバッグ: フォールバックが発生した場合はログ出力
        import os
        if os.environ.get('DEBUG_ACTION_GUARD'):
            print(f"[ACTION-GUARD-FALLBACK] action_key={k_act} legal_keys={list(legal_map.keys())[:5]}")
        
        for cand in legal:
            if cand is None:
                continue
            try:
                return [str(c) for c in cand]
            except Exception:
                return cand
        return None if has_pass else None

    # 5) Ultimate guard via rule_checker.is_valid_move (best-effort)
    try:
        g = env.game
        rc = getattr(g, "rule_checker", None)
        field = (getattr(g, "current_field", None) or [])[:]
        player = g.players[g.turn]
        # Map strings to Card objects in hand by str()
        if not isinstance(action_env, list):
            try:
                action_list = list(action_env)
            except Exception:
                action_list = []
        else:
            action_list = action_env
        card_objs = []
        used_idx = set()
        ok_cards = True
        for s in action_list:
            found = None
            for i, c in enumerate(getattr(player, "hand", []) or []):
                if i in used_idx:
                    continue
                if str(c) == str(s):
                    found = c
                    used_idx.add(i)
                    break
            if found is None:
                ok_cards = False
                break
            card_objs.append(found)

        if ok_cards and rc is not None and card_objs:
            # empty field special-case
            if not field:
                is_first_turn = bool(getattr(g, "turn_count", 0) == 0 and getattr(g, "last_player", None) is None)
                if is_first_turn or rc.is_valid_move(card_objs, field):
                    return [str(c) for c in card_objs]
            else:
                if len(card_objs) == len(field) and rc.is_valid_move(card_objs, field):
                    return [str(c) for c in card_objs]
    except Exception:
        pass

    return None
