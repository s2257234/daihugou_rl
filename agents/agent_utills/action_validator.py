"""ACTION-MISMATCH 検証・デバッグ用ユーティリティ

このモジュールは、MCTSが選択した行動が実環境で違法になる原因を特定するための
検証ロジックを提供します。

主な機能:
- 環境コピーと実環境の状態比較
- 革命状態の検証
- 合法手リストの比較
- 詳細なデバッグログ出力
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def _get_revolution_state(env: Any) -> bool:
    """環境から革命状態を取得"""
    try:
        return bool(getattr(getattr(env.game, 'rule_checker', None), 'revolution', False))
    except Exception:
        return False


def _get_field(env: Any) -> List[str]:
    """環境から場のカードを取得"""
    try:
        return [str(c) for c in (getattr(env.game, 'current_field', []) or [])]
    except Exception:
        return []


def _get_hand(env: Any, player_id: int) -> List[str]:
    """環境から指定プレイヤーの手札を取得"""
    try:
        return [str(c) for c in env.game.players[player_id].hand]
    except Exception:
        return []


def _get_turn(env: Any) -> int:
    """環境から現在手番を取得"""
    try:
        return int(getattr(env.game, 'turn', 0))
    except Exception:
        return 0


def _action_to_key(action: Any) -> Optional[Tuple]:
    """行動をハッシュ可能なキーに変換"""
    if action is None or action == "pass":
        return ("pass",)
    try:
        if isinstance(action, (list, tuple)):
            return tuple(sorted(str(c) for c in action))
        return (str(action),)
    except Exception:
        return None


def compare_env_states(env_real: Any, env_copy: Any) -> Dict[str, Any]:
    """実環境とコピー環境の状態を比較し、差分を返す
    
    Returns:
        dict: {
            'revolution_match': bool,
            'field_match': bool,
            'turn_match': bool,
            'real_revolution': bool,
            'copy_revolution': bool,
            'real_field': List[str],
            'copy_field': List[str],
            'real_turn': int,
            'copy_turn': int,
        }
    """
    real_revo = _get_revolution_state(env_real)
    copy_revo = _get_revolution_state(env_copy)
    real_field = _get_field(env_real)
    copy_field = _get_field(env_copy)
    real_turn = _get_turn(env_real)
    copy_turn = _get_turn(env_copy)
    
    return {
        'revolution_match': (real_revo == copy_revo),
        'field_match': (real_field == copy_field),
        'turn_match': (real_turn == copy_turn),
        'real_revolution': real_revo,
        'copy_revolution': copy_revo,
        'real_field': real_field,
        'copy_field': copy_field,
        'real_turn': real_turn,
        'copy_turn': copy_turn,
    }


def compare_legal_actions(legal_mcts: List[Any], legal_real: List[Any]) -> Dict[str, Any]:
    """MCTS環境と実環境の合法手リストを比較
    
    Returns:
        dict: {
            'match': bool,
            'only_in_mcts': List[Tuple],
            'only_in_real': List[Tuple],
            'mcts_count': int,
            'real_count': int,
        }
    """
    mcts_keys = set()
    for act in legal_mcts:
        k = _action_to_key(act)
        if k is not None:
            mcts_keys.add(k)
    
    real_keys = set()
    for act in legal_real:
        k = _action_to_key(act)
        if k is not None:
            real_keys.add(k)
    
    only_in_mcts = mcts_keys - real_keys
    only_in_real = real_keys - mcts_keys
    
    return {
        'match': (len(only_in_mcts) == 0 and len(only_in_real) == 0),
        'only_in_mcts': list(only_in_mcts),
        'only_in_real': list(only_in_real),
        'mcts_count': len(mcts_keys),
        'real_count': len(real_keys),
    }


def validate_mcts_action(
    action: Any,
    env_real: Any,
    env_copy: Any,
    legal_mcts: List[Any],
    legal_real: List[Any],
    player_id: int,
) -> Dict[str, Any]:
    """MCTSが選択した行動の妥当性を検証し、詳細な診断情報を返す
    
    Returns:
        dict: {
            'is_valid': bool,
            'action_key': Tuple,
            'in_mcts_legal': bool,
            'in_real_legal': bool,
            'env_state_diff': Dict,
            'legal_diff': Dict,
            'diagnosis': str,
        }
    """
    action_key = _action_to_key(action)
    
    mcts_keys = set(_action_to_key(a) for a in legal_mcts if _action_to_key(a) is not None)
    real_keys = set(_action_to_key(a) for a in legal_real if _action_to_key(a) is not None)
    
    in_mcts = action_key in mcts_keys if action_key else False
    in_real = action_key in real_keys if action_key else False
    
    env_diff = compare_env_states(env_real, env_copy)
    legal_diff = compare_legal_actions(legal_mcts, legal_real)
    
    # 診断
    diagnosis = "UNKNOWN"
    if in_mcts and in_real:
        diagnosis = "OK: Action is legal in both environments"
    elif in_mcts and not in_real:
        if not env_diff['revolution_match']:
            diagnosis = (
                f"REVOLUTION_MISMATCH: MCTS used revolution={env_diff['copy_revolution']}, "
                f"but real env has revolution={env_diff['real_revolution']}"
            )
        elif not env_diff['field_match']:
            diagnosis = (
                f"FIELD_MISMATCH: MCTS field={env_diff['copy_field']}, "
                f"but real env field={env_diff['real_field']}"
            )
        elif not env_diff['turn_match']:
            diagnosis = (
                f"TURN_MISMATCH: MCTS turn={env_diff['copy_turn']}, "
                f"but real env turn={env_diff['real_turn']}"
            )
        else:
            # 手札の差分チェック
            real_hand = _get_hand(env_real, player_id)
            copy_hand = _get_hand(env_copy, player_id)
            if set(real_hand) != set(copy_hand):
                diagnosis = (
                    f"HAND_MISMATCH: MCTS hand size={len(copy_hand)}, "
                    f"real hand size={len(real_hand)}"
                )
            else:
                diagnosis = "LEGAL_GENERATION_DIFF: Same state but different legal actions"
    elif not in_mcts:
        diagnosis = "NOT_IN_MCTS: Action was not in MCTS legal actions (strange)"
    
    return {
        'is_valid': in_real,
        'action_key': action_key,
        'in_mcts_legal': in_mcts,
        'in_real_legal': in_real,
        'env_state_diff': env_diff,
        'legal_diff': legal_diff,
        'diagnosis': diagnosis,
    }


def log_action_mismatch_details(
    action: Any,
    env_real: Any,
    env_copy: Any,
    legal_mcts: List[Any],
    legal_real: List[Any],
    player_id: int,
    log_fn=None,
):
    """ACTION-MISMATCHの詳細をログ出力
    
    Args:
        log_fn: ログ出力関数。Noneの場合はlogger.warningを使用
    """
    result = validate_mcts_action(
        action, env_real, env_copy, legal_mcts, legal_real, player_id
    )
    
    if log_fn is None:
        log_fn = logger.warning
    
    if not result['is_valid']:
        log_fn(
            f"[ACTION-MISMATCH-DETAIL] pid={player_id} "
            f"action={result['action_key']} "
            f"diagnosis={result['diagnosis']} "
            f"revo_real={result['env_state_diff']['real_revolution']} "
            f"revo_copy={result['env_state_diff']['copy_revolution']} "
            f"field={result['env_state_diff']['real_field']}"
        )
    
    return result
