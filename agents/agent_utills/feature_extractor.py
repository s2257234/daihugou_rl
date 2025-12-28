from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


class CardEncoder:
  def card_index(self, card: Any) -> int:
    try:
      if getattr(card, 'is_joker', False):
        return 52
      # String fallback (e.g. '♦7', 'HJ', 'JOKER(♦7)')
      if isinstance(card, str):
        s = card.strip()
        if not s:
          return 52
        if s.upper().startswith('JOKER'):
          return 52
        suit_ch = s[0]
        suit_order = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}
        sid = suit_order.get(suit_ch, 0)
        core = s[1:]
        mp = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
        r = mp.get(core.upper(), None)
        if r is None:
          r = int(core)
        r = max(1, min(13, int(r)))
        return sid * 13 + (r - 1)
      suit_order = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}
      return suit_order.get(getattr(card, 'suit', 'S'), 0) * 13 + (int(getattr(card, 'rank', 1)) - 1)
    except Exception:
      return 52

  def encode_bits_53(self, cards: Any) -> List[float]:
    bits = [0.0] * 53
    for c in list(cards or []):
      try:
        idx = self.card_index(c)
        if 0 <= idx < 53:
          bits[idx] = 1.0
      except Exception:
        pass
    return bits


class HistoryAnalyzer:
  def __init__(self, card_encoder: CardEncoder):
    self._ce = card_encoder

  def analyze(
    self,
    *,
    action_hist: List[Dict[str, Any]],
    opponents: List[int],
    pid: int,
  ) -> Dict[str, Any]:
    opp_discards_map = {i: [0.0] * 53 for i in opponents}
    union_bits = [0.0] * 53

    last_action_player = None
    last_action_card_indices: List[int] = []

    for h in reversed(action_hist):
      try:
        act = h.get('action')
        pid_act = h.get('pid')
        if act is None or act == 'pass':
          continue
        if pid_act == pid:
          continue
        cards_iter = act if isinstance(act, (list, tuple)) else [act]
        for c in cards_iter:
          try:
            idx = self._ce.card_index(c)
            if 0 <= idx < 53:
              last_action_card_indices.append(idx)
          except Exception:
            pass
        last_action_player = pid_act
        break
      except Exception:
        continue

    for h in action_hist:
      try:
        pid_act = h.get('pid')
        act = h.get('action')
        if pid_act in opp_discards_map and act not in (None, 'pass'):
          cards_iter = act if isinstance(act, (list, tuple)) else [act]
          for c in cards_iter:
            try:
              idx = self._ce.card_index(c)
              if 0 <= idx < 53:
                opp_discards_map[pid_act][idx] = 1.0
            except Exception:
              pass
      except Exception:
        continue

    discard_union_indices: List[int] = []
    try:
      for i in opponents:
        mp = opp_discards_map.get(i)
        if not mp:
          continue
        for ci, b in enumerate(mp):
          if b > 0.5:
            union_bits[ci] = 1.0
      discard_union_indices = [ci for ci, b in enumerate(union_bits) if b > 0.5]
    except Exception:
      discard_union_indices = []

    pass_matrix_map = {i: [0.0] * 13 for i in opponents}

    def _rank_str_to_int(s: str):
      try:
        core = s[1:]
        mp = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
        return mp.get(core.upper(), int(core))
      except Exception:
        return None

    for h in action_hist:
      try:
        if h.get('action') != 'pass':
          continue
        pid_pass = h.get('pid')
        if pid_pass not in pass_matrix_map:
          continue
        fb = h.get('field_before') or []
        if not fb:
          continue
        combo_type = h.get('combo_type')
        if combo_type not in (None, 'single', 'pair', 'triple', 'four', 'joker_single'):
          continue
        base_rank = None
        ranks_tmp = []
        for s in fb:
          r = _rank_str_to_int(s)
          if r is not None:
            ranks_tmp.append(r)
        if ranks_tmp:
          base_rank = min(ranks_tmp)
        if base_rank is None:
          continue
        revo_flag = bool(h.get('revo', False))
        if not revo_flag:
          target_ranks = [r for r in range(base_rank + 1, 14)]
        else:
          target_ranks = [r for r in range(1, base_rank)]
        for r in target_ranks:
          if 1 <= r <= 13:
            pass_matrix_map[pid_pass][r - 1] = 1.0
      except Exception:
        continue

    return {
      'opp_discards_map': opp_discards_map,
      'union_bits': union_bits,
      'discard_union_indices': discard_union_indices,
      'pass_matrix_map': pass_matrix_map,
      'last_action_player': last_action_player,
      'last_action_card_indices': last_action_card_indices,
    }


class StateComposer:
  def __init__(self, card_encoder: CardEncoder, history_analyzer: HistoryAnalyzer):
    self._ce = card_encoder
    self._ha = history_analyzer

  def compose(self, agent: Any, env: Any, *, prev_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
      g = env.game
      if g is None:
        num_players = agent.config.get('num_players', 4)
        expected_full_dim = 72 * num_players + 73
        return {
          "turn": 0,
          "full_input": [0.0] * expected_full_dim,
          "full_input_dim": expected_full_dim,
          "full_input_version": 7,
        }

      pid_env = getattr(g, 'turn', None)
      view_pid = getattr(agent, 'player_id', None)
      pid = None
      try:
        if view_pid is not None:
          if hasattr(g, 'players') and 0 <= int(view_pid) < len(getattr(g, 'players', [])):
            pid = int(view_pid)
      except Exception:
        pid = None
      if pid is None:
        pid = pid_env
      if pid is None:
        num_players = agent.config.get('num_players', 4)
        expected_full_dim = 72 * num_players + 73
        return {
          "turn": 0,
          "full_input": [0.0] * expected_full_dim,
          "full_input_dim": expected_full_dim,
          "full_input_version": 7,
        }
      if not hasattr(g, 'players') or g.players is None:
        num_players = agent.config.get('num_players', 4)
        expected_full_dim = 72 * num_players + 73
        return {
          "turn": 0,
          "full_input": [0.0] * expected_full_dim,
          "full_input_dim": expected_full_dim,
          "full_input_version": 7,
        }

      me = g.players[pid]
      if me is None:
        num_players = agent.config.get('num_players', 4)
        expected_full_dim = 72 * num_players + 73
        return {
          "turn": 0,
          "full_input": [0.0] * expected_full_dim,
          "full_input_dim": expected_full_dim,
          "full_input_version": 7,
        }

      rule_checker = getattr(g, "rule_checker", None)
      revo = bool(getattr(rule_checker, "revolution", False)) if rule_checker else False
      use_full = bool(getattr(agent, 'config', {}).get('use_full_features', False))
      base: Dict[str, Any] = {"turn": pid}
      base.update({
        "hand_size": len(getattr(me, 'hand', [])),
        "field_size": len(getattr(g, 'current_field', [])),
        "revolution": revo,
      })
      try:
        base['is_leader'] = 1.0 if (len(getattr(g, 'current_field', []) or []) == 0) else 0.0
      except Exception:
        base['is_leader'] = 0.0

      if not use_full:
        return base

      num_players = len(g.players)
      expected_full_dim = 73 * num_players + 74

      # Self block
      self_bits = self._ce.encode_bits_53(getattr(me, 'hand', []))
      self_pass = 1.0 if (pid < len(getattr(g, 'passed', [])) and getattr(g, 'passed')[pid]) else 0.0
      # 正規化基準: プレイヤーあたりの最大初期枚数 (total_cards / num_players の切り上げ)
      try:
        total_cards = int(getattr(g, 'total_cards', None) or (len(getattr(getattr(g, 'deck', None), 'cards', [])) if getattr(g, 'deck', None) is not None else None) or 53)
      except Exception:
        total_cards = 53
      try:
        # allow agent.config override to use legacy scaling (/14)
        use_old = bool(getattr(agent, 'config', {}).get('use_old_remain_scaling', False))
      except Exception:
        use_old = False
      try:
        if use_old:
          max_initial_hand = 14
        else:
          max_initial_hand = max(1, math.ceil(float(total_cards) / max(1, int(num_players))))
      except Exception:
        max_initial_hand = 14
      # hand_size / max_initial_hand でスケール (0..1)
      try:
        self_remain = float(len(getattr(me, 'hand', []))) / float(max_initial_hand)
      except Exception:
        self_remain = 0.0
      if self_remain < 0.0:
        self_remain = 0.0
      elif self_remain > 1.0:
        self_remain = 1.0
      feat: List[float] = self_bits + [self_pass, self_remain]

      # Opponent summaries (+ optional hand_labels)
      rank_labels = ['daifugo', 'fugo', 'hinmin', 'daihinmin']

      def encode_rank(pl: Any) -> List[float]:
        one = [0.0] * 4
        try:
          rc = getattr(pl, 'rank_class', None)
          if rc is None and hasattr(pl, 'rank'):
            rc = getattr(pl, 'rank')
          if isinstance(rc, str) and rc.lower() in rank_labels:
            one[rank_labels.index(rc.lower())] = 1.0
          elif isinstance(rc, int) and 0 <= rc < 4:
            one[rc] = 1.0
        except Exception:
          pass
        return one

      opponents = [i for i in range(num_players) if i != pid]
      try:
        if bool(getattr(agent, 'config', {}).get('enable_hand_prediction_head', False)) and \
           float(getattr(agent, 'config', {}).get('hand_pred_loss_coef', 0.0) or 0.0) > 0.0:
          hand_labels: List[float] = []
          for i in opponents:
            vec = [0.0] * 53
            for c in getattr(g.players[i], 'hand', []):
              try:
                idx = self._ce.card_index(c)
                if 0 <= idx < 53:
                  vec[idx] = 1.0
              except Exception:
                pass
            hand_labels.extend(vec)
          base['hand_labels'] = hand_labels
          base['hand_labels_dim'] = len(hand_labels)
      except Exception:
        pass

      for i in opponents:
        try:
          # hand_size / max_initial_hand でスケール (0..1)
          opp_rem = float(len(g.players[i].hand)) / float(max_initial_hand)
          if opp_rem < 0.0:
            opp_rem = 0.0
          elif opp_rem > 1.0:
            opp_rem = 1.0
        except Exception:
          opp_rem = 0.0
        feat.append(opp_rem)
        feat.extend(encode_rank(g.players[i]))

      # Field block
      field = getattr(g, 'current_field', []) or []
      combo_type_onehot = [0.0] * 7
      rank_onehot = [0.0] * 13
      field_size = len(field)
      try:
        base['is_leader'] = 1.0 if (field_size == 0) else 0.0
      except Exception:
        base['is_leader'] = 0.0
      if field_size == 0:
        combo_type_onehot[0] = 1.0
      else:
        combo = rule_checker.classify_combo(field) if rule_checker else None
        ctype = combo['type'] if combo else None
        mapping = {'single': 1, 'pair': 2, 'triple': 3, 'four': 4, 'straight': 5, 'joker_single': 6}
        if ctype in mapping:
          combo_type_onehot[mapping[ctype]] = 1.0
        base_rank = None
        if combo:
          if ctype in ('single', 'pair', 'triple', 'four') and combo.get('rank') is not None:
            base_rank = combo['rank']
          elif ctype == 'straight':
            ranks = combo.get('ranks', [])
            base_rank = ranks[0] if ranks else None
        if base_rank is not None and 1 <= base_rank <= 13:
          rank_onehot[base_rank - 1] = 1.0

      revolution_bit = 1.0 if revo else 0.0
      field_size_norm = field_size / 13.0
      feat.extend([revolution_bit] + combo_type_onehot + rank_onehot + [field_size_norm])

      field_bits = self._ce.encode_bits_53(field)
      feat.extend(field_bits)

      # History blocks
      try:
        action_hist = list(getattr(g, '_action_history', []) or [])
      except Exception:
        action_hist = []
      hist = self._ha.analyze(action_hist=action_hist, opponents=opponents, pid=pid)

      for i in opponents:
        feat.extend(hist['opp_discards_map'][i])
      for i in opponents:
        feat.extend(hist['pass_matrix_map'][i])

      # Turn one-hot
      turn_onehot = [0.0] * num_players
      if 0 <= pid < num_players:
        turn_onehot[pid] = 1.0
      feat.extend(turn_onehot)

      # RankRemain(13) + JokerRemain(1)
      try:
        seen = [0.0] * 53
        for ci, b in enumerate(self_bits):
          if b > 0.5:
            seen[ci] = 1.0
        for ci, b in enumerate(field_bits):
          if b > 0.5:
            seen[ci] = 1.0
        for ci, b in enumerate(hist['union_bits']):
          if b > 0.5:
            seen[ci] = 1.0
        rank_remain = []
        for r in range(1, 14):
          occ = 0.0
          rank_base = (r - 1)
          for s in range(4):
            idx = s * 13 + rank_base
            if 0 <= idx < 52 and seen[idx] > 0.5:
              occ += 1.0
          rem = max(0.0, 4.0 - occ) / 4.0
          rank_remain.append(rem)
        joker_occ = 1.0 if seen[52] > 0.5 else 0.0
        joker_remain = max(0.0, 1.0 - joker_occ)
        feat.extend(rank_remain)
        feat.append(joker_remain)
      except Exception:
        feat.extend([0.0] * 14)

      # v8: is_leader + last_actor_onehot
      is_leader_bit = 1.0 if (field_size == 0) else 0.0
      feat.append(is_leader_bit)
      last_actor_onehot = [0.0] * num_players
      try:
        for h in reversed(action_hist):
          act = h.get('action')
          if act is None or act == 'pass':
            continue
          pid_act = h.get('pid')
          if pid_act is not None and 0 <= int(pid_act) < num_players:
            last_actor_onehot[int(pid_act)] = 1.0
          break
      except Exception:
        pass
      if field_size == 0:
        last_actor_onehot = [0.0] * num_players
      feat.extend(last_actor_onehot)

      cur_len = len(feat)
      if cur_len != expected_full_dim:
        if cur_len < expected_full_dim:
          feat.extend([0.0] * (expected_full_dim - cur_len))
        else:
          del feat[expected_full_dim:]
        if not hasattr(agent, '_warned_full_dim_autofix'):
          print(
            f"[WARN] adjusted full_input length from {cur_len} to expected {expected_full_dim} "
            f"(layout: no Belief/PlayHistory + OpponentDiscards + PassMatrix)"
          )
          agent._warned_full_dim_autofix = True

      try:
        target_dim = int(getattr(getattr(agent, 'model', None), 'full_feature_dim', expected_full_dim))
      except Exception:
        target_dim = expected_full_dim
      if target_dim != expected_full_dim:
        cur_len2 = len(feat)
        if cur_len2 < target_dim:
          feat.extend([0.0] * (target_dim - cur_len2))
        elif cur_len2 > target_dim:
          del feat[target_dim:]
        if not hasattr(agent, '_warned_full_dim_model_override'):
          print(f"[INFO] override full_input length to {target_dim} (model.full_feature_dim; was {expected_full_dim})")
          agent._warned_full_dim_model_override = True

      if not isinstance(base, dict):
        try:
          print(f"[WARN][_extract_state] base was {type(base).__name__}; reconstructing dict for pid={pid}")
        except Exception:
          pass
        base = {"turn": pid}
        try:
          base.update({
            "hand_size": len(getattr(me, 'hand', [])),
            "field_size": len(getattr(g, 'current_field', [])),
            "revolution": revo,
          })
          try:
            base['is_leader'] = 1.0 if (len(getattr(g, 'current_field', []) or []) == 0) else 0.0
          except Exception:
            base['is_leader'] = 0.0
        except Exception:
          pass

      base['full_input'] = feat
      base['full_input_dim'] = expected_full_dim
      base['full_input_version'] = 8

      try:
        self_hand_indices: List[int] = []
        for c in getattr(me, 'hand', []):
          try:
            idx = self._ce.card_index(c)
            if 0 <= idx < 53:
              self_hand_indices.append(idx)
          except Exception:
            pass
        base['self_hand_indices'] = self_hand_indices

        field_card_indices: List[int] = []
        for c in field:
          try:
            idx = self._ce.card_index(c)
            if 0 <= idx < 53:
              field_card_indices.append(idx)
          except Exception:
            pass
        base['field_card_indices'] = field_card_indices

        base['discard_union_indices'] = hist['discard_union_indices']
        base['last_action_player'] = hist['last_action_player']
        base['last_action_card_indices'] = hist['last_action_card_indices']
        base['num_players'] = num_players
        base['self_player_id'] = pid
      except Exception:
        pass

      return base
    except Exception as e:
      try:
        import traceback
        print(f"[ERROR][_extract_state] exception: {e}")
        traceback.print_exc()
      except Exception:
        pass
      num_players = agent.config.get('num_players', 4)
      expected_full_dim = 72 * num_players + 73
      return {
        "turn": 0,
        "full_input": [0.0] * expected_full_dim,
        "full_input_dim": expected_full_dim,
        "full_input_version": 7,
      }


class FeatureExtractor:
  def __init__(self):
    self.card_encoder = CardEncoder()
    self.history_analyzer = HistoryAnalyzer(self.card_encoder)
    self.state_composer = StateComposer(self.card_encoder, self.history_analyzer)

  def extract(self, agent: Any, env: Any, *, prev_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return self.state_composer.compose(agent, env, prev_state=prev_state)


_DEFAULT_EXTRACTOR = FeatureExtractor()


def extract_state(agent: Any, env: Any, *, prev_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract state features for AlphaZeroAgent.

    Responsibility: feature construction / encoding.

    Phase-1 scope: move the heavy feature logic out of drl_agent.py while keeping
    the output schema identical.

    Implementation note:
    - This function delegates to the existing agent implementation for now.
      (We keep the *logic* in one place during transition; phase-2 will
      fully relocate the body here once stabilized.)
    """
    return _DEFAULT_EXTRACTOR.extract(agent, env, prev_state=prev_state)
