from __future__ import annotations

from typing import Any


class AlphaZeroTTView(dict):
    """Thin LRU-ish TT view over an agent-owned dict.

    Internal representation expected on agent:
      - agent._mcts_tt: { (model_version, key): (result, tick) }
      - agent._mcts_tt_tick: int
      - agent.mcts_tt_capacity: int
      - agent.tt_hits / agent.tt_misses: counters (best-effort)

    This wrapper is dict-like for the subset used by MCTS.
    """

    def __init__(self, agent: Any):
        self._agent = agent

    def _mk(self, k: Any):
        return (getattr(self._agent, 'model_version', 0), k)

    def __contains__(self, k: Any) -> bool:
        try:
            store = self._agent._mcts_tt
            found = self._mk(k) in store
            try:
                if found:
                    self._agent.tt_hits += 1
                else:
                    self._agent.tt_misses += 1
            except Exception:
                pass
            return found
        except RecursionError:
            try:
                self._agent.enable_mcts_tt = False
                self._agent._mcts_tt = {}
            except Exception:
                pass
            return False

    def __getitem__(self, k: Any):
        try:
            store = self._agent._mcts_tt
            key = self._mk(k)
            res, _tick = store[key]
            self._agent._mcts_tt_tick += 1
            store[key] = (res, self._agent._mcts_tt_tick)
            return res
        except RecursionError:
            try:
                self._agent.enable_mcts_tt = False
                self._agent._mcts_tt = {}
            except Exception:
                pass
            raise KeyError(k)

    def __setitem__(self, k: Any, v: Any) -> None:
        try:
            agent = self._agent
            store = agent._mcts_tt
            agent._mcts_tt_tick += 1
            store[self._mk(k)] = (v, agent._mcts_tt_tick)

            cap = max(1, int(getattr(agent, 'mcts_tt_capacity', 1) or 1))
            if len(store) > cap:
                oldest_k = min(store.items(), key=lambda kv: kv[1][1])[0]
                store.pop(oldest_k, None)
        except RecursionError:
            try:
                agent = self._agent
                agent.enable_mcts_tt = False
                agent._mcts_tt = {}
            except Exception:
                pass
            return


# ---------------- Determinization (Imperfect Information) ----------------

def maybe_start_det_pool(agent: Any, env: Any) -> None:
    if getattr(agent, '_det_pool', None) is not None:
        return
    try:
        import threading
        from collections import deque as _dq

        agent._det_pool = _dq(maxlen=agent._det_cfg['capacity'])
        agent._det_pool_lock = threading.Lock()
        agent._det_stop_event = threading.Event()

        # ワーカが参照するベース環境を確実にセット
        try:
            if env is not None:
                agent.env_ref = env
        except Exception:
            pass

        g = env.game
        agent._det_root_signature = {
            'num_players': len(getattr(g, 'players', [])),
            'revo': bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False)),
            'root_turn': getattr(g, 'turn', 0),
        }
        agent._det_pool_thread = threading.Thread(target=agent._determinization_worker, daemon=True)
        agent._det_pool_thread.start()
        # プールが必要数貯まるまで明示的に待機（安全策）
        try:
            import time as _time
            warmup_count = int(agent.config.get('det_pool_warmup_count', 1) or 1)
            warmup_timeout = float(agent.config.get('det_pool_warmup_timeout_sec', 0.5) or 0.5)
            if warmup_count > 0:
                _start = _time.time()
                while True:
                    if getattr(agent, '_det_stop_event', None) is None or agent._det_stop_event.is_set():
                        break
                    with agent._det_pool_lock:
                        cur = len(agent._det_pool) if agent._det_pool is not None else 0
                    if cur >= warmup_count:
                        break
                    if warmup_timeout > 0 and (_time.time() - _start) >= warmup_timeout:
                        break
                    _time.sleep(0.001)
        except Exception:
            pass
        try:
            if getattr(agent, 'logger', None):
                agent.logger.log_text(
                    f"[det-pool-init] cap={agent._det_cfg['capacity']} refill={agent._det_cfg['refill_ratio']} sig={agent._det_root_signature}"
                )
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[DetPool][ERROR] 起動失敗: {type(e).__name__}: {e}")
        except Exception:
            pass
        agent._det_pool = None


def determinization_worker(agent: Any) -> None:
    import time

    while getattr(agent, '_det_stop_event', None) is not None and not agent._det_stop_event.is_set():
        try:
            with agent._det_pool_lock:
                cur_len = len(agent._det_pool)
                cap = agent._det_cfg['capacity']
            refill_threshold = int(agent._det_cfg['capacity'] * agent._det_cfg['refill_ratio'])
            if cur_len >= cap or (cur_len > refill_threshold and cur_len > 0):
                time.sleep(0.002)
                continue

            base_env = getattr(agent, 'env_ref', None)
            if base_env is None:
                time.sleep(0.01)
                continue

            env_clone = agent._copy_env(base_env)
            ok, assignment = build_single_determinization(agent, env_clone, base_env, getattr(base_env.game, 'turn', 0))
            if ok and assignment:
                g = base_env.game
                sig = {
                    'num_players': len(getattr(g, 'players', [])),
                    'revo': bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False)),
                    'root_turn': getattr(g, 'turn', 0),
                }
                if sig != getattr(agent, '_det_root_signature', sig):
                    try:
                        agent._det_stats['discard_mismatch'] += 1
                    except Exception:
                        pass
                    try:
                        prev = getattr(agent, '_det_root_signature', None)
                        if prev and prev.get('num_players') == sig['num_players'] and prev.get('revo') == sig['revo'] and prev.get('root_turn') != sig['root_turn']:
                            agent._det_root_signature = sig
                            if getattr(agent, 'logger', None) and (agent._det_stats['discard_mismatch'] % 50 == 1):
                                agent.logger.log_text(
                                    f"[det-pool-update] root_turn change prev={prev.get('root_turn')} new={sig['root_turn']} discards={agent._det_stats['discard_mismatch']}"
                                )
                        else:
                            if getattr(agent, 'logger', None) and (agent._det_stats['discard_mismatch'] % 100 == 1):
                                agent.logger.log_text(
                                    f"[det-pool-mismatch] discards={agent._det_stats['discard_mismatch']} prev={prev} cur={sig}"
                                )
                    except Exception:
                        pass
                    time.sleep(0.001)
                    continue

                # assignmentの検証: 文字列が正しくパース可能かチェック
                if assignment and 'hands' in assignment:
                    from game.card import Card
                    is_valid = True
                    try:
                        for pid, cards_str in assignment['hands'].items():
                            for s in cards_str:
                                # 文字列が空でないことを確認
                                if not isinstance(s, str) or len(s) == 0:
                                    is_valid = False
                                    break
                                # Card.from_stringでパースして検証
                                parsed = Card.from_string(s)
                                # パース失敗を検出（JOKERでないのにis_joker=True）
                                if parsed.is_joker and not s.upper().startswith('JOKER'):
                                    is_valid = False
                                    try:
                                        agent._det_stats['pool_invalid_gen'] = agent._det_stats.get('pool_invalid_gen', 0) + 1
                                    except Exception:
                                        pass
                                    break
                            if not is_valid:
                                break
                    except Exception:
                        is_valid = False
                    
                    if not is_valid:
                        # 不正なassignmentは破棄
                        time.sleep(0.001)
                        continue

                with agent._det_pool_lock:
                    if len(agent._det_pool) < agent._det_cfg['capacity']:
                        agent._det_pool.append(assignment)
                        try:
                            agent._det_stats['generated'] += 1
                        except Exception:
                            pass
            else:
                time.sleep(0.001)
        except Exception as e:
            try:
                print(f"[DetPool][ERROR] worker 例外: {type(e).__name__}: {e}")
            except Exception:
                pass
            time.sleep(0.005)


def apply_from_det_pool(agent: Any, e_clone: Any, original_env: Any, root_pid: int) -> bool:
    if getattr(agent, '_det_pool', None) is None:
        return False
    try:
        import random as _r
        import time

        # プールが空の場合、短時間待機してから再試行（最大3回）
        max_retries = 3
        for retry in range(max_retries):
            with agent._det_pool_lock:
                if agent._det_pool:
                    # プールに割当がある - 取得
                    if agent._det_cfg['sampling'] == 'random':
                        idx = _r.randrange(len(agent._det_pool))
                        for _ in range(idx):
                            agent._det_pool.append(agent._det_pool.popleft())
                        assignment = agent._det_pool.popleft()
                    else:
                        assignment = agent._det_pool.popleft()
                    break
                elif retry < max_retries - 1:
                    # プールが空 - ロックを解放して少し待機
                    pass
                else:
                    # 最終試行でも空 - inline_determinizeにフォールバック
                    try:
                        agent._det_stats['pool_fallback_empty'] = agent._det_stats.get('pool_fallback_empty', 0) + 1
                        agent._det_stats['pool_fallback_reason'] = 'empty'
                    except Exception:
                        pass
                    return False
            
            # ロック外で待機（生成ワーカーがプールを補充できるように）
            if retry < max_retries - 1:
                time.sleep(0.001)  # 1ms待機
        else:
            # ループが正常終了しなかった（通常ここには到達しない）
            return False

        g_new = e_clone.game
        from game.card import Card

        for pid, cards_str in assignment['hands'].items():
            if pid == root_pid:
                continue
            try:
                # Card.from_stringがパース失敗時にJOKERを返すため、検証を追加
                parsed_cards = []
                for s in cards_str:
                    card = Card.from_string(s) if hasattr(Card, 'from_string') else Card(s)
                    # パース失敗の検出：元の文字列が'JOKER'でないのにis_joker=Trueになっている
                    if card.is_joker and not s.upper().startswith('JOKER'):
                        # パース失敗 - assignmentを破棄してFalseを返す（inline_determinizeにフォールバック）
                        try:
                            agent._det_stats['pool_parse_failures'] = agent._det_stats.get('pool_parse_failures', 0) + 1
                            # 最初の10回だけログ出力
                            if agent._det_stats['pool_parse_failures'] <= 10:
                                print(f"[DetPool][WARN] Parse failure: '{s}' -> JOKER (expected non-JOKER)")
                        except Exception:
                            pass
                        # プールに戻さない（不正なassignment）
                        try:
                            agent._det_stats['pool_fallback_parse'] = agent._det_stats.get('pool_fallback_parse', 0) + 1
                            agent._det_stats['pool_fallback_reason'] = 'parse'
                        except Exception:
                            pass
                        return False
                    parsed_cards.append(card)
                g_new.players[pid].hand = parsed_cards
            except Exception:
                g_new.players[pid].hand = list(cards_str)
        try:
            agent._det_stats['pool_hits'] += 1
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            print(f"[DetPool][ERROR] apply 失敗: {type(e).__name__}: {e}")
        except Exception:
            pass
        try:
            agent._det_stats['pool_fallback_exception'] = agent._det_stats.get('pool_fallback_exception', 0) + 1
            agent._det_stats['pool_fallback_reason'] = 'exception'
        except Exception:
            pass
        return False


def inline_determinize(agent: Any, e_clone: Any, original_env: Any, root_pid: int) -> None:
    build_single_determinization(agent, e_clone, original_env, root_pid, apply_direct=True)


def build_single_determinization(agent: Any, e_clone: Any, original_env: Any, root_pid: int, apply_direct: bool = False):
    try:
        g_orig = original_env.game
        g_new = e_clone.game
        history = getattr(g_orig, '_action_history', []) or []
        # Cardオブジェクトを直接扱う（文字列変換を避ける）
        from game.card import Card
        
        root_hand_cards = list(g_orig.players[root_pid].hand)
        field_cards = list(getattr(g_orig, 'current_field', []) or [])
        root_hand_ids = {str(c) for c in root_hand_cards}
        field_ids = {str(c) for c in field_cards}

        # 全カードをCardオブジェクトとして収集
        all_cards_objs = []
        try:
            if hasattr(g_orig, 'deck') and g_orig.deck:
                deck_obj = g_orig.deck
                # CardDeck は .cards に全カードを保持する
                if hasattr(deck_obj, 'cards') and getattr(deck_obj, 'cards', None):
                    all_cards_objs = list(deck_obj.cards)
                else:
                    all_cards_objs = list(deck_obj)
        except Exception:
            pass
        if not all_cards_objs:
            for p in g_orig.players:
                all_cards_objs.extend(p.hand)
            all_cards_objs.extend(field_cards)
            # 重複除去（文字列IDで判定）
            seen_ids = set()
            unique_cards = []
            skipped_cards = []
            from game.card import SUITS
            
            for card in all_cards_objs:
                card_id = str(card)
                if card_id not in seen_ids:
                    seen_ids.add(card_id)
                    # Cardオブジェクトを正規化（suitが数値インデックスの場合、SUITS文字列に変換）
                    if not getattr(card, 'is_joker', False):
                        suit_val = getattr(card, 'suit', None)
                        rank_val = getattr(card, 'rank', None)
                        
                        # suitが数値インデックス（0-3）の場合、SUITS文字列に変換
                        if isinstance(suit_val, int) and 0 <= suit_val < len(SUITS):
                            normalized_card = Card(suit=SUITS[suit_val], rank=rank_val, is_joker=False)
                            unique_cards.append(normalized_card)
                        elif suit_val in SUITS:
                            # suitがすでに正しい文字列形式
                            unique_cards.append(card)
                        else:
                            # 不正なsuit値でも、カードを保持（元のままで追加）
                            # これはカード数を維持するため
                            skipped_cards.append((str(card), suit_val, rank_val))
                            unique_cards.append(card)
                    else:
                        # Jokerはそのまま追加
                        unique_cards.append(card)
            
            all_cards_objs = unique_cards

        known = set(root_hand_ids) | field_ids
        for rid in getattr(g_orig, 'rankings', []):
            if rid != root_pid:
                known.update(str(c) for c in g_orig.players[rid].hand)
        for h in history:
            act = h.get('action')
            if isinstance(act, (list, tuple)):
                known.update(str(c) for c in act)
        # 未知カードをCardオブジェクトのリストとして保持
        unknown_seed = [card for card in all_cards_objs if str(card) not in known]

        # ===== パス制約の抽出（観測可能な情報に基づく） =====
        # 各プレイヤーがパスした場面で、場に出ていたカードを記録し、
        # それを上回れなかったことを制約として扱う
        pass_reqs = []
        for h in history:
            if h.get('action') == 'pass':
                fb = h.get('field_before') or []
                if not fb:
                    continue
                cnt = len(fb)
                ranks = []
                for cid in fb:
                    try:
                        core = cid[:-1]
                        ranks.append(int(core))
                    except Exception:
                        pass
                base_rank = min(ranks) if ranks else 0
                pass_reqs.append({
                    'pid': h.get('pid'), 'count': cnt, 'min_rank': base_rank,
                    'revo': bool(h.get('revo', False)), 'combo_type': h.get('combo_type'),
                    'size': cnt, 'ranks_ref': sorted(ranks),
                })

        def rank_of(cid: str):
            core = cid[:-1]
            try:
                return int(core)
            except Exception:
                return 0

        import random as _r

        opp_ids = [i for i in range(len(g_new.players)) if i != root_pid]
        max_retry = max(1, int(agent._det_cfg.get('retry_max', 8)))

        guided_enable = False
        hand_card_probs = None  # dict: opponent_pid -> list[53] (probabilities)
        try:
            mdl = agent.model
            if bool(getattr(agent.config, 'get', lambda k, d=None: d)('hand_pred_use_in_determinization', False)) or \
               bool(agent.config.get('hand_pred_use_in_determinization', False)):
                # 予測ヘッドがモデルに存在するか確認
                num_players_local = len(g_new.players)
                remote_hp = getattr(agent, '_remote_hand_probs', None)
                fresh = True
                try:
                    ttl = float(agent.config.get('hand_pred_cache_ttl_sec', 0.5) or 0.5)
                    ts = float(getattr(agent, '_remote_hand_probs_ts', 0.0) or 0.0)
                    import time as _time
                    if ttl > 0 and (ts <= 0 or (_time.time() - ts) > ttl):
                        fresh = False
                except Exception:
                    fresh = True
                if fresh and isinstance(remote_hp, (list, tuple)) and len(remote_hp) == 53 * (num_players_local - 1):
                    opponents_order = [i for i in range(num_players_local) if i != root_pid]
                    hand_card_probs = {}
                    for oi, pid_ in enumerate(opponents_order):
                        start = oi * 53
                        hand_card_probs[pid_] = list(remote_hp[start:start+53])
                    guided_enable = True
                elif agent.model is not None and getattr(agent.model, 'enable_hand_prediction_head', False) and hasattr(agent.model, 'forward_with_belief'):
                    state_root = agent._extract_state(original_env)
                    import torch as _t
                    _dev = getattr(agent.model, 'device', None)
                    _use_amp = bool(getattr(_dev, 'type', None) == 'cuda')
                    with _t.no_grad():
                        with _t.amp.autocast('cuda', enabled=_use_amp):
                            try:
                                _, _, hand_logits_t = agent.model.forward_with_belief(state_root)
                            except Exception:
                                hand_logits_t = None
                    if hand_logits_t is not None:
                        try:
                            import torch as _t
                            if isinstance(hand_logits_t, _t.Tensor):
                                hp = _t.sigmoid(hand_logits_t).detach().cpu().tolist()
                            else:
                                import math as _m
                                seq = (hand_logits_t.detach().cpu().tolist() if hasattr(hand_logits_t, 'detach') else (hand_logits_t.tolist() if hasattr(hand_logits_t, 'tolist') else list(hand_logits_t)))
                                hp = [1.0/(1.0+_m.exp(-float(x))) for x in seq]
                        except Exception:
                            hp = list(hand_logits_t)
                        expected = 53 * (num_players_local - 1)
                        if len(hp) == expected:
                            opponents_order = [i for i in range(num_players_local) if i != root_pid]
                            hand_card_probs = {}
                            for oi, pid_ in enumerate(opponents_order):
                                start = oi * 53
                                hand_card_probs[pid_] = hp[start:start+53]
                            guided_enable = True
        except Exception:
            guided_enable = False
            hand_card_probs = None

        for attempt in range(max_retry):
            unknown = list(unknown_seed)
            guided_used = False
            if guided_enable and attempt == 0:
                try:
                    capacities = {i: len(g_new.players[i].hand) for i in opp_ids}
                    assignment_hands = {i: [] for i in opp_ids}
                    from game.card import Card

                    def _card_index_from_obj(c_obj) -> int:
                        """CardオブジェクトからインデックスIDを取得"""
                        try:
                            if getattr(c_obj, 'is_joker', False):
                                return 52
                            suit_order = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}
                            s = suit_order.get(getattr(c_obj, 'suit', 'S'), 0)
                            r = int(getattr(c_obj, 'rank', 1)) - 1
                            idx = s * 13 + r
                            if idx < 0 or idx >= 53:
                                return 52
                            return idx
                        except Exception:
                            return 52

                    # Step A (確定枠): 高確率カードを確率順にソートして割り当て
                    threshold = float(agent.config.get('hand_pred_deterministic_threshold', 0.95))
                    candidate_list = []  # [(card_obj, player_id, probability), ...]
                    
                    for card_obj in unknown:
                        idx = _card_index_from_obj(card_obj)
                        for pid_ in opp_ids:
                            p_val = 0.0
                            try:
                                p_list = hand_card_probs.get(pid_, None)
                                if p_list and 0 <= idx < len(p_list):
                                    p_val = float(p_list[idx])
                            except (KeyError, IndexError, TypeError, ValueError):
                                p_val = 0.0
                            if p_val > 0.0:
                                candidate_list.append((card_obj, pid_, p_val))
                    
                    # 確率の降順でソート（高い確率から順に割り当て）
                    candidate_list.sort(key=lambda x: x[2], reverse=True)
                    
                    # Step A: 高確率カードを確定割り当て
                    deterministic_cards = set()  # Cardオブジェクトのセット（IDで管理）
                    for card_obj, pid_, prob in candidate_list:
                        card_id = str(card_obj)
                        if prob >= threshold and card_id not in {str(c) for c in deterministic_cards}:
                            if capacities.get(pid_, 0) > 0:
                                assignment_hands[pid_].append(card_obj)  # Cardオブジェクトを格納
                                capacities[pid_] -= 1
                                deterministic_cards.add(card_obj)
                    
                    # Step B (抽選枠): 残りのカードを確率分布に従って抽選
                    deterministic_ids = {str(c) for c in deterministic_cards}
                    remaining_unknown = [card for card in unknown if str(card) not in deterministic_ids]
                    _r.shuffle(remaining_unknown)
                    
                    for card_obj in remaining_unknown:
                        remaining_opps = [i for i in opp_ids if capacities[i] > 0]
                        if not remaining_opps:
                            break
                        idx = _card_index_from_obj(card_obj)
                        probs_raw = []
                        total = 0.0
                        for pid_ in remaining_opps:
                            p_val = 0.0
                            try:
                                p_list = hand_card_probs.get(pid_, None)
                                if p_list and 0 <= idx < len(p_list):
                                    p_val = float(p_list[idx])
                            except (KeyError, IndexError, TypeError, ValueError):
                                p_val = 0.0
                            probs_raw.append(p_val)
                            total += p_val
                        
                        if total <= 1e-12:
                            probs_norm = [1.0 / len(remaining_opps)] * len(remaining_opps)
                        else:
                            probs_norm = [p / total for p in probs_raw]
                        
                        r = _r.random()
                        cum = 0.0
                        chosen = remaining_opps[-1]
                        for j, pid_ in enumerate(remaining_opps):
                            cum += probs_norm[j]
                            if r <= cum:
                                chosen = pid_
                                break
                        assignment_hands[chosen].append(card_obj)  # Cardオブジェクトを保持
                        capacities[chosen] -= 1

                    # 未割り当てカードがあれば均等に配分
                    assigned_ids = set()
                    for hand_list in assignment_hands.values():
                        assigned_ids.update(str(c) for c in hand_list)
                    leftover_unknown = [card for card in unknown if str(card) not in assigned_ids]
                    if leftover_unknown:
                        for card_obj in leftover_unknown:
                            targets = [i for i in opp_ids if capacities[i] > 0]
                            if not targets:
                                break
                            choice = _r.choice(targets)
                            assignment_hands[choice].append(card_obj)  # Cardオブジェクトを保持
                            capacities[choice] -= 1

                    # 割り当て結果の検証（枚数一致チェック）
                    mismatch = False
                    for i in opp_ids:
                        expected_count = len(g_new.players[i].hand)
                        actual_count = len(assignment_hands[i])
                        if actual_count != expected_count:
                            mismatch = True
                            break
                            break
                    
                    if not mismatch:
                        guided_used = True
                        if hasattr(agent, '_det_stats'):
                            agent._det_stats['guided_assignments'] = agent._det_stats.get('guided_assignments', 0) + 1
                except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
                    # 想定される例外のみ捕捉（デバッグ情報付き）
                    guided_used = False
                    if hasattr(agent, '_det_stats'):
                        agent._det_stats['guided_failures'] = agent._det_stats.get('guided_failures', 0) + 1
                except Exception as e:
                    # 予期しない例外は再スローして問題を可視化
                    guided_used = False
                    raise

            if not guided_used:
                _r.shuffle(unknown)
                cursor = 0
                assignment_hands = {}
                for i in opp_ids:
                    p_new = g_new.players[i]
                    sz = len(p_new.hand)
                    pick = unknown[cursor:cursor + sz]
                    cursor += sz
                    # Cardオブジェクトをそのまま保持
                    assignment_hands[i] = list(pick)

            # ===== パス制約の検証（観測と矛盾しない手札割当か） =====
            # プレイヤーがパスした場面で、実際にはその場を上回れる手札を持っていたなら、
            # その割当は観測と矛盾するため棄却する
            consistent = True
            for req in pass_reqs:
                pidc = req['pid']
                if pidc == root_pid or pidc is None:
                    continue
                # assignment_handsはCardオブジェクトのリスト
                hand_cards = assignment_hands.get(pidc, [])
                hand_ids = {str(c) for c in hand_cards}
                
                # 手札のランクごとの枚数をカウント
                counts = {}
                has_joker = False
                for card_obj in hand_cards:
                    if getattr(card_obj, 'is_joker', False):
                        has_joker = True
                    else:
                        rnk = int(getattr(card_obj, 'rank', 0))
                        counts[rnk] = counts.get(rnk, 0) + 1
                
                combo_type = req.get('combo_type')
                feas = []
                
                # 階段（ストレート）の場合
                if combo_type == 'straight':
                    size_req = req.get('size', req['count'])
                    min_rank_ref = req['min_rank']
                    ranks_sorted = sorted(counts.keys())
                    
                    # Jokerがあれば柔軟に判定（簡易版: Jokerで1枚補完可能と仮定）
                    if ranks_sorted:
                        for idx in range(len(ranks_sorted)):
                            window = ranks_sorted[idx:idx + size_req]
                            if len(window) < size_req:
                                break
                            # 連続性チェック
                            is_consecutive = all(window[i + 1] - window[i] == 1 for i in range(size_req - 1))
                            # Jokerがあれば1枚分のギャップを許容
                            if not is_consecutive and has_joker:
                                gaps = [window[i + 1] - window[i] for i in range(size_req - 1)]
                                if gaps.count(2) == 1 and all(g in (1, 2) for g in gaps):
                                    is_consecutive = True  # Jokerで補完可能
                            
                            if is_consecutive:
                                if req['revo']:
                                    if window[0] <= min_rank_ref:
                                        feas.append(tuple(window))
                                else:
                                    if window[0] >= min_rank_ref:
                                        feas.append(tuple(window))
                # ペア・トリプル・フォーカードの場合
                else:
                    count_needed = req['count']
                    if not req['revo']:
                        # 通常: 場のランク以上が必要
                        feas = [r for r, cnt in counts.items() 
                                if r >= req['min_rank'] and cnt >= count_needed]
                        # Jokerがあれば、count_needed-1枚でも成立可能
                        if has_joker:
                            feas.extend([r for r, cnt in counts.items() 
                                        if r >= req['min_rank'] and cnt >= (count_needed - 1) and cnt < count_needed])
                    else:
                        # 革命: 場のランク以下が必要
                        feas = [r for r, cnt in counts.items() 
                                if r <= req['min_rank'] and cnt >= count_needed]
                        # Jokerがあれば、count_needed-1枚でも成立可能
                        if has_joker:
                            feas.extend([r for r, cnt in counts.items() 
                                        if r <= req['min_rank'] and cnt >= (count_needed - 1) and cnt < count_needed])
                
                # 上回れる手があるのにパスしたなら矛盾
                if feas:
                    consistent = False
                    break

            if consistent:
                if apply_direct:
                    from game.card import Card
                    
                    def _clone_card(c):
                        """Cardオブジェクトをクローン"""
                        if getattr(c, 'is_joker', False):
                            return Card(is_joker=True)
                        else:
                            return Card(
                                suit=getattr(c, 'suit', None),
                                rank=getattr(c, 'rank', None),
                                is_joker=False
                            )
                    
                    for pid, cards_list in assignment_hands.items():
                        if pid == root_pid:
                            continue
                        g_new.players[pid].hand = [_clone_card(c) for c in cards_list]
                    return True, None
                else:
                    # apply_direct=Falseの場合、文字列IDに変換して返す
                    # 検証: Cardオブジェクトが正しい形式であることを確認してから文字列化
                    assignment_str = {}
                    from game.card import Card, SUITS
                    
                    for pid, cards in assignment_hands.items():
                        str_cards = []
                        for c in cards:
                            try:
                                # Cardオブジェクトを正規化（suitが数値の場合、SUITS文字列に変換）
                                normalized_card = c
                                if not getattr(c, 'is_joker', False):
                                    suit_val = getattr(c, 'suit', None)
                                    rank_val = getattr(c, 'rank', None)
                                    
                                    # suitが数値インデックス（0-3）の場合、SUITS文字列に変換
                                    if isinstance(suit_val, int) and 0 <= suit_val < len(SUITS):
                                        # 新しいCardオブジェクトを作成（正しいsuit文字列で）
                                        normalized_card = Card(suit=SUITS[suit_val], rank=rank_val, is_joker=False)
                                    elif suit_val not in SUITS:
                                        # 不正なsuit値 - スキップ
                                        raise ValueError(f"Invalid suit: {suit_val}")
                                    
                                    # rankの検証
                                    if rank_val is None or not isinstance(rank_val, int) or rank_val < 1 or rank_val > 13:
                                        raise ValueError(f"Invalid rank: {rank_val}")
                                
                                # 文字列化
                                card_str = str(normalized_card)
                                
                                # 空文字列でないことを確認
                                if not card_str:
                                    raise ValueError("Empty card string")
                                
                                # パースして検証（ラウンドトリップテスト）
                                parsed = Card.from_string(card_str)
                                # パース失敗を検出
                                if parsed.is_joker and not card_str.upper().startswith('JOKER'):
                                    # 元のカードがJOKERかどうか確認
                                    if not getattr(c, 'is_joker', False):
                                        # 不正な文字列化 - スキップ
                                        try:
                                            agent._det_stats['str_roundtrip_fail'] = agent._det_stats.get('str_roundtrip_fail', 0) + 1
                                            # デバッグ: 最初の5回だけログ出力
                                            fail_count = agent._det_stats['str_roundtrip_fail']
                                            if fail_count <= 5:
                                                print(f"[DEBUG] Roundtrip fail #{fail_count}: orig_card(suit={getattr(c, 'suit', None)}, rank={getattr(c, 'rank', None)}, is_joker={getattr(c, 'is_joker', False)}) -> '{card_str}' -> parsed(suit={parsed.suit}, rank={parsed.rank}, is_joker={parsed.is_joker})")
                                        except Exception:
                                            pass
                                        raise ValueError(f"Parse failure: '{card_str}' -> JOKER")
                                str_cards.append(card_str)
                            except Exception as e:
                                # 文字列化に失敗したカードはスキップ（assignmentを無効化）
                                try:
                                    agent._det_stats['str_conversion_fail'] = agent._det_stats.get('str_conversion_fail', 0) + 1
                                    # デバッグ: 最初の3回だけログ出力
                                    fail_count = agent._det_stats['str_conversion_fail']
                                    if fail_count <= 3:
                                        print(f"[DEBUG] Conversion fail #{fail_count}: card(suit={getattr(c, 'suit', None)}, rank={getattr(c, 'rank', None)}, is_joker={getattr(c, 'is_joker', False)}) Error: {type(e).__name__}: {e}")
                                except Exception:
                                    pass
                                return False, None  # assignment生成失敗
                        assignment_str[pid] = str_cards
                    try:
                        agent._det_stats['retries_total'] += attempt
                    except Exception:
                        pass
                    return True, {'hands': assignment_str}

        # 再試行が尽きた場合、最後のassignment_handsを使用
        if apply_direct:
            from game.card import Card
            
            def _clone_card(c):
                """Cardオブジェクトをクローン"""
                if getattr(c, 'is_joker', False):
                    return Card(is_joker=True)
                else:
                    return Card(
                        suit=getattr(c, 'suit', None),
                        rank=getattr(c, 'rank', None),
                        is_joker=False
                    )
            
            for pid, cards_list in assignment_hands.items():
                if pid == root_pid:
                    continue
                g_new.players[pid].hand = [_clone_card(c) for c in cards_list]
            return True, None
        else:
            return False, None
    except Exception as e:
        try:
            print(f"[DetBuild][ERROR] 例外: {type(e).__name__}: {e}")
        except Exception:
            pass
        return False, None


def shutdown_det_pool(agent: Any) -> None:
    try:
        if getattr(agent, '_det_stop_event', None) is not None:
            agent._det_stop_event.set()
        th = getattr(agent, '_det_pool_thread', None)
        if th and th.is_alive():
            th.join(timeout=1.0)
    except Exception:
        pass
    try:
        agent._det_pool = None
        agent._det_pool_thread = None
        agent._det_pool_lock = None
        agent._det_stop_event = None
    except Exception:
        pass


__all__ = [
    'AlphaZeroTTView',
    'maybe_start_det_pool',
    'determinization_worker',
    'apply_from_det_pool',
    'inline_determinize',
    'build_single_determinization',
    'shutdown_det_pool',
]
