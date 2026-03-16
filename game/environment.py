import random
import numpy as np
from game.game import Game
from agents.straight_agent import StraightAgent
from game.card import Card
from agents.mcts import MCTSAgent
from agents.drl_agent import AlphaZeroAgent



class DaifugoSimpleEnv:

    def __init__(self, num_players=4, agent_classes=None):
        self.num_players = num_players   #プレイヤーの人数設定
        self.game = Game(num_players=self.num_players)  # Game クラスのインスタンス生成
        self.current_player = self.game.turn  # 現在のプレイヤー番号（ターン）
        self.done = False  # ゲーム終了フラグ
        # agent_classes: [AgentClass, ...] で指定できる。なければ全員StraightAgent
        if agent_classes is None:
            agent_classes = [StraightAgent] * num_players
        self.agents = [agent_classes[i](player_id=i) for i in range(num_players)]
        # 区間履歴バッファ
        self.stage_history = []  # 各区間のstep履歴（dictのリスト）
        self.already_won_players = set()  # 区間開始時点ですでに上がっていたプレイヤー
        self.stage_id = 0  # 区間ID
        self.turn_idx = 0  # ゲーム全体の手番番号
        # デバッグ用: env.step 内の合法手/選択手のミスマッチを詳細ログするフラグ
        self.debug_action_mismatch = True
        # フォールバックログの重複抑制
        self._fallback_logged = set()

    def _log_fallback_once(self, key: str, msg: str, exc: Exception | None = None) -> None:
        try:
            if key in self._fallback_logged:
                return
            self._fallback_logged.add(key)
        except Exception:
            pass
        try:
            text = f"{msg} ({type(exc).__name__}: {exc})" if exc is not None else msg
            print(text)
        except Exception:
            pass

    def _is_pair(self, cards):
        """
        与えられたカードリストがペア（同ランク or ジョーカー）か判定
        """
        if not cards or len(cards) < 2:
            return False
        non_jokers = [c for c in cards if not c.is_joker]
        if not non_jokers:
            return True
        rank = non_jokers[0].rank
        return all(c.rank == rank or c.is_joker for c in cards)

    def _make_pair_sets(self, hand, jokers, field_count, rule_checker, field):
        """
        手札からペア・スリーカード・フォーカードの組み合わせを生成
        """
        import itertools
        legal_actions = []
        rank_map = {}
        for card in hand:
            if not card.is_joker:
                rank_map.setdefault(card.rank, []).append(card)
        for rank, cards_same_rank in rank_map.items():
            max_size = min(len(cards_same_rank) + len(jokers), 4)
            for k in range(field_count, max_size + 1):
                # 実際の純粋カード部分の選択 (0..len(cards_same_rank))
                for comb in itertools.combinations(cards_same_rank, min(len(cards_same_rank), k)):
                    needed_jokers = k - len(comb)
                    if needed_jokers < 0 or needed_jokers > len(jokers):
                        continue
                    pair = list(comb)
                    used_jokers_state = []
                    # 必要数ジョーカーを実際の手札ジョーカーから利用 (複製せずに一時的に代用ランクをセット)
                    for j_idx in range(needed_jokers):
                        jk = jokers[j_idx]
                        used_jokers_state.append((jk, jk.joker_as_rank, jk.joker_as_suit))
                        jk.set_joker_substitute(rank, comb[0].suit if comb else jk.joker_as_suit or '♠')
                        pair.append(jk)
                    if rule_checker.is_valid_move(pair, field):
                        legal_actions.append(list(pair))
                    # ジョーカーの代用情報を元に戻す
                    for jk, r_old, s_old in used_jokers_state:
                        jk.joker_as_rank = r_old
                        jk.joker_as_suit = s_old
        return legal_actions

    def _make_straight_sets(self, hand, jokers, field_count, rule_checker, field):
        """
        手札から階段の組み合わせを生成
        """
        import itertools
        legal_actions = []
        suit_map = {}
        for card in hand:
            if not card.is_joker:
                suit_map.setdefault(card.suit, []).append(card)
        for suit, cards_in_suit in suit_map.items():
            for start in range(1, 15 - field_count):
                expected = [(start + i - 1) % 13 + 1 for i in range(field_count)]
                # 2は末尾以外に出現不可
                if 2 in expected and expected[-1] != 2:
                    continue
                seq = []
                used_jokers_state = []  # (joker, old_rank, old_suit)
                success = True
                # 手札コピー (破壊しない)
                pool = cards_in_suit[:]
                for val in expected:
                    placed = False
                    for i, c in enumerate(pool):
                        if c.rank == val:
                            seq.append(pool.pop(i))
                            placed = True
                            break
                    if not placed:
                        # Joker で埋める
                        joker_needed_idx = len(used_jokers_state)
                        if joker_needed_idx < len(jokers):
                            jk = jokers[joker_needed_idx]
                            used_jokers_state.append((jk, jk.joker_as_rank, jk.joker_as_suit))
                            jk.set_joker_substitute(val, suit)
                            seq.append(jk)
                        else:
                            success = False
                            break
                if success and len(seq) == field_count:
                    # 期待順に並べ替え (seq は既に順序通りだが安全のため)
                    ordered = []
                    for v in expected:
                        for c in seq:
                            rank = c.joker_as_rank if c.is_joker else c.rank
                            if rank == v:
                                ordered.append(c)
                                break
                    if rule_checker.is_valid_move(ordered, field):
                        legal_actions.append(list(ordered))
                # Joker 元に戻す
                for jk, r_old, s_old in used_jokers_state:
                    jk.joker_as_rank = r_old
                    jk.joker_as_suit = s_old
        return legal_actions

    def _remove_duplicate_actions(self, legal_actions):
        """
        legal_actionsの重複除去（カードの等価性で）
        """
        def cardset_key(cardset):
            if cardset is None:
                return (None,)
            return tuple(sorted(str(c) for c in cardset))
        unique = {}
        for action in legal_actions:
            unique[cardset_key(action)] = action
        return list(unique.values())

    def _action_key(self, action_cards):
        """
        List[Card] / List[str] / None を順序非依存のキーへ正規化
        """
        if action_cards is None:
            return (None,)
        try:
            return tuple(sorted(str(c) for c in action_cards))
        except Exception:
            return (None,)

    def _joker_fuzzy_key(self, action_cards):
        """Joker の表記ゆれを吸収した順序非依存キーを生成する。

        - 'JOKER', 'JOKER(♥7)', 'JOKER(♦2)' などをすべて 'JOKER' とみなす
        - それ以外のカードは str(c) をそのまま使う
        - sorted してタプル化することで、ペア/スリーカード/階段などの
          枚数と組み合わせを厳密に一致判定できる
        """
        if action_cards is None:
            return (None,)
        tokens = []
        try:
            for c in action_cards:
                s = str(c)
                if s.startswith("JOKER"):
                    tokens.append("JOKER")
                else:
                    tokens.append(s)
            return tuple(sorted(tokens))
        except Exception:
            return (None,)

    def _normalize_action_input(self, action_cards):
        """
        A-1の契約に従い、受け取ったactionを None or List[str] に正規化
        （List[Card] の場合はそのまま返す）
        """
        if action_cards == [] or action_cards == "pass":
            return None
        if isinstance(action_cards, str):
            return [action_cards]
        if isinstance(action_cards, tuple):
            # MCTS(AlphaZero) のキー tuple(sorted(str(card))) を想定
            try:
                return list(action_cards)
            except Exception:
                return None
        if action_cards is not None and not isinstance(action_cards, list):
            self._log_fallback_once(
                "normalize_invalid_type",
                f"[env-fallback] invalid action type -> None (type={type(action_cards).__name__})"
            )
            return None
        return action_cards

    def reset(self):
        # ゲームをリセット（インスタンスは使い回し、rankingsを維持）
        self.game.reset()
        self.current_player = self.game.turn
        self.done = False
        self.stage_history = []
        self.already_won_players = set()
        self.stage_id = 0
        self.turn_idx = 0
        return self._get_obs()

    def _generate_legal_actions(self, hand, field):
        """
        現在の手札と場の状態から出せる全ての合法なカードセット（legal actions）を列挙する。
        出せるカードがない場合のみパス(None)を含める。
        """
        rule_checker = self.game.rule_checker
        legal_actions = []
        field_combo = rule_checker.classify_combo(field) if field else None
        import itertools

        # 候補生成: 1枚 / 同ランク集合(2-4) / 階段長(3-6) を網羅探索
        # 1枚
        for card in hand:
            candidate = [card]
            if rule_checker.is_valid_move(candidate, field):
                legal_actions.append(candidate)

        # 同ランク系
        rank_map = {}
        for card in hand:
            if not card.is_joker:
                rank_map.setdefault(card.rank, []).append(card)
        jokers = [c for c in hand if c.is_joker]
        for rank, same_cards in rank_map.items():
            base_len = len(same_cards)
            for size in range(2, 5):  # 最大4枚
                if base_len + len(jokers) < size:
                    continue
                # 実カード組合せ (0..base_len)
                for r in range(max(1, size - len(jokers)), min(base_len, size) + 1):
                    for comb in itertools.combinations(same_cards, r):
                        needed_j = size - r
                        if needed_j < 0 or needed_j > len(jokers):
                            continue
                        used_j = []
                        cand = list(comb)
                        for j in range(needed_j):
                            jk = jokers[j]
                            used_j.append((jk, jk.joker_as_rank, jk.joker_as_suit))
                            jk.set_joker_substitute(rank, comb[0].suit if comb else '♠')
                            cand.append(jk)
                        if rule_checker.is_valid_move(cand, field):
                            legal_actions.append(list(cand))
                        for jk, r_old, s_old in used_j:
                            jk.joker_as_rank = r_old
                            jk.joker_as_suit = s_old

        # 階段候補: スート別 → 長さ 3..6 を brute force (既存関数再利用)
        for length in range(3, 7):
            legal_actions += self._make_straight_sets(hand, jokers, length, rule_checker, field)

        # Joker単体 (他ロジックと重複するが安全に明示)
        for card in hand:
            if card.is_joker and rule_checker.is_valid_move([card], field):
                legal_actions.append([card])

        # フィルタ & 重複除去
        hand_counts = {}
        for c in hand:
            hand_counts[str(c)] = hand_counts.get(str(c), 0) + 1
        filtered = []
        for act in legal_actions:
            if not act:
                continue
            local = {}
            ok = True
            for c in act:
                sc = str(c)
                local[sc] = local.get(sc, 0) + 1
                if local[sc] > hand_counts.get(sc, 0):
                    ok = False
                    break
            if ok:
                filtered.append(act)
        filtered = self._remove_duplicate_actions(filtered)

        # 場が存在する場合: type/size 不一致のものを削除
        if field_combo:
            keep = []
            for a in filtered:
                combo = rule_checker.classify_combo(a)
                if combo and combo['type'] == field_combo['type'] and combo['size'] == field_combo['size'] and rule_checker.compare_combos(combo, field_combo):
                    keep.append(a)
            filtered = keep

        # Shibari constraint
        if hasattr(self.game, 'shibari_active') and getattr(self.game, 'shibari_active', False):
            lock_suits = getattr(self.game, 'lock_suits', None)
            if lock_suits:
                shibari_filtered = []
                for action in filtered:
                    if action is None:  # Pass always allowed
                        shibari_filtered.append(action)
                        continue
                    # Extract suit pattern
                    try:
                        action_suits = sorted(set(c.suit for c in action if not getattr(c, 'is_joker', False)))
                        # Check match
                        if action_suits == lock_suits:
                            shibari_filtered.append(action)
                    except Exception:
                        pass
                # If no valid actions remain, ensure at least pass is available
                filtered = shibari_filtered if shibari_filtered else [None]

        if not filtered:
            # 合法手がない場合はパスのみ（正常動作なのでログ不要）
            filtered = [None]
        return filtered

    def step(self, return_info=False, external_action=None, mcts_result=None, force_action=None, simulate=False):
        rule_checker = self.game.rule_checker
        current_player_id = self.game.turn
        player = self.game.players[current_player_id]
        hand = player.hand
        field = self.game.current_field[:]

        # legal_actions生成
        legal_actions = self._generate_legal_actions(hand, field)
        obs = {'hand': hand, 'field': field}

        # --- 行動選択 ---
        corrected_external = False
        corrected_from = None
        if force_action is not None:
            action_cards = self._normalize_action_input(force_action)
            mcts_result = None
        elif external_action is not None:
            action_cards = self._normalize_action_input(external_action)
            corrected_from = action_cards
        else:
            is_field_straight = rule_checker.is_straight(field) if field else False
            is_field_pair = False
            if field and not is_field_straight:
                non_jokers = [c for c in field if not c.is_joker]
                if non_jokers and all(c.rank == non_jokers[0].rank or c.is_joker for c in field):
                    is_field_pair = True if len(field) >= 2 else False

            if is_field_straight:
                filtered_actions = [a for a in legal_actions if a is not None and rule_checker.is_straight(a)]
                if not filtered_actions:
                    filtered_actions = [None]
            elif is_field_pair:
                filtered_actions = [a for a in legal_actions if a is not None
                                    and rule_checker.is_same_rank_or_joker(a)
                                    and len(a) == len(field)
                                    and rule_checker.is_valid_move(a, field)]
                if not filtered_actions:
                    filtered_actions = [None]
            else:
                filtered_actions = legal_actions

            if simulate and isinstance(self.agents[current_player_id], (MCTSAgent, AlphaZeroAgent)):
                # simulate=True (MCTS内部シミュレーション) ではランダムに行動を選ぶ。
                # ただし、pass(None) は合法手に含まれる場合のみ候補に入れる。
                legal_actions_filtered = [a for a in legal_actions if a is not None]
                allow_pass = any(a is None for a in legal_actions)
                pool = legal_actions_filtered + ([None] if allow_pass else [])
                self._log_fallback_once(
                    "simulate_random_action",
                    "[env-fallback] simulate=True -> random action chosen for MCTS/AlphaZero"
                )
                action_cards = random.choice(pool) if pool else None
            else:
                # Try to obtain model policy/value for the current state (best-effort)
                agent = self.agents[current_player_id]
                try:
                    pol, val = None, None
                    if hasattr(agent, '_policy_value'):
                                try:
                                    pol_raw, val_raw = agent._policy_value(self)
                                    pol = {}
                                    # normalize keys to strings for JSON logging
                                    for a, p in (pol_raw or {}).items():
                                        try:
                                            if a is None:
                                                key = 'pass'
                                            elif isinstance(a, (list, tuple)):
                                                key = '|'.join(str(x) for x in a)
                                            else:
                                                key = str(a)
                                            pol[key] = float(p)
                                        except Exception:
                                            continue
                                    # convert val_raw to scalar for current player if possible
                                    val = None
                                    try:
                                        if val_raw is None:
                                            val = None
                                        elif isinstance(val_raw, dict):
                                            # try integer key or string key
                                            val = val_raw.get(current_player_id, val_raw.get(str(current_player_id)))
                                        elif isinstance(val_raw, (list, tuple)):
                                            if 0 <= current_player_id < len(val_raw):
                                                val = val_raw[current_player_id]
                                            else:
                                                val = None
                                        else:
                                            # scalar
                                            val = float(val_raw)
                                    except Exception:
                                        val = None
                                except Exception:
                                    pol, val = None, None
                except Exception:
                    pol, val = None, None

                # select action (may use MCTS)
                # AlphaZeroAgent needs the env to enforce final legality gates reliably.
                if isinstance(agent, AlphaZeroAgent):
                    action_cards = agent.select_action(self, training=True, legal_actions=filtered_actions)
                else:
                    action_cards = agent.select_action(obs, legal_actions=filtered_actions)

        # 入力正規化（A-1）
        raw_action_cards = action_cards
        action_cards = self._normalize_action_input(action_cards)
        if raw_action_cards is not None and action_cards is None:
            self._log_fallback_once(
                "normalize_to_none",
                "[env-fallback] action normalization resulted in None"
            )

        # 合法手チェック（A-4）: 生成済み合法手に含まれない出しはパス or 合法手へ矯正
        try:
            pass_only = (legal_actions is not None and all(a is None for a in legal_actions))
        except Exception:
            pass_only = False

        # pass-only の局面では、常にパスへクランプ
        # external_action で非パスが来た場合はデバッグ用にミスマッチを記録する
        # simulate=True (MCTS内部シミュレーション) の場合はログをスキップ
        if pass_only:
            if external_action is not None and getattr(self, "debug_action_mismatch", False) and not simulate:
                try:
                    if action_cards is not None:
                        _revo_state = bool(getattr(rule_checker, 'revolution', False))
                        print(
                            "[ACTION-MISMATCH] pid=", current_player_id,
                            " field=", [str(c) for c in field],
                            " revo=", _revo_state,
                            " raw_action=", raw_action_cards,
                            " norm_action=", [str(c) for c in action_cards] if action_cards is not None else None,
                            " legal_actions=", [None],
                            " act_key=", self._action_key(action_cards),
                            " legal_keys=", [],
                        )
                except Exception:
                    pass
            action_cards = None
        elif action_cards is not None:
            legal_keys = set(self._action_key(a) for a in legal_actions if a is not None)
            act_key = self._action_key(action_cards)
            if act_key not in legal_keys:
                # --- Joker の曖昧マッチ（Fuzzy Match） ---
                # str 表現だけ異なる JOKER(X) 同士を同一視して、
                # 枚数・組み合わせが一致する合法手があればそれを採用する
                fuzzy_act_key = self._joker_fuzzy_key(action_cards)
                chosen_action = None
                for cand in legal_actions:
                    if cand is None:
                        continue
                    if fuzzy_act_key == self._joker_fuzzy_key(cand):
                        # env 側の表現（str(card)）に合わせて正規化
                        chosen_action = [str(c) for c in cand]
                        break

                if chosen_action is not None:
                    # Joker の表記ゆれだけで合法手に対応がある場合はそれを採用
                    action_cards = chosen_action
                else:
                    # Joker 曖昧マッチでも対応が見つからなかった場合:
                    #   1) デバッグ時は詳細をログ
                    #   2) 合法手集合からフォールバックを選択
                    #      - 非 None の合法手があればその1つ目を採用
                    #      - 合法手がパス(None)のみならパスのまま

                    # 1) デバッグオプション有効時はミスマッチの詳細を1行ログに出す
                    # simulate=True (MCTS内部シミュレーション) の場合はログをスキップ
                    if getattr(self, "debug_action_mismatch", False) and force_action is None and not simulate:
                        try:
                            la_str = [[str(c) for c in a] if a is not None else None for a in legal_actions]
                            # 革命状態を取得して出力
                            _revo_state = bool(getattr(rule_checker, 'revolution', False))
                            print("[ACTION-MISMATCH] pid=", current_player_id,
                                  " field=", [str(c) for c in field],
                                  " revo=", _revo_state,
                                  " raw_action=", raw_action_cards,
                                  " norm_action=", [str(c) for c in action_cards] if action_cards is not None else None,
                                  " legal_actions=", la_str,
                                  " act_key=", act_key,
                                  " legal_keys=", list(legal_keys))
                        except Exception:
                            pass

                    # 2) フォールバック: 合法手が存在するならパスではなく何かを出す
                    non_pass_candidates = [a for a in legal_actions if a is not None]
                    if non_pass_candidates:
                        # env 側表現に合わせて文字列化して渡す（正常な修正動作なのでログ不要）
                        fallback = non_pass_candidates[0]
                        action_cards = [str(c) for c in fallback]
                    else:
                        # 出せるカードが本当に無い場合のみパスにする（正常動作なのでログ不要）
                        action_cards = None

                    # external_action の場合、矯正が発生したことを記録
                    if external_action is not None:
                        corrected_external = True

        # --- [REAL LOG] ---
        if force_action is None and external_action is None:
            # print(f"[REAL LOG] turn={self.turn_idx} player={current_player_id} action={action_cards} field_before={[str(c) for c in field]}")
            pass

        # --- プレイ実行 ---
        obs_, done, reset_happened, reset_reason = self.game.step(current_player_id, action_cards)
        # reset_reason は "eight_cut" | "joker_cut" | "all_pass" を取りうる
        self.done = self.game.done
        new_field = self.game.current_field[:]

        if reset_happened:
            if reset_reason == "eight_cut":
                # 8切りの場合、出したプレイヤーが続行
                # print(f"[LOG] 8切りによる場リセット - 続行プレイヤー: {current_player_id}")
                pass
            elif reset_reason == "joker_cut":
                # ジョーカー単出し流しの場合も、出したプレイヤーが続行
                # print(f"[LOG] ジョーカー流し - 続行プレイヤー: {current_player_id}")
                pass
            elif reset_reason == "all_pass":
                # 全員パスの場合、最後に出したプレイヤーから続行
                # print(f"[LOG] 全員パスによる場リセット - 続行プレイヤー: {self.game.turn}")
                pass

            # print(f"[LOG] reset_happened at turn={self.turn_idx} by player={current_player_id} action={action_cards}")
            # print(f"[LOG] Field reset - next turn will be player {self.game.turn}")
            pass

        # --- [REAL LOG] ---
        if force_action is None and external_action is None:
            # print(f"[REAL LOG] field_after={[str(c) for c in new_field]} reset_happened={reset_happened}")
            pass

        # obs更新
        obs = self._get_obs()

        # --- 各手番の履歴記録 ---
        others_hand_counts = [len(self.game.players[i].hand) for i in range(self.num_players)]
        is_revolution = getattr(self.game.rule_checker, 'revolution', False)
        field_type = 'empty'
        if field:
            if rule_checker.is_straight(field):
                field_type = 'straight'
            elif len(field) >= 2 and all((c.rank == field[0].rank or c.is_joker) for c in field):
                field_type = 'pair'
            else:
                field_type = 'single'

        legal_actions_list = [[str(c) for c in action] if action is not None else None for action in legal_actions]
        action_taken = [str(c) for c in action_cards] if action_cards is not None else None
        remaining_players = [i for i in range(self.num_players) if i not in self.already_won_players]
        already_won = set(self.already_won_players)
        step_idx_in_stage = len(self.stage_history)

        step_record = {
            'game_id': None,
            'stage_id': self.stage_id,
            'turn_idx': self.turn_idx,
            'step_idx_in_stage': step_idx_in_stage,
            'player_id': current_player_id,
            'remaining_players': remaining_players,
            'already_won': already_won,
            'obs': {
                'hand': [str(c) for c in hand],
                'field': [str(c) for c in field],
                'revolution': is_revolution,
                'others_hand_counts': others_hand_counts,
                'field_type': field_type
            },
            'legal_actions': legal_actions_list,
            'legal_actions_mask': None,
            'policy_target': mcts_result['policy_target'] if mcts_result else None,
            'action_taken': action_taken,
            'value_target': None,
            'value_weight': None,
            'is_terminal_in_stage': False,
            'stage_winner': None,
            'mcts_root_value': mcts_result['mcts_root_value'] if mcts_result else None,
            'mcts_visits': mcts_result['mcts_visits'] if mcts_result else None,
            'exploration_meta': None,
            'reason_tag': None
        }
        self.stage_history.append(step_record)
        self.turn_idx += 1

        # --- 区間終了時の報酬付与 ---
        reward = 0.0
        new_winners = [pid for pid in self.game.rankings if pid not in self.already_won_players]
        if new_winners:
            winner_id = new_winners[0]
            self.assign_stage_rewards(self.stage_history, winner_id, self.already_won_players)
            reward = self.stage_history[-1]['value_target']
            self.already_won_players.update(new_winners)
            self.stage_id += 1
            self.stage_history = []

        if return_info:
            info = {
                "player_id": player.player_id,
                "played_cards": action_cards,
                "reset_happened": reset_happened,
                "reset_reason": reset_reason,
                "field_after_play": [str(c) for c in new_field]
            }
            # expose correction info for external_action strict checking
            try:
                info['corrected_external_action'] = bool(corrected_external)
                info['external_action_before'] = corrected_from
            except Exception:
                pass
            # attach model outputs if available (best-effort)
            try:
                if 'pol' in locals() and pol is not None:
                    info['policy'] = pol
                if 'val' in locals() and val is not None:
                    # value may be scalar/list/dict; try to coerce to float when scalar
                    info['value'] = val
            except Exception:
                pass
            return obs, reward, self.done, info
        else:
            # Shibari statistics (ゲーム終了時のみログ出力)
            # shibari 統計はワーカー/上位ロジック側でまとめて出力するため、ここではログ出力しない
            # ログ出力が必要な場合はワーカーが `env.game` の統計を収集してまとめて出力する。
            return obs, reward, self.done
        
        

    def assign_stage_rewards(self, stage_history, winner_id, already_won_players):
        """
        区間内の全ステップに対して、次に上がった人だけ1.0、それ以外の残っていた人は0.0、既に上がっていた人は評価外(None)を付与
        value_weight, reason_tag, is_terminal_in_stage, stage_winnerも付与
        """
        n = len(stage_history)
        for i, step in enumerate(stage_history):
            pid = step['player_id']
            if pid == winner_id:
                step['value_target'] = 1.0
                step['reason_tag'] = 'winner_in_stage'
            elif pid not in already_won_players:
                step['value_target'] = 0.0
                step['reason_tag'] = 'not_winner'
            else:
                step['value_target'] = None  # 評価外
                step['reason_tag'] = 'already_won'
            step['value_weight'] = 1.0 / n if n > 0 else 1.0
            step['stage_winner'] = winner_id
            step['is_terminal_in_stage'] = (i == n - 1)

    # カード情報を数値に変換
    def _encode_card(self, card):
        if card is None:
            return -1 # 場が空の場合は -1
        try:
            # Card オブジェクト想定
            if getattr(card, 'is_joker', False):
                return 53
            suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
            s = getattr(card, 'suit', None)
            r = getattr(card, 'rank', None)
            if s in suit_map and isinstance(r, int):
                return suit_map[s] * 13 + (r - 1)
        except Exception:
            pass
        # 文字列 fallback: 例 '♦7'
        try:
            if isinstance(card, str):
                if card.lower() == 'joker':
                    return 53
                suit = card[0]
                rank_part = card[1:]
                suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
                if suit in suit_map:
                    rank = int(rank_part)
                    return suit_map[suit] * 13 + (rank - 1)
        except Exception:
            pass
        return -1  # 不明形式

    def _get_obs(self):
        player = self.game.players[self.game.turn]  # 現在のプレイヤー
        # 手札を数値化して長さを27枚に固定（足りない分は -1 で埋める）
        hand_encoded = [self._encode_card(c) for c in player.hand]
        hand_encoded += [-1] * (27 - len(hand_encoded))
        # 現在場に出ているカードを数値化
        field_card = self.game.current_field[-1] if self.game.current_field else None
        field_encoded = self._encode_card(field_card)

        return {
            "hand": np.array(hand_encoded, dtype=np.int32),
            "field": field_encoded
        }
