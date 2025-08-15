import random
import numpy as np
from game.game import Game
from agents.straight_agent import StraightAgent
from game.card import Card



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
        for rank, cards in rank_map.items():
            for k in range(field_count, min(len(cards) + len(jokers), 4) + 1):
                for comb in itertools.combinations(cards, min(len(cards), k)):
                    needed_jokers = k - len(comb)
                    if needed_jokers <= len(jokers):
                        pair = list(comb)
                        # ジョーカーを代用として追加
                        for i in range(needed_jokers):
                            joker_card = Card(is_joker=True)
                            joker_card.set_joker_substitute(rank, pair[0].suit)
                            pair.append(joker_card)
                        if rule_checker.is_valid_move(pair, field):
                            legal_actions.append(pair)
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
                if 2 in expected and expected[-1] != 2:
                    continue
                temp = []
                used_jokers = 0
                available_cards = cards_in_suit[:]
                available_jokers = jokers[:]
                for val in expected:
                    found = False
                    for i, c in enumerate(available_cards):
                        if c.rank == val:
                            temp.append(available_cards.pop(i))
                            found = True
                            break
                    if not found:
                        if used_jokers < len(available_jokers):
                            joker_card = available_jokers.pop(0)
                            joker_card = Card(is_joker=True)
                            joker_card.set_joker_substitute(val, suit)
                            temp.append(joker_card)
                            used_jokers += 1
                        else:
                            break
                if len(temp) == field_count:
                    temp_sorted = []
                    for v in expected:
                        for c in temp:
                            rank = c.joker_as_rank if c.is_joker else c.rank
                            if rank == v:
                                temp_sorted.append(c)
                                break
                    if rule_checker.is_valid_move(temp_sorted, field):
                        legal_actions.append(temp_sorted)
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
        パス(None)も必ず含める。
        場の状態に応じて出せる役種・枚数を限定する。
        """
        rule_checker = self.game.rule_checker
        legal_actions = []
        field_count = len(field)
        jokers = [c for c in hand if c.is_joker]
        is_field_straight = rule_checker.is_straight(field) if field else False
        is_field_pair = False
        if field and not is_field_straight:
            non_jokers = [c for c in field if not c.is_joker]
            if non_jokers and all(c.rank == non_jokers[0].rank or c.is_joker for c in field):
                is_field_pair = True if len(field) >= 2 else False

        if not field:
            # 1枚出し
            for card in hand:
                if rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
            # ペア・スリーカード・フォーカード
            legal_actions += self._make_pair_sets(hand, jokers, 2, rule_checker, field)
            # 階段
            for length in range(3, 6):
                legal_actions += self._make_straight_sets(hand, jokers, length, rule_checker, field)
            # ジョーカー単体
            for card in hand:
                if card.is_joker and rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
        elif is_field_straight:
            legal_actions += self._make_straight_sets(hand, jokers, field_count, rule_checker, field)
        elif is_field_pair:
            legal_actions += self._make_pair_sets(hand, jokers, field_count, rule_checker, field)
        else:
            for card in hand:
                if rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
            for card in hand:
                if card.is_joker and rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
        legal_actions.append(None)
        return self._remove_duplicate_actions(legal_actions)

    def step(self, return_info=False):
        # 現在のターンのプレイヤーIDを保存
        current_player_id = self.game.turn
        player = self.game.players[current_player_id]
        hand = player.hand
        field = self.game.current_field[:]
        # legal_actions生成
        legal_actions = self._generate_legal_actions(hand, field)
        # --- ここからエージェントによる行動選択 ---
        obs = {
            'hand': hand,
            'field': field
        }
        rule_checker = self.game.rule_checker
        is_field_straight = rule_checker.is_straight(field) if field else False
        is_field_pair = False
        if field and not is_field_straight:
            non_jokers = [c for c in field if not c.is_joker]
            if non_jokers and all(c.rank == non_jokers[0].rank or c.is_joker for c in field):
                is_field_pair = True if len(field) >= 2 else False
        filtered_actions = []
        if is_field_straight:
            for action in legal_actions:
                if action is not None and rule_checker.is_straight(action):
                    filtered_actions.append(action)
            if not filtered_actions:
                filtered_actions = [None]
        elif is_field_pair:
            for action in legal_actions:
                if (
                    action is not None
                    and rule_checker.is_same_rank_or_joker(action)
                    and len(action) == len(field)
                    and rule_checker.is_valid_move(action, field)
                ):
                    filtered_actions.append(action)
            if not filtered_actions:
                filtered_actions = [None]
        else:
            filtered_actions = legal_actions
        action_cards = self.agents[current_player_id].select_action(obs, legal_actions=filtered_actions)
        # --- ここまで ---
        # プレイ実行（Noneならパス）
        obs_, done, reset_happened = self.game.step(current_player_id, action_cards)
        self.done = self.game.done
        # プレイ後の最新の場を取得
        new_field = self.game.current_field[:]
        # legal_actions/filtered_actionsを再生成（次のプレイヤーのための状態管理用）
        # obsも再取得
        obs = self._get_obs()

        # --- 各手番のレコード生成 ---
        # 各相手の残り枚数
        others_hand_counts = [len(self.game.players[i].hand) for i in range(self.num_players)]
        # 革命フラグ
        is_revolution = getattr(self.game.rule_checker, 'revolution', False)
        # 場の役種
        field_type = 'empty'
        if field:
            if rule_checker.is_straight(field):
                field_type = 'straight'
            elif len(field) >= 2 and all((c.rank == field[0].rank or c.is_joker) for c in field):
                field_type = 'pair'
            else:
                field_type = 'single'
        # 合法手（カード集合のリスト）
        legal_actions_list = [[str(c) for c in action] if action is not None else None for action in legal_actions]
        # 選択行動（カード集合のリスト）
        action_taken = [str(c) for c in action_cards] if action_cards is not None else None
        # 区間開始時点のremaining_players, already_won
        remaining_players = [i for i in range(self.num_players) if i not in self.already_won_players]
        already_won = set(self.already_won_players)
        # 区間内の手番番号
        step_idx_in_stage = len(self.stage_history)
        # レコード生成
        step_record = {
            'game_id': None,  # ゲーム識別子（後で付与）
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
            'legal_actions_mask': None,  # MCTS未実装なのでNone
            'policy_target': None,  # MCTS未実装なのでNone
            'action_taken': action_taken,
            'value_target': None,  # 後で一括付与
            'value_weight': None,  # 後で一括付与
            'is_terminal_in_stage': False,  # 後で一括付与
            'stage_winner': None,  # 後で一括付与
            'mcts_root_value': None,  # MCTS未実装なのでNone
            'mcts_visits': None,  # MCTS未実装なのでNone
            'exploration_meta': None,  # MCTS未実装なのでNone
            'reason_tag': None  # 後で一括付与
        }
        self.stage_history.append(step_record)
        self.turn_idx += 1

        # --- 誰かが上がったら区間の全履歴に一括で報酬付与 ---
        reward = 0.0
        new_winners = [pid for pid in self.game.rankings if pid not in self.already_won_players]
        if new_winners:
            winner_id = new_winners[0]
            self.assign_stage_rewards(self.stage_history, winner_id, self.already_won_players)
            # このstepのvalue_targetを履歴から取得
            reward = self.stage_history[-1]['value_target']
            # デバッグ出力
            print(f"[DEBUG] 区間終了: winner={winner_id}, already_won={self.already_won_players}")
            for i, step in enumerate(self.stage_history):
                print(f"  [DEBUG] step{i}: player={step['player_id']} value_target={step['value_target']} value_weight={step['value_weight']} reason_tag={step['reason_tag']}")
            # 区間終了後、履歴をリセットし、すでに上がった人を更新
            self.already_won_players.update(new_winners)
            self.stage_id += 1
            self.stage_history = []
        else:
            # 誰も上がっていなければrewardは0.0
            reward = 0.0

        if return_info:
            return obs, reward, self.done, {
                "player_id": player.player_id,
                "played_cards": action_cards,
                "reset_happened": reset_happened,
                "field_after_play": [str(c) for c in new_field]
            }
        else:
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
        if card.is_joker:
            return 53 # ジョーカーは 53 としてエンコード
        suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
        return suit_map[card.suit] * 13 + (card.rank - 1)

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
