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

    def get_final_rewards(self):
        """
        エピソード終了時点で全プレイヤーの最終報酬を返す。
        1位: +1.0, それ以外: -1.0
        Returns:
            dict: {player_id: reward, ...}
        """
        rewards = {pid: 0.0 for pid in range(self.num_players)}
        if hasattr(self.game, "rankings") and self.game.rankings:
            for i, pid in enumerate(self.game.rankings):
                if i == 0:
                    rewards[pid] = 1.0
                else:
                    rewards[pid] = -1.0
        return rewards

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
        # 場の役種を判定
        rule_checker = self.game.rule_checker
        is_field_straight = rule_checker.is_straight(field) if field else False
        is_field_pair = False
        if field and not is_field_straight:
            non_jokers = [c for c in field if not c.is_joker]
            if non_jokers and all(c.rank == non_jokers[0].rank or c.is_joker for c in field):
                is_field_pair = True if len(field) >= 2 else False
        # legal_actionsから場の役種に合わせてフィルタ
        filtered_actions = []
        if is_field_straight:
            for action in legal_actions:
                if action is not None and rule_checker.is_straight(action):
                    filtered_actions.append(action)
            if not filtered_actions:
                filtered_actions = [None]
        elif is_field_pair:
            for action in legal_actions:
                if action is not None and rule_checker.is_same_rank_or_joker(action) and len(action) == len(field):
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
        obs = self._get_obs()
        # 報酬設計（例: 上がりで+1, それ以外0）
        reward = self._calc_reward(player, done, reset_happened)
        if return_info:
            # プレイヤーIDと出したカードの情報も返す
            return obs, reward, self.done, {
                "player_id": player.player_id,
                "played_cards": action_cards,
                "reset_happened": reset_happened
            }
        else:
            return obs, reward, self.done

    def _calc_reward(self, player, done, reset_happened):
        """
        1位（最初に上がったプレイヤー）のみ+1.0、
        それ以外の順位で上がった場合は-1.0、
        途中経過は0.0
        """
        if done and player.player_id in self.game.rankings:
            if self.game.rankings[0] == player.player_id:
                return 1.0
            else:
                return -1.0
        return 0.0

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
