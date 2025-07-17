import random
import numpy as np
from game.game import Game
from agents.straight_agent import StraightAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
import itertools

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

    def reset(self):
        # ゲームをリセット
        self.game = Game(num_players=self.num_players)
        self.current_player = self.game.turn
        self.done = False
        return self._get_obs()

    def _generate_legal_actions(self, hand, field):
        """
        現在の手札と場の状態から出せる全ての合法なカードセット（legal actions）を列挙する。
        パス(None)も必ず含める。
        場の状態に応じて出せる役種・枚数を限定する。
        """
        import itertools
        rule_checker = self.game.rule_checker
        legal_actions = []
        n = len(hand)
        field_count = len(field)
        # 場の役種判定
        is_field_straight = rule_checker.is_straight(field) if field else False
        is_field_pair = False
        if field and not is_field_straight:
            # 通常出し（同ランク or ジョーカーのみ）
            non_jokers = [c for c in field if not c.is_joker]
            if non_jokers and all(c.rank == non_jokers[0].rank or c.is_joker for c in field):
                is_field_pair = True if len(field) >= 2 else False
        # 場が空
        if not field:
            # 1枚出し
            for card in hand:
                if rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
            # ペア・スリーカード・フォーカード
            rank_map = {}
            for card in hand:
                key = (card.rank, card.suit) if not card.is_joker else ("JOKER", None)
                rank_map.setdefault(card.rank, []).append(card)
            for cards in rank_map.values():
                if len(cards) >= 2:
                    for k in range(2, len(cards)+1):
                        for comb in itertools.combinations(cards, k):
                            if rule_checker.is_valid_move(list(comb), field):
                                legal_actions.append(list(comb))
            # 階段
            suit_map = {}
            jokers = [c for c in hand if c.is_joker]
            for card in hand:
                if not card.is_joker:
                    suit_map.setdefault(card.suit, []).append(card)
            for suit, cards_in_suit in suit_map.items():
                for length in range(3, min(5, len(cards_in_suit) + len(jokers)) + 1):
                    for start in range(1, 15 - length):
                        expected = [(start + i - 1) % 13 + 1 for i in range(length)]
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
                                    temp.append(available_jokers.pop(0))
                                    used_jokers += 1
                                else:
                                    break
                        if len(temp) == length:
                            if rule_checker.is_valid_move(temp, field):
                                legal_actions.append(temp)
            # ジョーカー単体
            for card in hand:
                if card.is_joker and rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
        # 場が階段
        elif is_field_straight:
            suit_map = {}
            jokers = [c for c in hand if c.is_joker]
            for card in hand:
                if not card.is_joker:
                    suit_map.setdefault(card.suit, []).append(card)
            for suit, cards_in_suit in suit_map.items():
                for start in range(1, 15 - field_count):
                    expected = [(start + i - 1) % 13 + 1 for i in range(field_count)]
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
                                temp.append(available_jokers.pop(0))
                                used_jokers += 1
                            else:
                                break
                    if len(temp) == field_count:
                        if rule_checker.is_valid_move(temp, field):
                            legal_actions.append(temp)
        # 場がペア・スリーカード・フォーカード
        elif is_field_pair:
            rank_map = {}
            for card in hand:
                key = (card.rank, card.suit) if not card.is_joker else ("JOKER", None)
                rank_map.setdefault(card.rank, []).append(card)
            for cards in rank_map.values():
                if len(cards) >= field_count:
                    for comb in itertools.combinations(cards, field_count):
                        if rule_checker.is_valid_move(list(comb), field):
                            legal_actions.append(list(comb))
        # 場が1枚出し
        else:
            for card in hand:
                if rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
            for card in hand:
                if card.is_joker and rule_checker.is_valid_move([card], field):
                    legal_actions.append([card])
        # パス
        legal_actions.append(None)
        # 重複除去（カードの等価性で）
        def cardset_key(cardset):
            if cardset is None:
                return (None,)
            return tuple(sorted(str(c) for c in cardset))
        unique = {}
        for action in legal_actions:
            unique[cardset_key(action)] = action
        return list(unique.values())

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
        action_cards = self.agents[current_player_id].select_action(obs, legal_actions=legal_actions)
        # --- ここまで ---
        # プレイ実行（Noneならパス）
        obs_, reward, done, reset_happened = self.game.step(current_player_id, action_cards)
        self.done = self.game.done
        obs = self._get_obs()
        if return_info:
            # プレイヤーIDと出したカードの情報も返す
            return obs, reward, self.done, {
                "player_id": player.player_id,
                "played_cards": action_cards,
                "reset_happened": reset_happened
            }
        else:
            return obs, reward, self.done

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
