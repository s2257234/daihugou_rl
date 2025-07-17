import random

class StraightAgent:
    def __init__(self, player_id=None):
        self.player_id = player_id

    def select_action(self, observation, legal_actions=None):
        if legal_actions:
            # legal_actionsの中から階段優先で選択
            straights = [a for a in legal_actions if a and len(a) >= 3 and self._is_straight(a)]
            if straights:
                return random.choice(straights)
            # ペア・スリーカード
            pairs = [a for a in legal_actions if a and len(a) >= 2 and self._is_pair(a)]
            if pairs:
                return random.choice(pairs)
            # 単体
            singles = [a for a in legal_actions if a and len(a) == 1 and not a[0].is_joker]
            if singles:
                return random.choice(singles)
            # ジョーカー単体
            jokers = [a for a in legal_actions if a and len(a) == 1 and a[0].is_joker]
            if jokers:
                return random.choice(jokers)
            # パス
            if None in legal_actions:
                return None
            return random.choice(legal_actions)
        # legal_actionsがなければ従来通り
        hand = observation['hand']
        field = observation['field'] if 'field' in observation else []
        straights = self._find_straights(hand)
        for straight in straights:
            if self._is_valid_play(straight, field):
                return straight
        rank_map = {}
        for card in hand:
            if card.is_joker:
                continue
            rank_map.setdefault((card.suit, card.rank), []).append(card)
        for cards in rank_map.values():
            if len(cards) >= 2:
                for k in range(len(cards), 1, -1):
                    candidate = cards[:k]
                    if self._is_valid_play(candidate, field):
                        return candidate
        for card in hand:
            if self._is_valid_play([card], field):
                return [card]
        for card in hand:
            if card.is_joker and self._is_valid_play([card], field):
                return [card]
        return None

    def _is_pair(self, cards):
        if not cards or len(cards) < 2:
            return False
        # 全カードが同じランクまたはジョーカー
        rank = None
        for c in cards:
            if not c.is_joker:
                if rank is None:
                    rank = c.rank
                elif c.rank != rank:
                    return False
        return all(c.is_joker or c.rank == rank for c in cards)

    def _find_straights(self, hand):
        from collections import defaultdict
        suit_map = defaultdict(list)
        jokers = []
        for card in hand:
            if card.is_joker:
                jokers.append(card)
            else:
                suit_map[card.suit].append(card)
        straights = []
        for suit, cards_in_suit in suit_map.items():
            ranks = sorted([c.rank for c in cards_in_suit])
            n = len(cards_in_suit) + len(jokers)
            for length in range(5, 2, -1):
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
                        straights.append(temp)
        return straights

    def _is_valid_play(self, cards, field):
        if not cards:
            return False
        if not field:
            return True
        if len(cards) != len(field):
            return False
        if self._is_straight(cards) and self._is_straight(field):
            if cards[0].suit != field[0].suit:
                return False
            play_ranks = [c.rank for c in cards if not c.is_joker]
            field_ranks = [c.rank for c in field if not c.is_joker]
            if not play_ranks or not field_ranks:
                return False
            return max(play_ranks) > max(field_ranks)
        if all(c.rank == cards[0].rank or c.is_joker for c in cards):
            if all(c.rank == field[0].rank or c.is_joker for c in field):
                play_ranks = [c.rank for c in cards if not c.is_joker]
                field_ranks = [c.rank for c in field if not c.is_joker]
                if not play_ranks or not field_ranks:
                    return False
                return max(play_ranks) > max(field_ranks)
        return False

    def _is_straight(self, cards):
        if len(cards) < 3:
            return False
        suits = [c.suit for c in cards if not c.is_joker]
        if len(set(suits)) != 1:
            return False
        ranks = sorted([c.rank for c in cards if not c.is_joker])
        jokers = [c for c in cards if c.is_joker]
        n = len(cards)
        for start in range(1, 14):
            expected = [(start + i - 1) % 13 + 1 for i in range(n)]
            temp_ranks = ranks[:]
            match = 0
            for val in expected:
                if val in temp_ranks:
                    temp_ranks.remove(val)
                    match += 1
            if n - match <= len(jokers):
                return True
        return False 