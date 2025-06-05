class Player:
    def __init__(self, player_id, name=None):
        self.player_id = player_id               # ← player_id 追加
        self.name = name if name else f"P{player_id}"
        self.hand = []

    def draw_hand(self, deck, n):
        self.hand = sorted(
            [deck.pop() for _ in range(n)],
            key=lambda c: (c.rank if not c.is_joker else 100)
        )

    def play(self, cards):  # cards: list[Card]
        for card in cards:
            self.hand.remove(card)
        return cards

    def has_playable(self, current):
        if not current:
            return True  # 場が空ならなんでも出せる

        rank_count = {}
        for card in self.hand:
            rank = card.rank if not card.is_joker else -1
            rank_count.setdefault(rank, []).append(card)

        for cards in rank_count.values():
            for k in range(1, len(cards) + 1):
                candidate = cards[:k]
                if self._is_valid_set(candidate, current):
                    return True
        return False

    def _is_valid_set(self, cards, current):
        # すべて同ランクか
        if any(c.rank != cards[0].rank or c.is_joker != cards[0].is_joker for c in cards):
            return False

        if not current:
            return True

        # 枚数が同じ、かつランクが高い必要あり
        if len(cards) != len(current):
            return False
        return cards[0].rank > current[0].rank
