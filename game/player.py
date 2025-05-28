class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def draw_hand(self, deck, n):
        self.hand = sorted([deck.pop() for _ in range(n)], key=lambda c: (c.rank if not c.is_joker else 100))

    def play(self, card):
        self.hand.remove(card)
        return card

    def has_playable(self, current):
        if not current:
            return True
        return any(c.is_joker or c.rank >= current.rank for c in self.hand)

