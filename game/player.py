#------プレイヤークラス-----
class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.hand = []
        self.passed = False

    def receive_cards(self, cards):
        self.hand = sorted(cards, key=lambda c: (c.rank, c.suit.value))

    def play(self, current_field):
        # ダミー：必ず1枚出す（強化学習後で差し替え）
        if self.hand:
            return [self.hand.pop(0)]
        self.passed = True
        return []

    def reset(self):
        self.passed = False
