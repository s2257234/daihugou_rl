class Player:
    def __init__(self, player_id, name=None):
        self.player_id = player_id  # プレイヤーID（識別用）
        self.name = name if name else f"P{player_id}"  # 名前（指定がなければ"P0", "P1"などになる）
        self.hand = []  # プレイヤーの手札（Cardオブジェクトのリスト）

    def draw_hand(self, deck, n):
        """
        山札からカードをn枚引いて手札にする。
        ジョーカーは最後に来るように、ランク順にソート。
        """
        self.hand = sorted(
            [deck.pop() for _ in range(n)],
            key=lambda c: (c.rank if not c.is_joker else 100)  # ジョーカーは一番後ろにするためrank=100として扱う
        )

    def play(self, cards):  # cards: list[Card]
        """
        渡されたカードリストを手札から取り除き、出すカードとして返す。
        """
        for card in cards:
            self.hand.remove(card)  # 手札から指定カードを削除
        return cards

    def has_playable(self, current):
        """
        現在の場に出せるカードが手札に存在するかを判定。
        """
        if not current:
            return True  # 場が空ならなんでも出せる

        # ランクごとにカードを分類
        rank_count = {}
        for card in self.hand:
            rank = card.rank if not card.is_joker else -1  # ジョーカーは特殊扱い（rank=-1）
            rank_count.setdefault(rank, []).append(card)

        # 各ランクのカードから出せる組み合わせがあるかを探索
        for cards in rank_count.values():
            for k in range(1, len(cards) + 1):  # 1枚〜そのランクの枚数まで
                candidate = cards[:k]
                if self._is_valid_set(candidate, current):  # 出せるセットならTrue
                    return True
        return False  # 出せるカードがない

    def _is_valid_set(self, cards, current):
        """
        同ランクセットとして有効か、現在の場に出せるかを判定。
        """
        # 全カードが同じランク・ジョーカー種別でなければ無効
        if any(c.rank != cards[0].rank or c.is_joker != cards[0].is_joker for c in cards):
            return False

        if not current:
            return True  # 場が空なら出せる

        # 枚数が異なる場合は無効
        if len(cards) != len(current):
            return False

        # 出そうとしているカードのランクが場より高いなら出せる
        return cards[0].rank > current[0].rank
