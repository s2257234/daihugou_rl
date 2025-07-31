import random

# 使用するスート（♠, ♥, ♦, ♣）
SUITS = ['♠', '♥', '♦', '♣']

# 使用するランク（1〜13＝A〜K）
RANKS = list(range(1, 14))

# 大富豪におけるカードの強さ順を定義（比較のため）
DAIFUGO_STRENGTH = {
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 9, 12: 10, 13: 11, 1: 12, 2: 13  # 1=A, 11=J, 12=Q, 13=K
}

# カード1枚を表すクラス
class Card:
    def __init__(self, suit=None, rank=None, is_joker=False):
        self.suit = suit  # スート（♠, ♥, ♦, ♣）
        self.rank = rank  # ランク（1〜13、またはNone）
        self.is_joker = is_joker
        # ジョーカーが何の代用か（ペアや階段で使うとき一時的にセット）
        self.joker_as_suit = None
        self.joker_as_rank = None

    def strength(self):
        """
        大富豪ルールに基づくカードの強さを返す。
        Jokerは最強（15）、2が次に強く、3が最弱（1）
        ただし、joker_as_rankがセットされていればその強さを返す
        """
        if self.is_joker:
            if self.joker_as_rank is not None:
                # 代用するカードの強さ
                if self.joker_as_rank == 1:
                    return 13
                elif self.joker_as_rank == 2:
                    return 14
                else:
                    return self.joker_as_rank - 2
            return 15  # Joker単体は最強
        elif self.rank == 1:  # A
            return 13
        elif self.rank == 2:
            return 14
        else:
            return self.rank - 2  # 3→1, 4→2, ..., K→12

    def set_joker_substitute(self, rank, suit):
        """
        ジョーカーを特定のランク・スートの代用として使う場合にセットする
        """
        self.joker_as_rank = rank
        self.joker_as_suit = suit

    def reset_joker_substitute(self):
        """
        ジョーカーの代用情報をリセット（手札に戻す場合など）
        """
        self.joker_as_rank = None
        self.joker_as_suit = None

    def __lt__(self, other):
        # カードの強さに基づいて比較
        return self.strength() < other.strength()

    def __repr__(self):
        #カードの表示形式
        if self.is_joker:
            if self.joker_as_suit is not None and self.joker_as_rank is not None:
                rank_str = { 1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(self.joker_as_rank, str(self.joker_as_rank))
                return f'JOKER({self.joker_as_suit}{rank_str})'
            return 'JOKER'
        rank_str = { 1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(self.rank, str(self.rank))
        return f'{self.suit}{rank_str}'

    def __eq__(self, other):
        # カードの等価性を比較
        if not isinstance(other, Card):
            return False
        return (
            self.suit == other.suit and
            self.rank == other.rank and
            self.is_joker == other.is_joker and
            self.joker_as_suit == other.joker_as_suit and
            self.joker_as_rank == other.joker_as_rank
        )

# トランプのデッキを表すクラス（ジョーカー2枚を含む）
class CardDeck:
    
    def __init__(self):
        # 通常カード＋ジョーカー1枚を生成
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        self.cards.append(Card(is_joker=True))

    def shuffle(self):
        random.shuffle(self.cards)
