import random

# 使用するスート（♠, ♥, ♦, ♣）
SUITS = ['♠', '♥', '♦', '♣']

# 使用するランク（1〜13＝A〜K）
RANKS = list(range(1, 14))

# カード1枚を表すクラス
class Card:
    def __init__(self, suit=None, rank=None, is_joker=False):
        self.suit = suit        # スート（♠, ♥, ♦, ♣）
        self.rank = rank        # 数字（1〜13）
        self.is_joker = is_joker  # ジョーカーかどうかのフラグ（True/False）

    def __repr__(self):
        # カードの文字列表現を返す（例：♠A、♦10、JOKER）
        if self.is_joker:
            return 'JOKER'
        rank_str = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(self.rank, str(self.rank))
        return f'{self.suit}{rank_str}'

    def __eq__(self, other):
        # --- ここからが修正箇所 ---
        # 比較対象(other)がCardインスタンスでない場合は、Falseを返す
        if not isinstance(other, Card):
            return False
        # --- 修正箇所ここまで ---

        # 元の比較ロジック（スート・ランク・ジョーカー属性が同じならTrue）
        # このロジックは、otherがCardインスタンスであればジョーカーも正しく扱えます
        return (
            self.suit == other.suit and
            self.rank == other.rank and
            self.is_joker == other.is_joker
        )

# トランプのデッキを表すクラス（ジョーカー2枚を含む）
class CardDeck:
    def __init__(self):
        # 通常カード（52枚）＋ジョーカー（2枚）＝54枚のカードを生成
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        self.cards.append(Card(is_joker=True))
        self.cards.append(Card(is_joker=True))

    # デッキをシャッフルする
    def shuffle(self):
        random.shuffle(self.cards)