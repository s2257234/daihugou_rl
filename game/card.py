#--------カードや山札の定義-----------------
from enum import Enum
import random

class Suit(Enum):
    CLUB = "♣"
    DIAMOND = "♦"
    HEART = "♥"
    SPADE = "♠"
    JOKER = "JOKER"

class Card:
    def __init__(self, suit: Suit, rank: int):
        self.suit = suit
        self.rank = rank  # 1: A, 2〜10, 11: J, 12: Q, 13: K, 0: Joker

    def __repr__(self):
        if self.suit == Suit.JOKER:
            return "JOKER"
        rank_str = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit.value}"

class Deck:
    def __init__(self):
        # スート CLUB, DIAMOND, HEART, SPADE それぞれに1〜13のカードを作成（合計52枚）
        self.cards = [Card(suit, rank) for suit in list(Suit)[:-1] for rank in range(1, 14)]

        # ジョーカーを2枚追加（rankは0として表現）
        self.cards.append(Card(Suit.JOKER, 0))
        self.cards.append(Card(Suit.JOKER, 0))

        random.shuffle(self.cards)

    def deal(self, num_players):
        """ プレイヤー数に応じてカードを均等に配る（余りがあれば順に多く配る）"""
        return [self.cards[i::num_players] for i in range(num_players)]
