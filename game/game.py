import random
from .card import CardDeck
from .player import Player
from .rules import RuleChecker 

class Game:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [Player(player_id=i) for i in range(num_players)]
        self.deck = CardDeck()
        self.rule_checker = RuleChecker()
        self.current_field = []  # 場に出ているカード（最後に出されたカード）
        self.turn = 0
        self.turn_count = 0
        self.passed = [False] * num_players
        self.done = False
        self.last_player = None # 最後にカードを出したプレイヤー
        self.rankings = []
        self._deal_cards()

    def reset(self):
        self.current_field = []
        self.turn = 0
        self.turn_count = 0
        self.passed = [False] * self.num_players
        self.done = False
        self.last_player = None
        self.rankings = []
        self._deal_cards()

        # ♠3を持っているプレイヤーを探して、その人にターンをセットし、場に♠3を出す
        spade_3 = None
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.suit == '♠' and card.rank == 3:
                    spade_3 = card
                    self.turn = i
                    break
            if spade_3:
                break

        if spade_3:
            self.current_field = [spade_3]
            self.players[self.turn].hand.remove(spade_3)
            self.last_player = self.turn

        return self.get_state(self.turn)

    def is_valid_play(self, cards):
        """現在の場にこのカード群が出せるかどうか"""
        if cards is None or len(cards) == 0:
            return True  # パスは常に有効
        return self.rule_checker.is_valid(self.current_field, cards)  # ルール判定

    def _deal_cards(self):
        self.deck.shuffle()
        for i, card in enumerate(self.deck.cards):
            self.players[i % self.num_players].hand.append(card)

    def get_state(self, player_id):
        return {
            'hand': [str(card) for card in self.players[player_id].hand],
            'field': [str(card) for card in self.current_field],
            'turn': self.turn,
            'passed': self.passed,
            'turn_count': self.turn_count,
        }

    def step(self, player_id, action_cards):
        player = self.players[self.turn]
        valid = False

        if action_cards:
            # 出そうとしたカードがすべて手札にあるかチェック
            card_objs = []
            hand_card_strs = [str(c) for c in player.hand]
            if all(str(card) in hand_card_strs for card in action_cards):
                # 実際のCardオブジェクトを取得
                card_objs = [next(c for c in player.hand if str(c) == str(card)) for card in action_cards]

                if self.rule_checker.is_valid_move(card_objs, self.current_field):
                    self.current_field = card_objs[:]
                    for card in card_objs:
                        player.hand.remove(card)
                    self.passed = [False] * self.num_players
                    valid = True

                    # 8切り判定
                    if self.rule_checker.is_8cut(card_objs):
                        print(f"8切り発動 by Player {self.turn}!")
                        self.current_field = []  # 場を流す
                        self.passed = [False] * self.num_players
                        self.last_player = self.turn
                        # ターン継続のためここで戻る（このプレイヤーがもう一度手番）
                        return self.get_state(self.turn), 0.0, False

                    # ジョーカーが含まれていたら場を流す（従来の処理）
                    if any(card.is_joker for card in card_objs):
                        self.current_field = []
                        self.passed = [False] * self.num_players

        if not valid:
            self.passed[self.turn] = True
        else:
            self.last_player = self.turn

        if len(player.hand) == 0 and player_id not in self.rankings:
            self.rankings.append(player_id)

        if len(self.rankings) == self.num_players - 1:
            last_player = [i for i in range(self.num_players) if i not in self.rankings][0]
            self.rankings.append(last_player)
            self.done = True
            return self.get_state(self.turn), 1.0, True

        if all(self.passed[i] or len(self.players[i].hand) == 0 for i in range(self.num_players)):
            self.current_field = []
            self.passed = [False] * self.num_players
            self.turn = self.last_player if self.last_player is not None else (self.turn + 1) % self.num_players
            self.turn_count += 1

        next_turn = (self.turn + 1) % self.num_players
        while len(self.players[next_turn].hand) == 0:
            next_turn = (next_turn + 1) % self.num_players

        if next_turn != self.turn:
            self.turn_count += 1

        self.turn = next_turn

        return self.get_state(self.turn), 0.0, False
