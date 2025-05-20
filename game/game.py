#-------実際のゲーム進行ロジック-------

from .card import Deck
from .rules import is_valid_play

class Game:
    def __init__(self, players):
        self.players = players
        self.revolution = False
        self.field = []
        self.turn = 0
        self.history = []

    def deal_cards(self):
        deck = Deck()
        hands = deck.deal(len(self.players))
        for i, player in enumerate(self.players):
            player.receive_cards(hands[i])

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    def is_game_over(self):
        return all(len(p.hand) == 0 or p.passed for p in self.players)

    def play_game(self):
        self.deal_cards()
        while not self.is_game_over():
            current_player = self.players[self.turn]
            if current_player.passed:
                self.next_turn()
                continue
            play = current_player.play(self.field)
            if play and is_valid_play(play, self.field, self.revolution):
                self.field = play
                self.history.append((current_player.id, play))
            else:
                current_player.passed = True
            self.next_turn()
