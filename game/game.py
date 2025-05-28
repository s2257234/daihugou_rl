import random
from .card import CardDeck
from .player import Player
from .rules import RuleChecker 

class Game:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [Player(i) for i in range(num_players)]
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

    # game.py または rules.py に追加（例）
    def is_valid_play(self, card):
        """現在の場にこのカードが出せるかどうか"""
        if card is None:
            return True  # パスは常に有効
        return self.rule_checker.is_valid(self.current_field, card)  # ルール判定


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

    def step(self, player_id, action_card):
        player = self.players[player_id]
        player = self.players[self.turn]
        valid = False

        if action_card is not None:
            # 出そうとしたカードが手札にあるかチェック
            for card in player.hand:
                if str(card) == str(action_card):
                    if self.rule_checker.is_valid_move(card, self.current_field[-1] if self.current_field else None):
                        self.current_field = [card]
                        player.hand.remove(card)
                        self.passed = [False] * self.num_players
                        valid = True

                        # ジョーカーを出したら場をリセット（流れる）＝次のターンへ
                    if card.is_joker:
                        self.just_played_card = card 
                        self.field_was_reset = True   # 場が流れたかを記録
                        self.current_field = []
                        self.passed = [False] * self.num_players
                    break

        if not valid:
            self.passed[self.turn] = True
        else:
            self.last_player = self.turn

        # ゲーム終了条件：1人が上がる
        if len(player.hand) == 0:
            self.done = True
            #return self.get_state(self.turn), 1.0, True  # 勝利したプレイヤーに報酬
            if player_id not in self.rankings:
                self.rankings.append(player_id)
            return self.get_state(self.turn), 1.0, True  # 勝利したプレイヤーに報酬

        if len(self.rankings) == self.num_players - 1:
            last_player = [i for i in range(self.num_players) if i not in self.rankings][0]
            self.rankings.append(last_player)
            self.done = True


        # 場が流れる（全員パス）
        if all(self.passed[i] or len(self.players[i].hand) == 0 for i in range(self.num_players)):
            self.current_field = []
            self.passed = [False] * self.num_players
            if self.last_player is not None:
                self.turn = self.last_player
            else:
                self.turn = (self.turn + 1) % self.num_players
            self.turn_count += 1 
        
        # プレイヤーがカードを出したあとに手札が空なら勝ち抜け
        if len(self.players[player_id].hand) == 0 and player_id not in self.rankings:
            self.rankings.append(player_id)


        # ターンを進める
        next_turn = (self.turn + 1) % self.num_players
        while len(self.players[next_turn].hand) == 0:
            next_turn = (next_turn + 1) % self.num_players
            

        # ターンが変わるときにカウント
        if next_turn != self.turn:
            self.turn_count += 1
                    
        self.turn = next_turn

        

        return self.get_state(self.turn), 0.0, False  # 通常のステップ
