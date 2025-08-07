import random
from .card import CardDeck
from .player import Player
from .rules import RuleChecker 

# -----------------------------
# 大富豪のゲーム本体クラス
# -----------------------------
class Game:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [Player(player_id=i) for i in range(num_players)]
        self.deck = CardDeck() # トランプのデッキを生成
        self.rule_checker = RuleChecker()  # ルールチェッカーを用意
        self.current_field = []  # 場に出ているカード（最後に出されたカード）
        self.turn = 0  # 現在のプレイヤー番号
        self.turn_count = 0  # ターン数
        self.passed = [False] * num_players
        self.done = False  # ゲーム終了フラグ
        self.last_player = None # 最後にカードを出したプレイヤー
        self.rankings = []  # 上がった順に記録するリスト
        self._deal_cards()  # カードを配る

    def reset(self):
        """ゲームを初期状態にリセットする"""
        # まず手札をリセット
        for player in self.players:
            player.hand = []
        self.current_field = []
        self.turn = 0
        self.turn_count = 0
        self.passed = [False] * self.num_players
        self.done = False
        self.last_player = None
        # 前回の順位情報を一時保存
        prev_rankings = self.rankings[:] if hasattr(self, 'rankings') else []
        self.rankings = []
        self._deal_cards()  # 新しい手札を配る
        self.rule_checker.reset_revolution()  # 革命状態もリセット
        # 新しい手札が配られた後にカード交換を実施
        if prev_rankings and len(prev_rankings) == self.num_players:
            self.rule_checker.exchange_cards_by_rankings(self.players, prev_rankings)

        # ゲーム開始時はダイヤの3を持つ人が最初の権利を持つ（必ずしもダイヤの3を出す必要はない）
        diamond3_player = None
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.suit == '♢' and card.rank == 3:
                    diamond3_player = i
                    break
            if diamond3_player is not None:
                break

        if diamond3_player is not None:
            self.turn = diamond3_player
        # 場は空のまま、ダイヤ3を持つ人から自由に1枚出しでスタート
        return self.get_state(self.turn)  # 最初の状態を返す

    def is_valid_play(self, cards):
        """現在の場にこのカード群が出せるかどうか"""
        if cards is None or len(cards) == 0:
            return True  # パスは常に有効
        return self.rule_checker.is_valid(self.current_field, cards)  # ルール判定

    def _deal_cards(self):
        """山札をシャッフルしてプレイヤーにカードを配る"""
        self.deck.shuffle()
        for i, card in enumerate(self.deck.cards):
            self.players[i % self.num_players].hand.append(card)

    def get_state(self, player_id):
        if player_id is None:
            player_id = self.turn
        return {
            'hand': [str(card) for card in self.players[player_id].hand],
            'field': [str(card) for card in self.current_field],
            'turn': self.turn,
            'passed': self.passed,
            'turn_count': self.turn_count,
        }

    def step(self, player_id, action_cards):
        """
        1ターン進める。action_cards: 出すカードリスト or None（パス）
        戻り値: (状態, 報酬, 終了フラグ, 場リセットフラグ)
        """
        player = self.players[self.turn]
        valid = False
        reset_happened = False  # 場リセットフラグ

        # --- 場が空（リセット直後 or ゲーム開始直後） ---
        if not self.current_field:
            # 直前のプレイヤーがいない（=最初の1ターン目）のときだけ自由に出せる
            is_first_turn = self.turn_count == 0 and self.last_player is None

            if action_cards:
                card_objs = self._find_hand_cards(player, action_cards)
                if card_objs:
                    # 最初の出し手はなんでも出せる、それ以外は current_field が空でも形はチェックする
                    if is_first_turn or self.rule_checker.is_valid_move(card_objs, self.current_field):
                        self._play_cards(player, card_objs)
                        valid = True
                        self.last_player = self.turn

                        # 革命・8切り・階段等の特殊処理
                        if self.rule_checker.check_revolution(card_objs):
                            print(f"革命発生! 現在の革命状態: {self.rule_checker.revolution}")
                        if self.rule_checker.is_straight(card_objs):
                            print(f"Player {self.turn} が階段を出しました: {[str(c) for c in card_objs]}")
                        if self.rule_checker.is_8cut(card_objs):
                            print(f"8切り発動 by Player {self.turn}!")
                            self.last_player = self.turn
                            self._reset_field()
                            reset_happened = True
                            return self.get_state(self.turn), 0.0, False, reset_happened
                        if any(card.is_joker for card in card_objs):
                            self.last_player = self.turn
                            self._reset_field()
                            reset_happened = True
                            return self.get_state(self.turn), 0.0, False, reset_happened

            if not valid:
                action_cards = None
                self.passed[self.turn] = True

            # 上がり判定
            if len(player.hand) == 0 and player_id not in self.rankings:
                self.rankings.append(player_id)
            if len(self.rankings) == self.num_players - 1:
                last_player = [i for i in range(self.num_players) if i not in self.rankings][0]
                self.rankings.append(last_player)
                self.done = True
                return self.get_state(self.turn), 1.0, True, False

            # リセット直後は再度 same player に戻る
            if reset_happened:
                return self.get_state(self.turn), 0.0, False, True

            self._advance_turn()
            return self.get_state(self.turn), 0.0, False, False

        # --- 場が空でない（通常ターン） ---
        if action_cards:
            card_objs = self._find_hand_cards(player, action_cards)
            # 場の枚数と同じ枚数でなければ必ず無効
            if not card_objs or len(card_objs) != len(self.current_field):
                valid = False
            elif self.rule_checker.is_valid_move(card_objs, self.current_field):
                self._play_cards(player, card_objs)
                valid = True
                # 革命・8切り・階段等の特殊処理
                if self.rule_checker.check_revolution(card_objs):
                    print(f"革命発生! 現在の革命状態: {self.rule_checker.revolution}")
                if self.rule_checker.is_straight(card_objs):
                    print(f"Player {self.turn} が階段を出しました: {[str(c) for c in card_objs]}")
                if self.rule_checker.is_8cut(card_objs):
                    print(f"8切り発動 by Player {self.turn}!")
                    self.last_player = self.turn
                    self._reset_field()
                    reset_happened = True
                    return self.get_state(self.turn), 0.0, False, reset_happened
                if any(card.is_joker for card in card_objs):
                    self.last_player = self.turn
                    self._reset_field()
                    reset_happened = True
                    return self.get_state(self.turn), 0.0, False, reset_happened
                self.last_player = self.turn
        # パス or ルール違反
        if not valid:
            action_cards = None
            self.passed[self.turn] = True
            # 全員パス or 最後に出した人以外全員パス → 場リセット
            others_passed = all(self.passed[i] or len(self.players[i].hand) == 0 for i in range(self.num_players) if i != self.turn)
            if self.last_player == self.turn and others_passed:
                self._reset_field()
                reset_happened = True
                self.turn = self.last_player
                return self.get_state(self.turn), 0.0, False, reset_happened
        else:
            self.last_player = self.turn
        # 上がり判定
        if len(player.hand) == 0 and player_id not in self.rankings:
            self.rankings.append(player_id)
        if len(self.rankings) == self.num_players - 1:
            last_player = [i for i in range(self.num_players) if i not in self.rankings][0]
            self.rankings.append(last_player)
            self.done = True
            return self.get_state(self.turn), 1.0, True, False
        # 全員パス or 全員上がりで場リセット
        if all(self.passed[i] or len(self.players[i].hand) == 0 for i in range(self.num_players)):
            self._reset_field()
            reset_happened = True
            self.turn = self.last_player
            return self.get_state(self.turn), 0.0, False, reset_happened
        # 次のプレイヤーへ
        self._advance_turn()
        return self.get_state(self.turn), 0.0, False, False

    # --- 補助メソッド ---
    def _find_hand_cards(self, player, action_cards):
        """手札からaction_cardsに該当するCardオブジェクトリストを返す"""
        hand_card_strs = [str(c) for c in player.hand]
        if all(str(card) in hand_card_strs for card in action_cards):
            return [next(c for c in player.hand if str(c) == str(card)) for card in action_cards]
        return None

    def _play_cards(self, player, card_objs):
        """カードを場に出し、手札から削除し、場の状態を更新"""
        self.current_field = card_objs[:]
        for card in card_objs:
            for h in player.hand:
                if h == card:
                    player.hand.remove(h)
                    break
        self.passed = [False] * self.num_players

    def _reset_field(self):
        """場をリセットし、パス情報もリセット"""
        self.current_field = []
        self.passed = [False] * self.num_players
        self.turn_count += 1
        # 場リセット時は必ず最後に出したプレイヤーから再開
        if self.last_player is not None:
            self.turn = self.last_player
        print("--- 場がリセットされました ---")

    def _advance_turn(self):
        """次のプレイヤーにターンを進める（手札がない場合はスキップ）"""
        next_turn = (self.turn + 1) % self.num_players
        while len(self.players[next_turn].hand) == 0:
            next_turn = (next_turn + 1) % self.num_players
        if next_turn != self.turn:
            self.turn_count += 1
        self.turn = next_turn