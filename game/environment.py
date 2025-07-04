import random
import numpy as np
from game.game import Game

class DaifugoSimpleEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players   #プレイヤーの人数設定
        self.game = Game(num_players=self.num_players)  # Game クラスのインスタンス生成
        self.current_player = self.game.turn  # 現在のプレイヤー番号（ターン）
        self.done = False  # ゲーム終了フラグ

    def reset(self):
        # ゲームをリセット
        self.game = Game(num_players=self.num_players)
        self.current_player = self.game.turn
        self.done = False
        return self._get_obs()

    def step(self, return_info=False):
        # 現在のプレイヤーの手札を取得
        player = self.game.players[self.game.turn]
        hand = player.hand

        # 出せるカードセットを探す（最大4枚まで）
        action_cards = None
        # 1. 同ランクセット（ペア・3枚・4枚）
        for rank in range(1, 14):  # ランク1～13
            same_rank_cards = [card for card in hand if not card.is_joker and card.rank == rank]
            for count in range(4, 0, -1):  #4枚 3枚, 2枚, 1枚の順にチェック
                if len(same_rank_cards) >= count:
                    candidate = same_rank_cards[:count]
                    if self.game.is_valid_play(candidate): #出せるか判定
                        action_cards = candidate
                        break
            if action_cards:
                break

        # 2. 階段（ストレート）セットも探索
        if not action_cards:
            # スートごとにカードを分ける
            suit_map = {'♠': [], '♥': [], '♦': [], '♣': []}
            for card in hand:
                if not card.is_joker:
                    suit_map[card.suit].append(card)
            jokers = [card for card in hand if card.is_joker]
            for suit, cards_in_suit in suit_map.items():
                if len(cards_in_suit) + len(jokers) < 3:
                    continue
                ranks = sorted([c.rank for c in cards_in_suit])
                n = len(cards_in_suit) + len(jokers)
                for length in range(5, 2, -1):  # 5,4,3枚の階段
                    for start in range(1, 15 - length):
                        expected = [(start + i - 1) % 13 + 1 for i in range(length)]
                        temp = []
                        used_jokers = 0
                        # コピーリストを作り、使ったカードはpopで消す
                        available_cards = cards_in_suit[:]
                        available_jokers = jokers[:]
                        for val in expected:
                            found = False
                            for i, c in enumerate(available_cards):
                                if c.rank == val:
                                    temp.append(available_cards.pop(i))
                                    found = True
                                    break
                            if not found:
                                if used_jokers < len(available_jokers):
                                    temp.append(available_jokers.pop(0))
                                    used_jokers += 1
                                else:
                                    break
                        if len(temp) == length:
                            if self.game.is_valid_play(temp):
                                action_cards = temp
                                break
                    if action_cards:
                        break
                if action_cards:
                    break

        # ジョーカー単体を許可
        if not action_cards:
            jokers = [card for card in hand if card.is_joker]
            if jokers and self.game.is_valid_play([jokers[0]]):
                action_cards = [jokers[0]]

        # 出せなければパス
        if not action_cards:
            action_cards = None

        # プレイ実行（Noneならパス）
        valid, reward, _ = self.game.step(self.game.turn, action_cards)
        #ゲームが終了したか確認
        self.done = self.game.done
        obs = self._get_obs()

        if return_info:
            # プレイヤーIDと出したカードの情報も返す
            return obs, reward, self.done, {
                "player_id": player.player_id,
                "played_cards": action_cards
            }
        else:
            return obs, reward, self.done

    # カード情報を数値に変換
    def _encode_card(self, card):
        if card is None:
            return -1 # 場が空の場合は -1
        if card.is_joker:
            return 53 # ジョーカーは 53 としてエンコード
        suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
        return suit_map[card.suit] * 13 + (card.rank - 1)

    def _get_obs(self):
        player = self.game.players[self.game.turn]  # 現在のプレイヤー
        # 手札を数値化して長さを27枚に固定（足りない分は -1 で埋める）
        hand_encoded = [self._encode_card(c) for c in player.hand]
        hand_encoded += [-1] * (27 - len(hand_encoded))
        # 現在場に出ているカードを数値化
        field_card = self.game.current_field[-1] if self.game.current_field else None
        field_encoded = self._encode_card(field_card)

        return {
            "hand": np.array(hand_encoded, dtype=np.int32),
            "field": field_encoded
        }

    #def render(self):
        field_cards = self.game.current_field if self.game.current_field else ["（場リセット）"]
        print(f"--- 現在の場: {field_cards}")
        print(f"あなたの手札: {[str(c) for c in self.game.players[0].hand]}")
        for i, p in enumerate(self.game.players[1:], 1):
            print(f"Player {i} の手札枚数: {len(p.hand)}")
