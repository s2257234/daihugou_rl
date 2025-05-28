import random
import numpy as np
from game.game import Game


class DaifugoSimpleEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.game = Game(num_players=self.num_players)
        self.current_player = self.game.turn
        self.done = False

    def reset(self):
        self.game = Game(num_players=self.num_players)
        self.current_player = self.game.turn
        self.done = False
        return self._get_obs()

    def step(self):
        player = self.game.players[self.game.turn]

        # 出せるカードを探す（出せる最初のカードを選択）
        action_card = None
        for idx, card in enumerate(player.hand):
            if self.game.is_valid_play(card):  # ルールに合うか確認
                action_card = card
                break

        # プレイ実行（出せるカードがなければ action_card=None → パス扱い）
        valid, reward, _ = self.game.step(self.game.turn, action_card)
        self.done = self.game.done
        obs = self._get_obs()
        return obs, reward, self.done

    def _encode_card(self, card):
        if card is None:
            return -1
        if card.is_joker:
            return 53
        suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
        return suit_map[card.suit] * 13 + (card.rank - 1)

    def _get_obs(self):
        player = self.game.players[self.game.turn]
        hand_encoded = [self._encode_card(c) for c in player.hand]
        hand_encoded += [-1] * (27 - len(hand_encoded))
        field_card = self.game.current_field[-1] if self.game.current_field else None
        field_encoded = self._encode_card(field_card)

        return {
            "hand": np.array(hand_encoded, dtype=np.int32),
            "field": field_encoded
        }

    def render(self):
        field_cards = self.game.current_field if self.game.current_field else ["（場リセット）"]
        print(f"--- 現在の場: {field_cards}")
        print(f"あなたの手札: {[str(c) for c in self.game.players[0].hand]}")
        for i, p in enumerate(self.game.players[1:], 1):
            print(f"Player {i} の手札枚数: {len(p.hand)}")
