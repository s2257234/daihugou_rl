import random
from re_game.game import Game  # ゲームのロジックを持つクラスをインポート
from gym.spaces import Discrete, Dict, Box  # OpenAI Gym 形式の環境を作るための空間定義
import numpy as np

# 大富豪ゲームの強化学習用環境クラス
class DaifugoEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.game = Game(num_players=num_players)  # ゲームのインスタンスを生成
        self.action_space = Discrete(54)  # 行動空間：最大54枚のカード（ジョーカー含む）
        self.observation_space = Dict({  # 観測空間：手札と場のカード
            'hand': Box(low=-1, high=53, shape=(27,), dtype=np.int32),  # 手札（最大27枚まで）
            'field': Box(low=-1, high=53, shape=(), dtype=np.int32)  # 現在の場のカード（1枚）
        })
        self.reset()  # 初期化

    # 環境の初期化（ゲームの再スタート）
    def reset(self):
        self.game = Game(num_players=self.num_players)
        return self._get_obs()

    # 環境の1ステップを進める（行動を実行）
    def step(self, action_idx):
        player = self.game.players[self.game.turn]
        action_card = None

        # 行動が有効なインデックス内であればカードを取得
        if action_idx < len(player.hand):
            action_card = player.hand[action_idx]

        # ゲームに1ステップ進ませる
        valid, reward, false = self.game.step(self.game.turn, action_card)
        done = self.game.done  # ゲーム終了フラグ
        obs = self._get_obs()  # 次の観測を取得
        return obs, reward, done, {}  # Gym互換の戻り値

    # カード情報を数値に変換する関数（観測用）
    def _encode_card(self, card):
        if card.is_joker:
            return 53  # ジョーカーは53で表現
        suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}  # スートのマッピング
        return suit_map[card.suit] * 13 + (card.rank - 1)  # 0〜51の範囲でエンコード

    # 観測データの生成（手札と場のカードをエンコード）
    def _get_obs(self):
        player = self.game.players[self.game.turn]
        hand_encoded = [self._encode_card(c) for c in player.hand]
        hand_encoded += [-1] * (27 - len(hand_encoded))  # 手札が27枚未満なら-1で埋める
        field_card = self.game.current_field[-1] if self.game.current_field else None
        field_encoded = self._encode_card(field_card) if field_card else -1

        return {
            'hand': np.array(hand_encoded, dtype=np.int32),
            'field': field_encoded
        }

    # 現在の状態をコンソールに出力（人間向けデバッグ用）
    def render(self):
        field_cards = self.game.current_field if self.game.current_field else ["（場リセット）"]
        print(f"--- 現在の場: {field_cards}")
        print(f"あなたの手札: {[str(c) for c in self.game.players[0].hand]}")        
        for i, p in enumerate(self.game.players[1:], 1):
            print(f"Player {i} の手札枚数: {len(p.hand)}")
