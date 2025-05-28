#------ランダムな動作をするエージェント（デバッグ・学習比較用）------
import random
from game.environment import DaifugoEnvSimple

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def act(self, observation):
        hand = observation['hand']
        valid_indices = [i for i, card in enumerate(hand) if card != -1]
        return random.choice(valid_indices) if valid_indices else 0
