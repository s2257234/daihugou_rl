#------ランダムな動作をするエージェント（デバッグ・学習比較用）------
import random
#from game.environment import DaifugoEnvSimple

class RandomAgent:
    def __init__(self, player_id=None):
        self.player_id = player_id

    def select_action(self, observation, legal_actions=None):
        if legal_actions:
            return random.choice(legal_actions)
        return None
