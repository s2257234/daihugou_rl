#------ランダムな動作をするエージェント（デバッグ・学習比較用）------
import random

def select_random_action(observation):
    hand = observation['hand']
    # 手札のうち、カード枚数が0でないもののインデックスを列挙し、最後にパス用のアクション（handの長さ）を追加
    possible_actions = [i for i, r in enumerate(hand) if r != 0] + [len(hand)]  # PASS = last index
    # 可能なアクションの中からランダムに1つ選択して返す
    return random.choice(possible_actions)
