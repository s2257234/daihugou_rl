from game.environment import DaifugoSimpleEnv
from collections import defaultdict
from agents.straight_agent import StraightAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent

NUM_EPISODES = 10  # シミュレーションするゲームの回数

def main():
    # エージェントのクラスを指定
    agent_classes = [RandomAgent, RandomAgent, RuleBasedAgent, StraightAgent]
    env = DaifugoSimpleEnv(num_players=4, agent_classes=agent_classes)  # プレイヤー数4人で環境を初期化

    # 順位の集計用: {順位（1〜4）: {player_id: カウント数}}
    rank_stats = defaultdict(lambda: defaultdict(int))

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False

        print(f"\n🃏 Episode {episode + 1} 開始")  # ゲーム開始のログ表示

        # 1ゲームが終了するまでステップを繰り返す
        while not done:
            obs, reward, done, info = env.step(return_info=True)

            # 出されたカード or パスの表示
            if 'played_cards' in info:
                if info['played_cards']:
                    print(f"Player {info['player_id']} played: ", end="")
                    print(", ".join(str(card) for card in info['played_cards']))
                else:
                    print(f"Player {info['player_id']} passed.")

            # 場がリセットされた場合
            if info.get('reset_happened'):
                print("--- 場がリセットされました ---")

    
        # 順位集計
        for rank, player_id in enumerate(env.game.rankings):
            rank_stats[rank + 1][player_id] += 1  # 順位は1位〜で記録


    # 全エピソード終了後の順位集計を表示
    print("\n📊 累計順位集計（プレイヤー別）:")

    # 各プレイヤーごとに順位回数を表示
    for player_id in range(env.num_players):
        print(f"Player {player_id}: ", end="")
        for rank in range(1, env.num_players + 1):
            count = rank_stats[rank].get(player_id, 0)
            print(f"{rank}位: {count}回 ", end="")
        print()  # 改行

if __name__ == "__main__":
    main()
