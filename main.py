from game.environment import DaifugoSimpleEnv
from collections import defaultdict

NUM_EPISODES = 10

def main():
    env = DaifugoSimpleEnv(num_players=4)

    # 順位集計用の辞書: {順位: {player_id: カウント}}
    rank_stats = defaultdict(lambda: defaultdict(int))

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False

        while not done:
            obs, reward, done = env.step()

        # ゲーム終了後の順位を取得して集計
        for rank, player_id in enumerate(env.game.rankings):
            rank_stats[rank + 1][player_id] += 1  # 1位〜4位 の形式にする

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} 完了")

    print(f"\n🎉 全 {NUM_EPISODES} エピソード終了！")
    print("\n📊 累計順位集計（プレイヤー別）:")

    for player_id in range(env.num_players):
        print(f"Player {player_id}: ", end="")
        for rank in range(1, env.num_players + 1):
            count = rank_stats[rank].get(player_id, 0)
            print(f"{rank}位: {count}回 ", end="")
        print()  # 改行
        
if __name__ == "__main__":
    main()
