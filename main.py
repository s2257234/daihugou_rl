from game.environment import DaifugoSimpleEnv
from collections import defaultdict

NUM_EPISODES = 10  # シミュレーションするゲームの回数

def main():
    env = DaifugoSimpleEnv(num_players=4)  # プレイヤー数4人で環境を初期化

    # 順位の集計用: {順位（1〜4）: {player_id: カウント数}}
    rank_stats = defaultdict(lambda: defaultdict(int))

    for episode in range(NUM_EPISODES):
        obs = env.reset()  # 環境をリセット（カードを配るなど）
        done = False

        print(f"\n🃏 Episode {episode + 1} 開始")  # ゲーム開始のログ表示

        # 1ゲームが終了するまでステップを繰り返す
        while not done:
            # ゲームを1手進める（step）し、状態・報酬・終了フラグ・情報を取得
            obs, reward, done, info = env.step(return_info=True)

            # 出されたカード or パスの表示
            if 'played_cards' in info:
                if info['played_cards']:
                    # 出されたカードの表示
                    print(f"Player {info['player_id']} played: ", end="")
                    print(", ".join(str(card) for card in info['played_cards']))
                else:
                    # パスした場合の表示
                    print(f"Player {info['player_id']} passed.")

        # ゲーム終了後：順位を取得して集計
        for rank, player_id in enumerate(env.game.rankings):
            rank_stats[rank + 1][player_id] += 1  # 順位は1位〜で記録

    # 全ゲーム終了後の統計表示
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
