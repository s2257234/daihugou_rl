from game.environment import DaifugoSimpleEnv

def main():
    env = DaifugoSimpleEnv(num_players=4)
    obs = env.reset()
    env.render()

    while not env.done:
        #input("Enter を押すと次のターンに進みます...")
        obs, reward, done = env.step()
        env.render()
        if done:
            print("🎉 ゲーム終了！")
            print("🏆 順位：")
            for i, player_idx in enumerate(env.game.rankings):
                print(f"{i+1}位: Player {player_idx}")
                break

if __name__ == "__main__":
    main()
