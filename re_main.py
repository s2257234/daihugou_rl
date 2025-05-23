# 大富豪環境とランダムエージェントの行動選択関数をインポート
from re_game.environment import DaifugoEnv
from re_game.game import Game
from re_agents.random_agent import select_random_action

# メイン処理：このファイルが直接実行されたときに動作する部分
if __name__ == "__main__":
    # 環境の初期化（プレイヤー数4人）
    env = DaifugoEnv(num_players=4)
    game = Game()  # ゲームのインスタンスを生成
    
    # 環境をリセットして初期状態を取得
    obs = env.reset()
    done = False  # ゲーム終了フラグ

    # ゲームが終了するまで繰り返す
    while not done:
        env.render()  # 現在の場と手札を表示（人間向けの出力）
        
        # ランダムに行動を選択（手札の中からランダムにカードを選ぶ）
        action = select_random_action(obs)
        print("\n選択された行動:", action)

        print("ターン数:", env.game.turn_count)

        # 選択した行動を実行し、次の状態・報酬・終了判定を取得
        obs, reward, done, _ = env.step(action)

        
    # ゲーム終了後のメッセージと報酬を表示
    print("\nゲーム終了 🎮")
    print("報酬:", reward)
