from game.environment import DaifugoSimpleEnv
from collections import defaultdict
import numpy as np

from agents.straight_agent import StraightAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts import MCTSAgent

NUM_EPISODES = 10  # シミュレーションするゲームの回数


def main():
    # プレイヤー構成
    agent_classes = [MCTSAgent, RandomAgent, RuleBasedAgent, RandomAgent]
    env = DaifugoSimpleEnv(num_players=4, agent_classes=agent_classes)
    rank_stats = defaultdict(lambda: defaultdict(int))

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        print(f"\n🃏 Episode {episode + 1} 開始")

        while not done:
            current_player_id = env.game.turn
            agent = env.agents[current_player_id]

            # エージェントに渡す観測
            hand = env.game.players[current_player_id].hand
            field = env.game.current_field[:]
            obs_for_agent = {'hand': hand, 'field': field}
            legal_actions = env._generate_legal_actions(hand, field)

            # Player 0だけMCTS、他はenvを使わない
            if isinstance(agent, MCTSAgent):
                action = agent.select_action(
                    obs_for_agent,
                    legal_actions=legal_actions,
                    env=env  # 本番の env を渡す
                )
                mcts_result = None  # MCTSの結果は使用しない
            else:
                action = agent.select_action(obs_for_agent, legal_actions=legal_actions)
                mcts_result = None

            # step に渡してレコードに格納
            action_to_env = [] if action is None else action
            obs, reward, done, info = env.step(
                return_info=True,
                external_action=action_to_env,
                mcts_result=mcts_result
            )

            # プレイ内容表示
            if 'played_cards' in info:
                if info['played_cards']:
                    print(f"Player {info['player_id']} played: ", end="")
                    print(", ".join(str(card) for card in info['played_cards']))
                else:
                    print(f"Player {info['player_id']} passed.")
            #if info.get('reset_happened'):
                #print("--- 場がリセットされました ---")

        # エピソード終了後のランキング集計
        for rank, player_id in enumerate(env.game.rankings):
            rank_stats[rank + 1][player_id] += 1

    # 累計順位表示
    print("\n📊 累計順位集計（プレイヤー別）:")
    for player_id in range(env.num_players):
        print(f"Player {player_id}: ", end="")
        for rank in range(1, env.num_players + 1):
            count = rank_stats[rank].get(player_id, 0)
            print(f"{rank}位: {count}回 ", end="")
        print()


if __name__ == "__main__":
    main()
