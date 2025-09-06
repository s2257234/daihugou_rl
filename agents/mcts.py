import copy
import math
import random

class environmentEnv:
    def __init__(self):
        # 既存初期化
        from game.game import Game
        self.game = Game()  # 既存のGameクラス
        self.current_state = None  # MCTS用に状態を保持できるように
    
    def copy(self):
        """MCTSで使う環境のコピー"""
        new_env = environmentEnv()
        # Gameクラスのコピーを作る
        new_env.game = self._copy_game()  # Gameクラス側にcopy関数が必要
        return new_env
    
    def _copy_game(self):
        """Gameクラスのディープコピーを作成"""
        new_game = copy.deepcopy(self.game)
        return new_game
    
    def get_legal_actions(self):
        """現在手番で出せる手のリストを返す"""
        current_player = self.game.players[self.game.turn]
        hand = current_player.hand
        field = self.game.current_field[:]
        
        # 環境の_generate_legal_actionsを使用
        from game.environment import DaifugoSimpleEnv
        temp_env = DaifugoSimpleEnv()
        temp_env.game = self.game
        legal_actions = temp_env._generate_legal_actions(hand, field)
        
        return legal_actions
    
    def step(self, action):
        """actionを実行して次の状態へ"""
        current_player_id = self.game.turn
        self.game.step(current_player_id, action)
    
    def is_done(self):
        """ゲーム終了判定"""
        return self.game.done
    
    def get_reward(self, player_id=0):
        """終了時の報酬。自分視点での勝利評価"""
        if not self.is_done():
            return 0  # ゲーム終了前は0
        
        # 順位に基づく報酬計算
        if player_id in self.game.rankings:
            rank = self.game.rankings.index(player_id) + 1
            if rank == 1:
                return 1.0  # 1位
            elif rank == 2:
                return 0.5  # 2位
            elif rank == 3:
                return 0.0  # 3位
            else:
                return -0.5  # 4位
        else:
            return 0  # 順位が決まっていない場合


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state      # このノードが表す環境状態（environmentEnvのコピー）
        self.parent = parent    # 親ノード（Noneならルート）
        self.action = action    # 親ノードからこのノードに遷移したアクション
        self.children = []     # 子ノードのリスト
        self.visits = 0        # このノードが訪問された回数
        self.value = 0         # このノードの累積価値（報酬の合計）


class MCTSAgent:
    def __init__(self, player_id, env=None, num_simulations=10):
        self.player_id = player_id
        self.env = env          # 環境のコピー
        self.num_simulations = num_simulations


    def set_env(self, env):
        self.env = env

    def select_action(self, obs=None, legal_actions=None, env=None):
        """
        DaifugoSimpleEnv等から呼び出される標準API。
        必要に応じてself.envを更新してからsearch()を呼ぶ。
        obsやlegal_actionsは現状未使用。
        envが渡された場合はself.envを更新。
        DaifugoSimpleEnv型なら自動でenvironmentEnvにラップ。
        """
        if env is not None:
            self.env = env
        # DaifugoSimpleEnv型ならenvironmentEnvでラップ
        from game.environment import DaifugoSimpleEnv
        if isinstance(self.env, DaifugoSimpleEnv):
            wrapped_env = environmentEnv()
            wrapped_env.game = self.env.game  # ゲーム状態を引き継ぐ
            self.env = wrapped_env
        return self.search()

    def search(self):
        root_node = MCTSNode(state=self.env.copy())

        for _ in range(self.num_simulations):
            node = root_node
            
            # 選択フェーズ
            while node.children:
                node = self._uct_select(node)
            
            # 展開フェーズ
            node = self._expand(node) or node
            
            # シミュレーションフェーズ
            reward = self._simulate(node)

            # バックプロパゲーションフェーズ
            self._backpropagate(node, reward)
        
        # 最も訪問された子ノードのアクションを選択
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action if best_child else None
    
    def _uct_select(self, node, c=1.4):
        for child in node.children:
            if child.visits == 0:
                return child
        uct_values = [
            (child.value / child.visits) + c * math.sqrt(math.log(node.visits) / child.visits)
            for child in node.children
        ]
        # UCT値最大の子ノードを返す
        max_index = uct_values.index(max(uct_values))
        return node.children[max_index]

    def _expand(self, node):
        legal_actions = node.state.get_legal_actions()
        for action in legal_actions:
            if action not in [child.action for child in node.children]:
                new_state = node.state.copy()
                new_state.step(action)
                child_node = MCTSNode(state=new_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node
        return None

    def _simulate(self, node, max_steps=100):
        sim_env = node.state.copy()
        steps = 0
        while not sim_env.is_done() and steps < max_steps:
            legal_actions = sim_env.get_legal_actions()
            # legal_actionsが[None]の場合は["pass"]に置き換え
            if legal_actions == [None]:
                legal_actions = ["pass"]
            print(f"[MCTS SIM] step={steps} legal_actions={legal_actions}")
            if not legal_actions:
                action = "pass"
                print(f"[MCTS SIM] action=pass (no legal actions)")
            else:
                action = random.choice(legal_actions)
                # actionがNoneの場合は"pass"にする
                if action is None:
                    action = "pass"
                print(f"[MCTS SIM] action={action}")
            sim_env.step(action)
            steps += 1
        reward = sim_env.get_reward(player_id=0)
        print(f"[MCTS SIM] finished after {steps} steps, reward={reward}")
        return reward
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent