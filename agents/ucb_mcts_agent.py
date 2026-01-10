"""
UCB MCTS Agent - UCBで探索を決める単体のMCTSエージェント

既存のAlphaZero手法で使っているMCTSとは別に、UCB1で探索を決める単体のMCTSエージェント。
軽量なルールベースのロールアウトを使用し、各シミュレーションごとに相手の手札を
自分の手札を除いたカードからランダムにサンプリングする。
"""
import copy
import math
import random
from typing import Any, List, Optional


class UCBMCTSNode:
    """UCB MCTS用ノード
    
    ノードには環境状態を持たせない。アクション履歴のみを保持。
    """
    def __init__(self, parent=None, action=None):
        self.parent = parent    # 親ノード（Noneならルート）
        self.action = action    # 親ノードからこのノードへの遷移アクション
        self.children = []      # 子ノードのリスト（各子ノードは`action`を持つ）
        self.visits = 0         # このノードが訪問された回数
        self.value = 0          # このノードの累積価値（報酬の合計）


class UCBMCTSAgent:
    """UCB MCTS Agent
    
    UCB1で探索を決める単体のMCTSエージェント。
    ランダムロールアウトではなく、軽量なルールベースのロールアウトを使用。
    """
    
    def __init__(self, player_id=None, num_simulations=200, ucb_c=1.4):
        """
        Args:
            player_id: プレイヤーID
            num_simulations: シミュレーション回数（agents/config.pyのnum_simulationsを使用）
            ucb_c: UCBの定数（デフォルト1.4）
        """
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.ucb_c = ucb_c
        self.env = None  # 環境参照（必要に応じて設定）
    
    def select_action(self, observation, legal_actions=None):
        """行動を選択
        
        Args:
            observation: 観測情報（環境オブジェクトを含む場合がある）
            legal_actions: 合法手のリスト（現状未使用）
        
        Returns:
            選択した行動（Cardオブジェクトのリスト、またはNone）
        """
        # 環境を取得
        env = None
        if hasattr(observation, 'game'):
            env = observation
        elif hasattr(self, 'env') and self.env is not None:
            env = self.env
        else:
            # observationが辞書型の場合、環境を推測
            if isinstance(observation, dict) and 'env' in observation:
                env = observation['env']
        
        if env is None:
            # 環境が取得できない場合はランダムに行動を選択
            if legal_actions:
                return random.choice(legal_actions)
            return None
        
        # MCTS探索を実行
        return self.search(env)
    
    def search(self, env):
        """MCTS探索のメインループ
        
        Args:
            env: 環境オブジェクト
        
        Returns:
            最も訪問回数が多い行動
        """
        root_node = UCBMCTSNode()
        root_player_id = self.player_id if self.player_id is not None else env.game.turn
        
        # 各シミュレーションを実行
        for _ in range(self.num_simulations):
            # 環境をクローン
            env_copy = self._copy_env(env)
            
            # 手札をサンプリング（決定化）
            self._sample_opponent_hands(env_copy, env, root_player_id)
            
            # Selection → Expansion → Simulation → Backpropagation
            node = root_node
            current_env = env_copy
            
            # Selection: ルートから葉ノードまで選択
            while node.children:
                node, current_env = self._uct_select(node, current_env, root_player_id)
            
            # Expansion: 葉ノードを展開
            if not env_copy.game.done:
                expanded_node, current_env = self._expand(node, current_env, root_player_id)
                if expanded_node:
                    node = expanded_node
            
            # Simulation: ロールアウトを実行
            reward = self._rollout(current_env, root_player_id)
            
            # Backpropagation: 結果を逆伝播
            self._backpropagate(node, reward)
        
        # 最も訪問回数が多い子ノードのアクションを選択
        if not root_node.children:
            # 子ノードがない場合は合法手からランダムに選択
            hand = env.game.players[env.game.turn].hand
            field = env.game.current_field[:]
            legal_actions = env._generate_legal_actions(hand, field)
            if legal_actions:
                return random.choice(legal_actions)
            return None
        
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action if best_child else None
    
    def _uct_select(self, node, env, root_player_id):
        """UCB1に基づいて子ノードを選択
        
        Args:
            node: 現在のノード
            env: 現在の環境状態
            root_player_id: ルートプレイヤーID
        
        Returns:
            (選択された子ノード, アクション適用後の環境状態)
        """
        # 未訪問ノードを優先
        for child in node.children:
            if child.visits == 0:
                new_env = self._copy_env(env)
                self._apply_action(new_env, child.action)
                return child, new_env
        
        # UCB1値を計算
        uct_values = []
        for child in node.children:
            if child.visits == 0:
                uct_value = float('inf')
            else:
                avg_value = child.value / child.visits
                exploration = self.ucb_c * math.sqrt(math.log(node.visits) / child.visits)
                uct_value = avg_value + exploration
            uct_values.append(uct_value)
        
        # UCB値最大の子ノードを選択
        max_index = uct_values.index(max(uct_values))
        selected_child = node.children[max_index]
        
        # 環境にアクションを適用
        new_env = self._copy_env(env)
        self._apply_action(new_env, selected_child.action)
        
        return selected_child, new_env
    
    def _expand(self, node, env, root_player_id):
        """葉ノードを展開
        
        Args:
            node: 展開するノード
            env: 現在の環境状態
            root_player_id: ルートプレイヤーID
        
        Returns:
            (展開された子ノード, 環境状態（そのまま）) または (None, env)
        """
        # 合法手を取得
        current_player_id = env.game.turn
        hand = env.game.players[current_player_id].hand
        field = env.game.current_field[:]
        legal_actions = env._generate_legal_actions(hand, field)
        
        # 既に展開済みのアクションを除外
        existing_actions = {id(child.action) if child.action is not None else None for child in node.children}
        
        # 新しい子ノードを作成
        for action in legal_actions:
            action_id = id(action) if action is not None else None
            if action_id not in existing_actions:
                child_node = UCBMCTSNode(parent=node, action=action)
                node.children.append(child_node)
                return child_node, env
        
        return None, env
    
    def _rollout(self, env, root_player_id, max_steps=200):
        """軽量なルールベースのロールアウトを実行
        
        Args:
            env: 環境状態
            root_player_id: ルートプレイヤーID
            max_steps: 最大ステップ数
        
        Returns:
            報酬値
        """
        steps = 0
        while not env.game.done and steps < max_steps:
            current_player_id = env.game.turn
            hand = env.game.players[current_player_id].hand
            field = env.game.current_field[:]
            legal_actions = env._generate_legal_actions(hand, field)
            
            if not legal_actions:
                # 合法手がない場合はパス
                env.game.step(current_player_id, None)
            else:
                # 軽量ルールベース: 弱い順にソートして一番手前のものを出す
                best_action = min(legal_actions, key=self._rollout_heuristic)
                env.game.step(current_player_id, best_action)
            
            steps += 1
        
        # 報酬を計算
        return self._calculate_reward(env, root_player_id)
    
    def _rollout_heuristic(self, action):
        """ロールアウト用のヒューリスティックスコア
        
        Args:
            action: アクション（Cardオブジェクトのリスト、またはNone）
        
        Returns:
            スコア（低いほど優先）
        """
        if action is None:
            return 9999  # パスは最後の手段
        
        # actionが空リストの場合
        if not action:
            return 9999
        
        # ジョーカーを含む場合は温存
        if any(getattr(c, 'is_joker', False) for c in action):
            return 300
        
        # ランク2を含む場合は温存
        if any(getattr(c, 'rank', None) == 2 for c in action):
            return 200
        
        # それ以外は先頭カードのランク（弱い順）
        first_card = action[0]
        rank = getattr(first_card, 'rank', None)
        if rank is None:
            return 1000  # ランクが取得できない場合は後回し
        
        return rank
    
    def _calculate_reward(self, env, root_player_id):
        """報酬を計算
        
        Args:
            env: 環境状態
            root_player_id: ルートプレイヤーID
        
        Returns:
            報酬値（1位:1.0、2位:0.66、3位:0.33、4位:0.0）
        """
        if not env.game.done:
            return 0.0
        
        # 順位を取得
        rankings = getattr(env.game, 'rankings', [])
        if root_player_id in rankings:
            rank = rankings.index(root_player_id) + 1
            if rank == 1:
                return 1.0
            elif rank == 2:
                return 0.66
            elif rank == 3:
                return 0.33
            else:
                return 0.0
        
        return 0.0
    
    def _backpropagate(self, node, reward):
        """シミュレーション結果をルートまで逆伝播
        
        Args:
            node: 開始ノード
            reward: 報酬値
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _sample_opponent_hands(self, env, original_env, root_player_id):
        """相手の手札をランダムにサンプリング
        
        Args:
            env: 環境状態（クローン済み、手札を配分する先）
            original_env: 元の環境状態（既知カードを取得する元）
            root_player_id: ルートプレイヤーID
        """
        g = env.game
        g_orig = original_env.game
        
        # 既知のカードを収集（元の環境から、文字列IDとして）
        root_hand_ids = {str(c) for c in g_orig.players[root_player_id].hand}
        field_ids = {str(c) for c in getattr(g_orig, 'current_field', []) or []}
        
        # 全カードを収集（元の環境から、文字列IDとして）
        all_cards_dict = {}  # 文字列ID -> カード文字列ID のマッピング
        for p in g_orig.players:
            for c in p.hand:
                cid = str(c)
                if cid not in all_cards_dict:
                    all_cards_dict[cid] = cid
        for c in getattr(g_orig, 'current_field', []) or []:
            cid = str(c)
            if cid not in all_cards_dict:
                all_cards_dict[cid] = cid
        all_cards = list(all_cards_dict.values())
        
        # 既知カード（自分の手札、場のカード、既に上がったプレイヤーの手札）
        known = set(root_hand_ids) | field_ids
        for rid in getattr(g_orig, 'rankings', []):
            if rid != root_player_id:
                known.update(str(c) for c in g_orig.players[rid].hand)
        
        # 未知のカード（サンプリング対象）
        unknown_seed = [cid for cid in all_cards if cid not in known]
        
        # 相手プレイヤーIDのリスト
        opp_ids = [i for i in range(len(g.players)) if i != root_player_id and i not in getattr(g, 'rankings', [])]
        
        if not opp_ids or not unknown_seed:
            # 相手がいない、または未知カードがない場合は何もしない
            return
        
        # 各相手プレイヤーの残り手札数を計算（元の環境から）
        capacities = {}
        for pid in opp_ids:
            capacities[pid] = len(g_orig.players[pid].hand)
        
        # 残りカードをランダムにシャッフル
        random.shuffle(unknown_seed)
        
        # 各相手プレイヤーに手札を配分
        from game.card import Card
        card_idx = 0
        for pid in opp_ids:
            capacity = capacities[pid]
            new_hand = []
            assigned = 0
            while assigned < capacity and card_idx < len(unknown_seed):
                cid = unknown_seed[card_idx]
                card_idx += 1  # 常にインデックスを進める
                try:
                    # Card.from_stringがある場合
                    if hasattr(Card, 'from_string'):
                        card = Card.from_string(cid)
                    else:
                        # フォールバック: 文字列からCardオブジェクトを構築
                        card = Card(cid)
                    # カードが有効な場合のみ追加
                    if card is not None:
                        new_hand.append(card)
                        assigned += 1
                except Exception:
                    # パースに失敗した場合は次のカードへ
                    continue
            
            # 手札数が不足している場合は警告（通常は発生しないはず）
            if len(new_hand) < capacity:
                # 手札数が不足している場合は空のカードで埋める（通常は発生しない）
                pass
            
            g.players[pid].hand = new_hand
    
    def _copy_env(self, env):
        """環境をクローン
        
        Args:
            env: 元の環境
        
        Returns:
            クローンされた環境
        """
        # agents/drl_agent.pyの_copy_envを参考に実装
        base = env
        new_env = copy.copy(base)
        g = base.game
        g_new = copy.copy(g)
        
        # Card クラスのクローン関数
        from game.card import Card
        
        def _clone_card(c):
            try:
                if getattr(c, 'is_joker', False):
                    nc = Card(is_joker=True)
                    try:
                        nc.joker_as_rank = getattr(c, 'joker_as_rank', None)
                        nc.joker_as_suit = getattr(c, 'joker_as_suit', None)
                    except Exception:
                        pass
                    return nc
                return Card(suit=getattr(c, 'suit', None), rank=getattr(c, 'rank', None), is_joker=False)
            except Exception:
                return Card(is_joker=True)
        
        # プレイヤーをクローン
        new_players = []
        for p in g.players:
            p_new = copy.copy(p)
            try:
                p_new.hand = [_clone_card(c) for c in list(p.hand)]
            except Exception:
                p_new.hand = list(p.hand)
            new_players.append(p_new)
        g_new.players = new_players
        
        # 場の状態をコピー
        try:
            g_new.current_field = [_clone_card(c) for c in list(getattr(g, 'current_field', []) or [])]
        except Exception:
            g_new.current_field = list(getattr(g, 'current_field', []) or [])
        
        # passed状態・rankings・手番をコピー
        g_new.passed = list(g.passed) if hasattr(g, 'passed') else []
        g_new.rankings = list(getattr(g, 'rankings', []))
        try:
            g_new.turn = int(getattr(g, 'turn', 0))
        except Exception:
            pass
        
        # RuleCheckerをコピー
        try:
            from game.rules import RuleChecker
            rc_src = getattr(g, 'rule_checker', None)
            rc = RuleChecker()
            if rc_src is not None:
                try:
                    rc.revolution = bool(getattr(rc_src, 'revolution', False))
                except Exception:
                    rc.revolution = False
                for name in (
                    'strict_straight_progression',
                    'rev_enable_four_kind', 'rev_four_kind_allow_joker', 'rev_four_kind_exact',
                    'rev_enable_straight', 'rev_straight_min_len', 'rev_straight_allow_joker',
                    'enable_joker', 'joker_is_strongest', 'enable_eight_reverse',
                ):
                    try:
                        setattr(rc, name, getattr(rc_src, name))
                    except Exception:
                        pass
            g_new.rule_checker = rc
        except Exception:
            pass
        
        try:
            g_new.silent = True
        except Exception:
            pass
        
        new_env.game = g_new
        return new_env
    
    def _apply_action(self, env, action):
        """環境にアクションを適用
        
        Args:
            env: 環境状態
            action: 適用するアクション
        """
        current_player_id = env.game.turn
        env.game.step(current_player_id, action)

