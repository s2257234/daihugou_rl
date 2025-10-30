import copy
import math
import random

"""MCTS 実装共通化モジュール

提供要素:
1) 既存のシンプルランダムロールアウト型 MCTSAgent (後方互換用)
2) AlphaZero / PUCT 方式向けの軽量コア関数とノード定義 (drI_agent.py から再利用可能)

多人数(大富豪 4人)対応メモ:
 - シンプル版 MCTSAgent は従来通りロールアウト報酬をそのままバックアップします。
 - PUCT 版(PUCTNode) では "符号反転" は行わず、value は常に root 視点 (または policy_value_fn が返す視点) のスカラーとして上方伝播。
     複数プレイヤー時にプレイヤー視点変換を行いたい場合は policy_value_fn 側で root 視点に正規化してください。

再利用ポイント (AlphaZero 連携):
 - run_puct_mcts(...) を呼び出し policy_value_fn / get_legal_actions_fn を差し込む
 - 戻り値 root から visit 分布を抽出し方策ターゲット π を生成

注意: 既存コード互換のため MCTSAgent / environmentEnv は残す。
"""

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


# =============================================================
# PUCT / AlphaZero 用 ノード
# =============================================================
class PUCTNode:
    """PUCT (AlphaZero) 用ノード。

    value_sum: 累積価値 (root 視点で加算)。平均値 = value_sum / visit_count。
    prior: policy_value_fn が返した事前確率。
    children: {action: PUCTNode}
    to_play: 手番プレイヤーID (必要なら視点変換で利用)。
    """

    __slots__ = (
        "parent", "prior", "visit_count", "value_sum", "children", "action", "to_play"
    )

    def __init__(self, prior: float, parent=None, action=None, to_play: int = 0):
        self.parent = parent
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.action = action
        self.to_play = to_play

    # ---- 派生量 ----
    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    # ---- 展開 ----
    def expand(self, to_play: int, policy_dict):
        for act, p in policy_dict.items():
            if act not in self.children:
                self.children[act] = PUCTNode(prior=p, parent=self, action=act, to_play=to_play)

    # ---- バックアップ (4人ゲーム想定: 符号反転しない) ----
    def backup(self, leaf_value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += leaf_value
            node = node.parent


def _puct_select(node: PUCTNode, c_puct: float, virtual_counts=None) -> PUCTNode:
    """子ノードの中から PUCT スコア最大のものを返す"""
    if virtual_counts is None:
        virtual_counts = {}
    def vc(ch):
        return virtual_counts.get(id(ch), 0)
    total_visits = max(1, sum(child.visit_count + vc(child) for child in node.children.values()))
    best, best_score = None, -1e18
    sqrt_total = math.sqrt(total_visits)
    for child in node.children.values():
        visit_eff = child.visit_count + vc(child)
        u = c_puct * child.prior * sqrt_total / (1 + visit_eff)
        score = child.value + u
        if score > best_score:
            best_score = score
            best = child
    return best


def run_puct_mcts(root_env_copy,
                  num_simulations: int,
                  policy_value_fn,
                  get_legal_actions_fn,
                  c_puct: float = 1.4,
                  add_dirichlet: bool = True,
                  dirichlet_alpha: float = 0.3,
                  dirichlet_epsilon: float = 0.25,
                  root_player_id: int = 0,
                  *,
                  policy_value_batch_fn=None,
                  batch_eval_size: int = 1,
                  transposition_table: dict | None = None,
                  state_key_fn = None,
                  determinize_fn = None,
                  early_stop_enable: bool = False,
                  early_stop_min_sims: int = 16,
                  early_stop_visit_ratio: float = 0.75,
                  early_stop_gap_ratio: float = 0.10,
                  early_stop_log_sample_rate: float = 0.0,
                  early_stop_post_min_batch: int | None = None,
                  early_stop_debug: bool = False,
                  early_stop_logger=None):
    """AlphaZero 風 PUCT MCTS 実行 (正規化 & 欠損補完対応版)。"""
    # ルート合法手
    legal_root = get_legal_actions_fn(root_env_copy)
    if not legal_root:
        legal_root = ["pass"]
    policy_root, root_value = policy_value_fn(root_env_copy)
    # policy_root を合法手に合わせて補完・正規化
    if not policy_root:
        policy_root = {a: 1.0 / len(legal_root) for a in legal_root}
    else:
        # 欠損を均等割当
        missing = [a for a in legal_root if a not in policy_root]
        total = sum(policy_root.values())
        if total <= 0:
            policy_root = {a: 1.0 / len(legal_root) for a in legal_root}
        else:
            for k in list(policy_root.keys()):
                policy_root[k] /= total
            if missing:
                remain = max(0.0, 1.0 - sum(policy_root.values()))
                add = remain / len(missing) if missing else 0.0
                for m in missing:
                    policy_root[m] = add
            # 最終正規化
            s2 = sum(policy_root.values())
            if s2 > 0:
                for k in list(policy_root.keys()):
                    policy_root[k] /= s2

    root = PUCTNode(prior=1.0, to_play=root_player_id)
    root.expand(root_player_id, {a: policy_root[a] for a in legal_root if a in policy_root})
    root.visit_count = 1
    root.value_sum = root_value

    # Dirichlet ノイズ
    if add_dirichlet and root.children:
        actions = list(root.children.keys())
        noises = [random.gammavariate(dirichlet_alpha, 1.0) for _ in actions]
        s = sum(noises)
        noises = [n / s for n in noises]
        for a, n in zip(actions, noises):
            child = root.children[a]
            child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon

    def _fast_clone(env):
        # ルートと同じ軽量コピー方針: game の可変構造を手動複製
        g = env.game
        g_new = copy.copy(g)
        new_players = []
        for p in g.players:
            p_new = copy.copy(p)
            p_new.hand = list(p.hand)
            new_players.append(p_new)
        g_new.players = new_players
        g_new.current_field = list(g.current_field)
        g_new.passed = list(g.passed)
        g_new.rankings = list(getattr(g, 'rankings', []))
        # 行動履歴コピー (存在すれば)
        # 行動履歴は MCTS シミュレーションでは参照しないため複製しない（大幅なメモリ削減）
        # 重要: RuleChecker/Deck の deepcopy は非常に重いので回避し、最小限の状態のみを複製する
        from game.rules import RuleChecker
        rc_src = getattr(g, 'rule_checker', None)
        rc = RuleChecker()
        if rc_src is not None:
            # 革命フラグなど必要な最低限の状態のみコピー（イベントや履歴はコピーしない）
            try:
                rc.revolution = bool(getattr(rc_src, 'revolution', False))
            except Exception:
                rc.revolution = False
            for name in (
                'strict_straight_progression',
                'rev_enable_four_kind','rev_four_kind_allow_joker','rev_four_kind_exact',
                'rev_enable_straight','rev_straight_min_len','rev_straight_allow_joker',
            ):
                try:
                    setattr(rc, name, getattr(rc_src, name))
                except Exception:
                    pass
        g_new.rule_checker = rc
        # Deck は配り直し時にしか使用しないため、共有参照で十分（リセットは行わない経路）
        try:
            g_new.deck = getattr(g, 'deck', None)
        except Exception:
            g_new.deck = None
        env_new = copy.copy(env)
        env_new.game = g_new
        return env_new

    # --------------------------------------
    # 事前: キャッシュ用キー関数
    # --------------------------------------
    if state_key_fn is None:
        def state_key_fn_default(e):
            try:
                g = e.game
                # 安全なキー（手番 / 場 / 各手札のカードID / パス状況 / ランキング / 革命）
                turn = getattr(g, 'turn', 0)
                field = tuple(str(c) for c in getattr(g, 'current_field', []))
                hands = tuple(tuple(sorted(str(c) for c in p.hand)) for p in getattr(g, 'players', []))
                passed = tuple(bool(x) for x in getattr(g, 'passed', []))
                rankings = tuple(getattr(g, 'rankings', []))
                revo = bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False))
                return (turn, field, hands, passed, rankings, revo)
            except Exception:
                return None
        state_key_fn = state_key_fn_default

    # トランスポジションテーブル（None の場合はキャッシュ無効）
    TT = transposition_table if transposition_table is not None else None

    # --------------------------------------
    # 反復: バッチ評価付き MCTS
    # --------------------------------------
    sims_done = 0
    batch_eval_size = max(1, int(batch_eval_size or 1))
    # 早期停止後の細粒度バッチサイズ（有効な場合）
    post_min_batch = None
    if early_stop_post_min_batch is not None:
        try:
            pm = int(early_stop_post_min_batch)
            if pm > 1:
                post_min_batch = pm
        except Exception:
            post_min_batch = None

    # 内部ヘルパ: どの環境実装でも「外部から与えた行動」で前進させる
    def _step_env_safe(e, act):
        # 1) 新しい環境: external_action + simulate
        try:
            return e.step(return_info=False, external_action=act, simulate=True)
        except TypeError:
            pass
        # 2) 互換: force_action
        try:
            return e.step(force_action=act)
        except TypeError:
            pass
        # 3) 最後のフォールバック: 単一引数（古い step(action) 互換）
        try:
            return e.step(act)
        except Exception:
            # どうしても適用できない場合は no-op とする
            return None

    while sims_done < num_simulations:
        # 1バッチ分の葉を収集
        leaf_nodes = []
        leaf_envs = []
        leaf_keys = []
        leaf_legal = []
        virtual_counts = {}
        # min_sims を超えたらバッチサイズを縮小（より細かな early stop タイミング）
        cur_batch_size = batch_eval_size
        if post_min_batch and early_stop_enable and sims_done >= early_stop_min_sims:
            cur_batch_size = min(cur_batch_size, post_min_batch)
        for _ in range(min(cur_batch_size, num_simulations - sims_done)):
            env_copy = _fast_clone(root_env_copy)
            # --- Imperfect information support: determinization ---
            if determinize_fn is not None:
                try:
                    determinize_fn(env_copy, root_env_copy, root_player_id)
                except Exception:
                    pass  # フォールバックでそのまま
            node = root
            # 選択
            while node.children:
                node = _puct_select(node, c_puct, virtual_counts)
                # 同一バッチ中に同じ経路が過剰に選ばれないよう仮想訪問を加算
                virtual_counts[id(node)] = virtual_counts.get(id(node), 0) + 1
                _step_env_safe(env_copy, node.action)
            leaf_nodes.append(node)
            leaf_envs.append(env_copy)
            k = state_key_fn(env_copy)
            leaf_keys.append(k)
            leg = get_legal_actions_fn(env_copy)
            leaf_legal.append(leg)
        # まずキャッシュヒットを適用
        eval_indices = []
        eval_envs = []
        cached_results = {}
        for i, (k, leg) in enumerate(zip(leaf_keys, leaf_legal)):
            # TT ヒット時は保存済みの合法手を使えるよう (policy, value, legal) の形式を許容
            if TT is not None and k is not None and k in TT:
                cached = TT[k]
                if isinstance(cached, tuple) and len(cached) == 3:
                    cached_results[i] = cached  # (policy, value, legal)
                    # 法生成をスキップできる
                    leaf_legal[i] = cached[2]
                    continue
                elif isinstance(cached, tuple) and len(cached) == 2:
                    # 後方互換: (policy, value)
                    cached_results[i] = (cached[0], cached[1], leg)
                    continue
            # ここに来たら評価が必要
            if not leg:
                cached_results[i] = ({}, 0.0, [])
                continue
            eval_indices.append(i)
            eval_envs.append(leaf_envs[i])
        # 必要分だけ NN 評価
        evaluated = {}
        if eval_envs:
            # バッチ API が提供され、かつバッチサイズ>1 のとき可能なら利用
            used_batch = False
            if policy_value_batch_fn is not None and len(eval_envs) > 1:
                try:
                    outs = policy_value_batch_fn(eval_envs)
                    if isinstance(outs, list) and len(outs) == len(eval_envs):
                        for j, idx in enumerate(eval_indices):
                            evaluated[idx] = outs[j]
                        used_batch = True
                except Exception:
                    used_batch = False
            if not used_batch:
                for idx, e in zip(eval_indices, eval_envs):
                    evaluated[idx] = policy_value_fn(e)
        # マージして backup / expand
        for i in range(len(leaf_nodes)):
            node = leaf_nodes[i]
            leg = leaf_legal[i] or []
            # キャッシュ or 評価結果取得
            if i in cached_results:
                cached = cached_results[i]
                if isinstance(cached, tuple) and len(cached) == 3:
                    policy_leaf, leaf_value, leg = cached
                else:
                    policy_leaf, leaf_value = cached  # 後方互換
            elif i in evaluated:
                policy_leaf, leaf_value = evaluated[i]
                k = leaf_keys[i]
                if TT is not None and k is not None:
                    # 合法手も併せて保存し、次回ヒット時の法生成を省略
                    TT[k] = (policy_leaf, leaf_value, leg)
            else:
                # 安全側: 一様分布 + 0.0
                policy_leaf, leaf_value = ({a: 1.0 / len(leg) for a in leg} if leg else {}), 0.0

            if not leg:
                node.backup(leaf_value)
                continue

            if not policy_leaf:
                policy_leaf = {a: 1.0 / len(leg) for a in leg}
            else:
                missing_l = [a for a in leg if a not in policy_leaf]
                tot_l = sum(policy_leaf.values())
                if tot_l <= 0:
                    policy_leaf = {a: 1.0 / len(leg) for a in leg}
                else:
                    for k2 in list(policy_leaf.keys()):
                        policy_leaf[k2] /= tot_l
                    if missing_l:
                        rem_l = max(0.0, 1.0 - sum(policy_leaf.values()))
                        add_l = rem_l / len(missing_l) if missing_l else 0.0
                        for m in missing_l:
                            policy_leaf[m] = add_l
                    s3 = sum(policy_leaf.values())
                    if s3 > 0:
                        for k2 in list(policy_leaf.keys()):
                            policy_leaf[k2] /= s3
            node.expand(getattr(leaf_envs[i].game, 'turn', 0), {a: policy_leaf[a] for a in leg if a in policy_leaf})
            node.backup(leaf_value)

        sims_done += len(leaf_nodes)

        # ---------------- Early Stop 判定 ----------------
        if early_stop_enable and sims_done >= max(1, early_stop_min_sims):
            try:
                total_visits = sum(ch.visit_count for ch in root.children.values())
                if total_visits > 0 and root.children:
                    # ソートして上位2
                    vs = sorted((ch.visit_count for ch in root.children.values()), reverse=True)
                    top = vs[0]
                    second = vs[1] if len(vs) > 1 else 0
                    top_ratio = top / max(1, total_visits)
                    gap_ratio = (top - second) / max(1, total_visits)
                    # デバッグ出力（低頻度）
                    if early_stop_debug and early_stop_logger:
                        import random as _rdbg
                        if _rdbg.random() < 0.02:  # 2% サンプリング
                            try:
                                early_stop_logger.log_text(f"[mcts-early-stop-debug] sims={sims_done} top_ratio={top_ratio:.3f} gap={gap_ratio:.3f} batch={cur_batch_size} post_min_batch={post_min_batch}")
                            except Exception:
                                pass
                    if top_ratio >= early_stop_visit_ratio and gap_ratio >= early_stop_gap_ratio:
                        # ログ (低頻度サンプリング)
                        if early_stop_logger and early_stop_log_sample_rate > 0.0:
                            import random as _r
                            if _r.random() < early_stop_log_sample_rate:
                                try:
                                    early_stop_logger.log_text(f"[mcts-early-stop] sims={sims_done} top_ratio={top_ratio:.3f} gap={gap_ratio:.3f} children={len(root.children)}")
                                except Exception:
                                    pass
                        break
            except Exception:
                pass
    # root へ実行実績メタデータを付与 (呼び出し側計測用)
    try:
        root._actual_simulations = sims_done  # type: ignore[attr-defined]
        root._early_stopped = bool(sims_done < num_simulations)  # type: ignore[attr-defined]
    except Exception:
        pass
    return root


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


# 後方互換エイリアス (既存 import 対応)
__all__ = [
    "environmentEnv",
    "MCTSNode",
    "MCTSAgent",
    "PUCTNode",
    "run_puct_mcts",
]