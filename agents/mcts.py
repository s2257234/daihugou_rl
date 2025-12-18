import copy
import math
import random
try:
    # Optional fast helpers (Cython acceleration). Safe fallback if missing.
    from agents.mcts_fast import (
        puct_select_index_fast as _puct_select_index_fast,
        puct_backup_generic as _puct_backup_generic,
        puct_backup_scalar as _puct_backup_scalar,
    )
except Exception:
    _puct_select_index_fast = None
    _puct_backup_generic = None
    _puct_backup_scalar = None

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

    value_sum: 累積価値（各ノードの現在手番プレイヤー視点で加算）。平均値 = value_sum / visit_count。
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

    # ---- バックアップ (各ノードの現在手番視点で加算、符号反転なし) ----
    def backup(self, leaf_value):
        """
        leaf_value:
          - float: 単一スカラー（互換）
          - list/tuple: 各プレイヤーの値（インデックス=player_id）
          - dict: {player_id: value}
        いずれの場合も、このノードの to_play に対応する成分を加算する。
        値域は [0,1] を想定（BCE確率）。
        可能ならCython高速版でバックアップを行う。
        """
        # Fast path: prefer Cython implementations when available
        try:
            if _puct_backup_scalar is not None and isinstance(leaf_value, (int, float)):
                _puct_backup_scalar(self, float(leaf_value))
                return
            if _puct_backup_generic is not None:
                _puct_backup_generic(self, leaf_value)
                return
        except Exception:
            pass
        # Fallback: pure Python backup
        def _value_for_pid(v, pid: int) -> float:
            try:
                if isinstance(v, dict):
                    return float(v.get(pid, 0.0))
                if isinstance(v, (list, tuple)):
                    return float(v[pid]) if 0 <= pid < len(v) else 0.0
                return float(v)
            except Exception:
                return 0.0

        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += _value_for_pid(leaf_value, getattr(node, 'to_play', 0))
            node = node.parent


def _puct_select(node: PUCTNode, c_puct: float, virtual_counts=None) -> PUCTNode:
    """子ノードの中から PUCT スコア最大のものを返す"""
    if virtual_counts is None:
        virtual_counts = {}
    ch_vals = node.children
    if not ch_vals:
        return None
    # Fast path: Cython implementation if available
    if _puct_select_index_fast is not None:
        children = list(ch_vals.values())
        priors = [ch.prior for ch in children]
        values = [(0.0 if ch.visit_count == 0 else ch.value_sum / ch.visit_count) for ch in children]
        visits = [ch.visit_count for ch in children]
        vcounts = [virtual_counts.get(id(ch), 0) for ch in children]
        idx = _puct_select_index_fast(priors, values, visits, vcounts, float(c_puct), -1)
        if 0 <= idx < len(children):
            return children[idx]
    # NumPy vectorized path (when Cython is unavailable)
    try:
        import numpy as _np
        children = list(ch_vals.values())
        n = len(children)
        if n == 0:
            return None
        vcounts = _np.fromiter((virtual_counts.get(id(ch), 0) for ch in children), dtype=_np.int64, count=n)
        visits = _np.fromiter((ch.visit_count for ch in children), dtype=_np.int64, count=n)
        priors = _np.fromiter((ch.prior for ch in children), dtype=_np.float64, count=n)
        # value = value_sum / max(1, visit_count)
        vsum = _np.fromiter((ch.value_sum for ch in children), dtype=_np.float64, count=n)
        denom = _np.maximum(1, visits)
        values = vsum / denom
        total_visits = int(_np.maximum(1, (visits + vcounts).sum()))
        sqrt_total = math.sqrt(total_visits)
        u = (float(c_puct) * priors * sqrt_total) / (1.0 + visits + vcounts)
        score = values + u
        idx = int(score.argmax())
        return children[idx]
    except Exception:
        pass
    # Fallback: pure-Python loop
    def vc(ch):
        return virtual_counts.get(id(ch), 0)
    total_visits = max(1, sum(child.visit_count + vc(child) for child in ch_vals.values()))
    best, best_score = None, -1e18
    sqrt_total = math.sqrt(total_visits)
    for child in ch_vals.values():
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
                  early_stop_logger=None,
                  profile: bool = False,
                  profile_output: str | None = None,
                  enable_legal_cache: bool = True,
                  legal_cache_max_size: int | None = 4096,
                  # virtual loss options (for single-process batched MCTS)
                  enable_virtual_loss: bool = True,
                  virtual_loss_count: int = 1,
                  virtual_loss_value: float = -100.0):
    """AlphaZero 風 PUCT MCTS 実行 (正規化 & 欠損補完対応版)。"""
    import time as _time

    run_start = _time.perf_counter()

    # Optional: full cProfile for the entire run
    _prof = None
    if profile:
        try:
            import cProfile as _cProfile
            _prof = _cProfile.Profile()
            _prof.enable()
        except Exception:
            _prof = None

    # timing counters for legal action computation
    legal_calls = 0
    legal_total_s = 0.0

    # ルート合法手
    t0_lr = _time.perf_counter()
    legal_root = get_legal_actions_fn(root_env_copy)
    legal_total_s += (_time.perf_counter() - t0_lr)
    legal_calls += 1
    if not legal_root:
        legal_root = ["pass"]
    policy_root, root_value = policy_value_fn(root_env_copy)
    # policy_root を合法手に合わせて補完・正規化
    if not policy_root:
        # legal_root の要素を hashable に変換（list → tuple）
        policy_root = {(tuple(a) if isinstance(a, list) else a): 1.0 / len(legal_root) for a in legal_root}
    else:
        # 欠損を均等割当（legal_root の要素を tuple に変換して比較）
        missing = [(tuple(a) if isinstance(a, list) else a) for a in legal_root 
                   if (tuple(a) if isinstance(a, list) else a) not in policy_root]
        total = sum(policy_root.values())
        if total <= 0:
            policy_root = {(tuple(a) if isinstance(a, list) else a): 1.0 / len(legal_root) for a in legal_root}
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
    # legal_root を tuple に変換して expand に渡す
    legal_root_hashable = [(tuple(a) if isinstance(a, list) else a) for a in legal_root]
    root.expand(root_player_id, {a: policy_root[a] for a in legal_root_hashable if a in policy_root})
    root.visit_count = 1
    # ルート初期値は「ルート手番プレイヤー視点」の成分を使用
    try:
        if isinstance(root_value, dict):
            root.value_sum = float(root_value.get(root_player_id, 0.0))
        elif isinstance(root_value, (list, tuple)):
            rv = root_value[root_player_id] if 0 <= root_player_id < len(root_value) else 0.0
            root.value_sum = float(rv)
        else:
            root.value_sum = float(root_value)
    except Exception:
        root.value_sum = 0.0

    # Dirichlet ノイズ
    if add_dirichlet and root.children:
        actions = list(root.children.keys())
        noises = [random.gammavariate(dirichlet_alpha, 1.0) for _ in actions]
        s = sum(noises)
        noises = [n / s for n in noises]
        for a, n in zip(actions, noises):
            child = root.children[a]
            child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon

    # --- クローンプール + スナップショット方式 ---
    # 1) ルートゲームのスナップショットとカード参照辞書
    try:
        g_root = root_env_copy.game
        card_lookup = {}
        for pl in getattr(g_root, 'players', []):
            for c in getattr(pl, 'hand', []) or []:
                card_lookup[str(c)] = c
        for c in getattr(g_root, 'current_field', []) or []:
            card_lookup[str(c)] = c
        # ルート状態のスナップショット
        snapshot_root = None
        try:
            snapshot_root = g_root.get_state_data()  # 実装が無ければ except
        except Exception:
            snapshot_root = None
    except Exception:
        card_lookup = {}
        snapshot_root = None

    # 2) プール初期化（batch_eval_size 個）
    _clone_pool = []
    _pool_size = max(1, int(batch_eval_size or 1))
    try:
        for _ in range(_pool_size):
            base_env = copy.copy(root_env_copy)
            g = getattr(root_env_copy, 'game', None)
            if g is None:
                _clone_pool.append(base_env)
                continue
            g_new = copy.copy(g)
            # Player オブジェクトは新規に作成し、手札は空で初期化（restore 時に埋め戻す）
            try:
                from game.player import Player as _P
                np_ = getattr(g, 'num_players', len(getattr(g, 'players', [])))
                players_new = []
                for i in range(np_):
                    psrc = g.players[i]
                    pn = _P(player_id=getattr(psrc, 'player_id', i), name=getattr(psrc, 'name', None))
                    pn.hand = []
                    players_new.append(pn)
                g_new.players = players_new
            except Exception:
                # フォールバック: 既存プレイヤーを浅いコピー
                players_new = []
                for p in getattr(g, 'players', []):
                    p_new = copy.copy(p)
                    p_new.hand = []
                    players_new.append(p_new)
                g_new.players = players_new
            # フィールドは常にルートと同じ状態からスタートさせる。
            # これにより、スナップショット復元に失敗した場合でも「場が空」と誤認しない。
            try:
                g_new.current_field = list(getattr(g, 'current_field', []) or [])
            except Exception:
                g_new.current_field = []
            g_new.passed = list(getattr(g, 'passed', []))
            g_new.rankings = list(getattr(g, 'rankings', []))
            # RuleChecker は新規
            try:
                from game.rules import RuleChecker
                rc = RuleChecker()
                rc.revolution = bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False))
                g_new.rule_checker = rc
            except Exception:
                pass
            base_env.game = g_new
            _clone_pool.append(base_env)
    except Exception:
        _clone_pool = []

    def _fast_clone(env):
        # プールから 1 つ取り出し、スナップショット復元
        if _clone_pool:
            env_new = _clone_pool.pop()
            try:
                if snapshot_root is not None and hasattr(env_new.game, 'set_state_data'):
                    env_new.game.set_state_data(snapshot_root, card_lookup)
                else:
                    # フォールバック: 旧軽量コピー
                    raise RuntimeError('no-snapshot')
                return env_new
            except Exception:
                pass
            # フォールバック: 旧軽量コピー
        # ルートと同じ軽量コピー方針
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
                'rev_enable_four_kind','rev_four_kind_allow_joker','rev_four_kind_exact',
                'rev_enable_straight','rev_straight_min_len','rev_straight_allow_joker',
            ):
                try:
                    setattr(rc, name, getattr(rc_src, name))
                except Exception:
                    pass
        g_new.rule_checker = rc
        try:
            g_new.deck = getattr(g, 'deck', None)
        except Exception:
            g_new.deck = None
        env_new = copy.copy(env)
        env_new.game = g_new
        return env_new

    # トランスポジションテーブル（None の場合はキャッシュ無効）
    TT = transposition_table if transposition_table is not None else None
    # キー使用の有無（不要なら計算コストをゼロに）
    use_keys = (TT is not None) or bool(enable_legal_cache)

    # --------------------------------------
    # 事前: キャッシュ用キー関数（ビットマスク化）
    # --------------------------------------
    if state_key_fn is None:
        if use_keys:
            def state_key_fn_default(e):
                try:
                    g = e.game
                    # Zobrist ハッシュがあれば優先
                    if hasattr(g, 'get_zobrist_key'):
                        return g.get_zobrist_key(True)
                except Exception:
                    pass
                # フォールバック: 旧ビットマスク方式
                try:
                    g = e.game
                    turn = getattr(g, 'turn', 0)
                    def _cid(c):
                        try:
                            if getattr(c, 'is_joker', False):
                                return 52
                            suit = getattr(c, 'suit', 'S')
                            suit_map = {'S':0, 'H':1, 'D':2, 'C':3, '\u2660':0, '\u2665':1, '\u2666':2, '\u2663':3}
                            sid = suit_map.get(suit, 0)
                            r = int(getattr(c, 'rank', 1)) - 1
                            if r < 0: r = 0
                            if r > 12: r = 12
                            return sid * 13 + r
                        except Exception:
                            return 52
                    def _mask(iter_cards):
                        m = 0
                        for c in iter_cards:
                            m |= (1 << _cid(c))
                        return m
                    field_mask = _mask(getattr(g, 'current_field', []))
                    hands_mask = tuple(_mask(getattr(p, 'hand', [])) for p in getattr(g, 'players', []))
                    passed = tuple(bool(x) for x in getattr(g, 'passed', []))
                    rankings = tuple(getattr(g, 'rankings', []))
                    revo = bool(getattr(getattr(g, 'rule_checker', None), 'revolution', False))
                    return (turn, field_mask, hands_mask, passed, rankings, revo)
                except Exception:
                    return None
            state_key_fn = state_key_fn_default
        else:
            def state_key_fn(_e):
                return None

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

    # 例外分岐を一度だけ解決してキャッシュする高速版
    _step_mode = {"mode": None}
    def _step_env_fast(e, act):
        m = _step_mode["mode"]
        if m is None:
            # 1回だけ解決
            try:
                e.step(return_info=False, external_action=act, simulate=True)
                _step_mode["mode"] = "ext"
                return
            except Exception:
                pass
            try:
                e.step(force_action=act)
                _step_mode["mode"] = "force"
                return
            except Exception:
                pass
            _step_mode["mode"] = "positional"
            try:
                e.step(act)
                return
            except Exception:
                # 解決失敗時は no-op
                return
        else:
            try:
                if m == "ext":
                    e.step(return_info=False, external_action=act, simulate=True)
                elif m == "force":
                    e.step(force_action=act)
                else:
                    e.step(act)
                return
            except Exception:
                # 失敗したらモードをリセットして再解決
                _step_mode["mode"] = None
                return _step_env_fast(e, act)

    # ラン中の合法手キャッシュ（TTに無い近傍を節約）
    # オプションで LRU (OrderedDict) を使い上限を設ける
    from collections import OrderedDict
    if enable_legal_cache:
        _legal_cache = OrderedDict()
    else:
        _legal_cache = None
    legal_cache_hits = 0
    while sims_done < num_simulations:
        # track virtual losses applied in this batch so we can revert after real backups
        virtual_applied = []  # list of (node, cnt, val_delta)
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
                    # Zobrist: 外部から手札を書き換えた可能性があるため再計算要求
                    try:
                        if hasattr(env_copy, 'game'):
                            setattr(env_copy.game, '_zkey_dirty', True)
                    except Exception:
                        pass
                except Exception:
                    pass  # フォールバックでそのまま
            node = root
            # 選択
            while node.children:
                node = _puct_select(node, c_puct, virtual_counts)
                # 同一バッチ中に同じ経路が過剰に選ばれないよう仮想訪問を加算
                virtual_counts[id(node)] = virtual_counts.get(id(node), 0) + 1
                _step_env_fast(env_copy, node.action)
                # 仮想損失を適用して同一ノードがバッチに偏らないようにする
                if enable_virtual_loss:
                    try:
                        vc = int(virtual_loss_count)
                        vv = float(virtual_loss_value) * vc
                        node.visit_count += vc
                        node.value_sum += vv
                        virtual_applied.append((node, vc, vv))
                    except Exception:
                        pass
            leaf_nodes.append(node)
            leaf_envs.append(env_copy)
            k = state_key_fn(env_copy)
            leaf_keys.append(k)
            # 可能なら TT またはラン内キャッシュから取得
            leg = None
            if TT is not None and k is not None and (k in TT):
                cached = TT.get(k)
                if isinstance(cached, tuple) and len(cached) == 3:
                    leg = cached[2]
            if leg is None:
                # キャッシュ利用はキーが有効な場合のみ
                if _legal_cache is not None and k is not None and k in _legal_cache:
                    leg = _legal_cache[k]
                    legal_cache_hits += 1
                else:
                    # measure legal actions cost
                    try:
                        t0_leg = _time.perf_counter()
                        leg = get_legal_actions_fn(env_copy) or []
                        legal_total_s += (_time.perf_counter() - t0_leg)
                        legal_calls += 1
                    except Exception:
                        # fall back
                        leg = get_legal_actions_fn(env_copy) or []
                    # キャッシュに保存 (キーが None の場合はキャッシュしない)
                    if _legal_cache is not None and k is not None:
                        try:
                            _legal_cache[k] = leg
                            # サイズ制限 (LRU eviction)
                            if legal_cache_max_size is not None and len(_legal_cache) > legal_cache_max_size:
                                _legal_cache.popitem(last=False)
                        except Exception:
                            # 安全のため例外は無視
                            pass
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
                    # 事前計算済みの合法手を同じ順序で渡す（再計算を回避）
                    eval_legals = [leaf_legal[idx] for idx in eval_indices]
                    outs = policy_value_batch_fn(eval_envs, eval_legals)
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
                # 安全側: 一様分布 + 0.0（leg の要素を tuple に変換）
                policy_leaf, leaf_value = ({(tuple(a) if isinstance(a, list) else a): 1.0 / len(leg) for a in leg} if leg else {}), 0.0

            if not leg:
                node.backup(leaf_value)
                continue
            if not policy_leaf:
                # leg の要素を hashable に変換（list → tuple）
                policy_leaf = {(tuple(a) if isinstance(a, list) else a): 1.0 / len(leg) for a in leg}
            else:
                # 欠損を均等割当（leg の要素を tuple に変換して比較）
                missing_l = [(tuple(a) if isinstance(a, list) else a) for a in leg 
                             if (tuple(a) if isinstance(a, list) else a) not in policy_leaf]
                tot_l = sum(policy_leaf.values())
                if tot_l <= 0:
                    policy_leaf = {(tuple(a) if isinstance(a, list) else a): 1.0 / len(leg) for a in leg}
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
            # leg を tuple に変換して expand に渡す
            leg_hashable = [(tuple(a) if isinstance(a, list) else a) for a in leg]
            node.expand(getattr(leaf_envs[i].game, 'turn', 0), {a: policy_leaf[a] for a in leg_hashable if a in policy_leaf})
            node.backup(leaf_value)

        # 適用した仮想損失をリバート（正しい統計に戻す）
        if enable_virtual_loss and virtual_applied:
            for nd, vc, vv in virtual_applied:
                try:
                    nd.visit_count -= vc
                    nd.value_sum -= vv
                except Exception:
                    pass

        sims_done += len(leaf_nodes)

        # 使用したクローンをプールへ戻す（次バッチで再利用）
        try:
            if _clone_pool is not None:
                for e in leaf_envs:
                    _clone_pool.append(e)
        except Exception:
            pass

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
    # profile: dump / summary (SUPPRESSED: [PROFILE][mcts] line removed per request)
    # ここでは内部統計を保持したい場合に備え、値だけを root に埋め込む (ログ出力なし)
    try:
        run_end = _time.perf_counter()
        run_s = run_end - run_start
        pct = (legal_total_s / run_s) if run_s > 0 else 0.0
        cache_size = len(_legal_cache) if _legal_cache is not None else 0
        try:
            root._profile_stats = {
                'legal_calls': legal_calls,
                'legal_total_s': legal_total_s,
                'run_s': run_s,
                'legal_time_ratio': pct,
                'legal_cache_hits': legal_cache_hits,
                'legal_cache_size': cache_size,
            }
        except Exception:
            pass
    except Exception:
        pass

    # profiler dump
    if _prof is not None:
        try:
            _prof.disable()
            if profile_output:
                try:
                    import pstats as _pstats
                    _pstats.Stats(_prof).sort_stats('cumtime').dump_stats(profile_output)
                except Exception:
                    pass
            else:
                # write short top-10 cumtime to events.log
                try:
                    import io, pstats as _pstats
                    s = io.StringIO()
                    ps = _pstats.Stats(_prof, stream=s).sort_stats('cumtime')
                    ps.print_stats(10)
                    txt = s.getvalue()
                    # append to events.log
                    import os, datetime
                    log_dir = 'logs'
                    os.makedirs(log_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                        f.write(f"[{ts}] [PROFILE][cprofile]\n")
                        for line in txt.splitlines():
                            f.write(f"[{ts}] {line}\n")
                except Exception:
                    pass
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