import random
from .card import CardDeck, Card
from .player import Player
from .rules import RuleChecker 

# -----------------------------
# 大富豪のゲーム本体クラス
# -----------------------------
class Game:
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [Player(player_id=i) for i in range(num_players)]
        self.deck = CardDeck() # トランプのデッキを生成
        self.rule_checker = RuleChecker()  # ルールチェッカーを用意
        self.current_field = []  # 場に出ているカード（最後に出されたカード）
        self.turn = 0  # 現在のプレイヤー番号
        self.turn_count = 0  # ターン数
        self.passed = [False] * num_players
        self.done = False  # ゲーム終了フラグ
        self.last_player = None # 最後にカードを出したプレイヤー
        self.rankings = []  # 上がった順に記録するリスト
        self._deal_cards()  # カードを配る
        # 革命デバッグ用フラグ: True の場合 REV_EVENT 行を出力
        self.debug_revolution_trace = True
        # ゲーム終了時に革命イベントを自動出力するフラグ
        self.auto_dump_revolution_events = True
        # 見出しを一度だけ出すための内部フラグ
        self._rev_events_header_printed = False
        # Zobrist Hash 準備
        self._zkey = 0
        self._zkey_dirty = True
        self._zobrist = None  # lazy init
        self._zobrist_init_tables()
        self._zobrist_reset_full()

    def _all_others_passed(self):
        """
        最後に出したプレイヤー以外が全員パスまたは上がりならTrue
        """
        if self.last_player is None:
            return False
        return all(
            self.passed[i] or len(self.players[i].hand) == 0
            for i in range(self.num_players) if i != self.last_player
        )

    def reset(self):
        """ゲームを初期状態にリセットする"""
        # まず手札をリセット
        for player in self.players:
            player.hand = []
        self.current_field = []
        self.turn = 0
        self.turn_count = 0
        self.passed = [False] * self.num_players
        self.done = False
        self.last_player = None
        # 前回の順位情報を一時保存
        prev_rankings = self.rankings[:] if hasattr(self, 'rankings') else []
        self.rankings = []
        self._deal_cards()  # 新しい手札を配る
        self.rule_checker.reset_revolution()  # 革命状態もリセット
        self.rule_checker.clear_revolution_events()  # イベント履歴をクリア
        # 新しい手札が配られた後にカード交換を実施
        if prev_rankings and len(prev_rankings) == self.num_players:
            self.rule_checker.exchange_cards_by_rankings(self.players, prev_rankings)

        # ゲーム開始時はダイヤの3を持つ人が最初の権利を持つ（必ずしもダイヤの3を出す必要はない）
        diamond3_player = None
        for i, player in enumerate(self.players):
            for card in player.hand:
                # ダイヤは黒塗りの♦が正しい（以前は白ダイヤ♢を誤使用）
                if card.suit == '\u2666' and card.rank == 3:
                    diamond3_player = i
                    break
            if diamond3_player is not None:
                break

        if diamond3_player is not None:
            self.turn = diamond3_player
        # ハッシュ再計算
        self._zobrist_reset_full()
        # 場は空のまま、ダイヤ3を持つ人から自由に1枚出しでスタート
        return self.get_state(self.turn)  # 最初の状態を返す

    def is_valid_play(self, cards):
        """現在の場にこのカード群が出せるかどうか"""
        if cards is None or len(cards) == 0:
            return True  # パスは常に有効
        return self.rule_checker.is_valid(self.current_field, cards)  # ルール判定

    def _deal_cards(self):
        """山札をシャッフルしてプレイヤーにカードを配る"""
        self.deck.shuffle()
        for i, card in enumerate(self.deck.cards):
            self.players[i % self.num_players].hand.append(card)

    def get_state(self, player_id):
        if player_id is None:
            player_id = self.turn
        return {
            'hand': [str(card) for card in self.players[player_id].hand],
            'field': [str(card) for card in self.current_field],
            'turn': self.turn,
            'passed': self.passed,
            'turn_count': self.turn_count,
        }

    def log(self, msg):
        """
        ログ出力用メソッド。将来的なUI/ログ管理のためprintを一元化。
        """
        if getattr(self, 'silent', False):
            return
        print(msg)

    # --- スナップショット API ---
    def get_state_data(self):
        """現在のゲーム状態を最小限のプリミティブでスナップショットとして返す。

        返却フォーマット:
          {
            'num_players': int,
            'turn': int,
            'turn_count': int,
            'passed': List[bool],
            'last_player': Optional[int],
            'rankings': List[int],
            'done': bool,
            'field': List[str],
            'hands': List[List[str]],
            'revo': bool,
          }
        """
        try:
            hands = []
            for p in self.players:
                hands.append([str(c) for c in getattr(p, 'hand', [])])
            field = [str(c) for c in getattr(self, 'current_field', [])]
            data = {
                'num_players': int(self.num_players),
                'turn': int(self.turn),
                'turn_count': int(self.turn_count),
                'passed': list(self.passed),
                'last_player': None if self.last_player is None else int(self.last_player),
                'rankings': list(getattr(self, 'rankings', [])),
                'done': bool(getattr(self, 'done', False)),
                'field': field,
                'hands': hands,
                'revo': bool(getattr(getattr(self, 'rule_checker', None), 'revolution', False)),
            }
            return data
        except Exception:
            # 最低限のフォールバック
            return {
                'num_players': int(getattr(self, 'num_players', len(getattr(self, 'players', [])))) or 0,
                'turn': int(getattr(self, 'turn', 0)),
                'turn_count': int(getattr(self, 'turn_count', 0)),
                'passed': list(getattr(self, 'passed', [])) or [False]*len(getattr(self, 'players', [])),
                'last_player': getattr(self, 'last_player', None),
                'rankings': list(getattr(self, 'rankings', [])) or [],
                'done': bool(getattr(self, 'done', False)),
                'field': [],
                'hands': [[] for _ in range(len(getattr(self, 'players', [])))],
                'revo': bool(getattr(getattr(self, 'rule_checker', None), 'revolution', False)),
            }

    def set_state_data(self, data, card_lookup=None):
        """スナップショットからゲーム状態を復元する。

        Args:
          data: get_state_data で得た辞書
          card_lookup: Optional[Dict[str, Card]] 文字列表現→Card参照の辞書。
            無い場合は現在の self から可能な限り再構築（不足は from_string で補完）。
        """
        # プレイヤー数の整合
        try:
            np_ = int(data.get('num_players', self.num_players))
        except Exception:
            np_ = self.num_players
        if np_ != getattr(self, 'num_players', np_):
            # プレイヤーを作り直し
            from .player import Player
            self.num_players = np_
            self.players = [Player(player_id=i) for i in range(np_)]
        # カード辞書のフォールバック構築
        if card_lookup is None:
            card_lookup = {}
            try:
                for p in getattr(self, 'players', []):
                    for c in getattr(p, 'hand', []) or []:
                        card_lookup[str(c)] = c
                for c in getattr(self, 'current_field', []) or []:
                    card_lookup[str(c)] = c
            except Exception:
                card_lookup = {}
        # ハンド復元
        hands = data.get('hands', [[] for _ in range(self.num_players)])
        for i in range(self.num_players):
            ids = hands[i] if i < len(hands) else []
            lst = []
            for s in ids:
                c = card_lookup.get(s)
                if c is None:
                    try:
                        from .card import Card
                        c = Card.from_string(s) if hasattr(Card, 'from_string') else Card(s)
                    except Exception:
                        c = None
                if c is not None:
                    lst.append(c)
            self.players[i].hand = lst
        # フィールド復元
        field_ids = data.get('field', [])
        field_lst = []
        for s in field_ids:
            c = card_lookup.get(s)
            if c is None:
                try:
                    from .card import Card
                    c = Card.from_string(s) if hasattr(Card, 'from_string') else Card(s)
                except Exception:
                    c = None
            if c is not None:
                field_lst.append(c)
        self.current_field = field_lst
        # そのほかの状態
        try:
            self.turn = int(data.get('turn', self.turn))
        except Exception:
            pass
        try:
            self.turn_count = int(data.get('turn_count', self.turn_count))
        except Exception:
            pass
        try:
            self.passed = list(data.get('passed', self.passed))
        except Exception:
            pass
        self.last_player = data.get('last_player', self.last_player)
        try:
            self.rankings = list(data.get('rankings', self.rankings))
        except Exception:
            pass
        try:
            self.done = bool(data.get('done', getattr(self, 'done', False)))
        except Exception:
            pass
        # 革命フラグ
        try:
            if hasattr(self, 'rule_checker') and self.rule_checker is not None:
                self.rule_checker.revolution = bool(data.get('revo', bool(self.rule_checker.revolution)))
        except Exception:
            pass
        # Zobrist: 外部復元後は再計算要求（次回取得時にO(N)一度だけ）
        try:
            self._zkey_dirty = True
        except Exception:
            pass

    def step(self, player_id, action_cards):
        """
        1ターン進める。action_cards: 出すカードリスト or None（パス）
        戻り値: (状態, 終了フラグ, 場リセットフラグ, リセット理由)
        """
        # A-1: 入力正規化（パスは None、出しは List[str] or List[Card]）
        if action_cards == [] or action_cards == "pass":
            action_cards = None
        elif isinstance(action_cards, str):
            action_cards = [action_cards]
        elif isinstance(action_cards, tuple):
            action_cards = list(action_cards)
        elif action_cards is not None and not isinstance(action_cards, list):
            action_cards = None
        player = self.players[self.turn]
        # 場が空
        if not self.current_field:
            result = self._handle_action(player_id, player, action_cards, empty_field=True)
            return result
        # 場が空でない
        result = self._handle_action(player_id, player, action_cards, empty_field=False)
        return result

    def _handle_action(self, player_id, player, action_cards, empty_field):
        """
        1ターン分のアクション処理を行う。
        - 入力検証
        - ルール判定
        - カード出し
        - 特殊ルール処理
        - パス処理
        - 上がり判定
        - 場リセット判定
        - ターン進行
        empty_field: 場が空かどうか
        """
        valid = False
        reset_happened = False
        is_first_turn = empty_field and (self.turn_count == 0 and self.last_player is None)
        card_objs = self._find_hand_cards(player, action_cards) if action_cards else None

        # 出すカードの検証・ルール判定
        if card_objs is not None and len(card_objs) > 0:
            if empty_field:
                valid = is_first_turn or self.rule_checker.is_valid_move(card_objs, self.current_field)
            else:
                valid = len(card_objs) == len(self.current_field) and self.rule_checker.is_valid_move(card_objs, self.current_field)
        # カードを出す処理
        if valid:
            # 表示用に役を分類してジョーカー代用情報を確定させる（表示のため）
            # classify_combo は必要に応じて joker_as_* を設定するが、場出しが確定してから行うので副作用は許容範囲
            try:
                _ = self.rule_checker.classify_combo(card_objs)
            except Exception:
                pass
            # ゲーム中の出力を「Player X played: ...」形式に（解決済みの card_objs を使用）
            played_str = ', '.join(str(c) for c in card_objs) if card_objs else ''
            # 1) 場に反映
            self._play_cards(player, card_objs)
            self.last_player = self.turn
            # 2) 革命判定のみ先に実行 (出力順制御のため分離)
            prev_rev = self.rule_checker.revolution
            rev_triggered = False
            try:
                # 四枚同ランク or 5枚以上階段で革命トグル (イベント記録付き)
                rev_triggered = self.rule_checker.check_revolution(
                    card_objs,
                    player_id=self.turn,
                    turn_count=self.turn_count
                )
            except Exception:
                pass
            new_rev = self.rule_checker.revolution
            rev_changed = (prev_rev != new_rev)
            if rev_changed:
                # Zobrist: 革命フラグ切替
                self._zkey ^= self._zobrist['REVO']
            # トグル仕様: 変化したら [+REV] / [-REV]
            rev_flag = ' (REV)' if new_rev else ''
            trig = ''
            if rev_changed:
                trig = ' [+REV]' if new_rev else ' [-REV]'
            # 3) プレイログ (リセット系より先に必ず出す)
            #self.log(f"Player {self.turn} played: {played_str}{rev_flag}{trig}")
            # 革命発生メッセージ (オプション) - 従来互換
            if rev_triggered:
                #self.log(f"革命発生! 現在の革命状態: {self.rule_checker.revolution}")
                if self.debug_revolution_trace:
                    # 直近イベントのみ取り出し
                    ev = self.rule_checker.revolution_events[-1] if self.rule_checker.revolution_events else None
                    if ev:
                        """self.log(
                            "REV_EVENT "
                            f"turn={ev['turn']} player={ev['player']} reason={ev['reason']} "
                            f"old={ev['old_state']} new={ev['new_state']} cards={ev['cards']} meta={ev['meta']}"
                        )"""
            # 4) その他特殊ルール: 階段表示 / 8切り / ジョーカー流し
            try:
                # 階段 (表示のみ)
                if self.rule_checker.is_straight(card_objs):
                    #self.log(f"Player {self.turn} が階段を出しました: {[str(c) for c in card_objs]}")
                    pass
            except Exception:
                pass
            # 8切り
            if self.rule_checker.is_8cut(card_objs):
                #self.log(f"8切り発動 by Player {self.turn}!")
                self.last_player = self.turn
                self._reset_field()
                return self.get_state(self.turn), False, True, "eight_cut"
            # ジョーカー流し
            if len(card_objs) == 1 and card_objs[0].is_joker:
                self.last_player = self.turn
                self._reset_field()
                return self.get_state(self.turn), False, True, "joker_cut"
        else:
            # ゲーム中の出力を「Player X passed.」形式に
            rev_flag = ' (REV)' if self.rule_checker.revolution else ''
            #self.log(f"Player {self.turn} passed.{rev_flag}")
            action_cards = None
            if not self.passed[self.turn]:
                # Zobrist: pass フラグを立てる
                self._zkey ^= self._zobrist['PASSED'][self.turn]
                self.passed[self.turn] = True
            # 最後に出したプレイヤー以外が全員パス → 場リセット
            if self._all_others_passed():
                self._reset_field()
                reset_happened = True
                self.turn = self.last_player
                return self.get_state(self.turn), False, reset_happened, "all_pass"
        if valid:
            self.last_player = self.turn
        # 上がり判定
        if self._check_agari(player, player_id):
            return self.get_state(self.turn), True, False, None

        # 全員パス or 全員上がりで場リセット
        if not empty_field and self._all_others_passed():
            self._reset_field()
            reset_happened = True
            if self.last_player is not None:
                self.turn = self.last_player
            return self.get_state(self.turn), False, reset_happened, "all_pass"

        # リセット直後は再度 same player に戻る
        if reset_happened:
            return self.get_state(self.turn), False, True, None

        # どの分岐にも入らなかった場合、通常ターン進行
        self._advance_turn()
        return self.get_state(self.turn), False, False, None

    def _advance_turn(self):
        """次のプレイヤーにターンを進める（手札がない場合はスキップ）"""
        old_turn = self.turn
        next_turn = (self.turn + 1) % self.num_players
        skip_count = 0
        while len(self.players[next_turn].hand) == 0:
            next_turn = (next_turn + 1) % self.num_players
            skip_count += 1
            if skip_count > self.num_players:
                break
        if next_turn != self.turn:
            self.turn_count += 1
        # Zobrist: turn 切替
        if next_turn != old_turn:
            self._zkey ^= self._zobrist['TURN'][old_turn]
            self._zkey ^= self._zobrist['TURN'][next_turn]
        self.turn = next_turn

    def _handle_special_rules(self, card_objs):
        """
        革命・階段・8切り・ジョーカー流し等の特殊ルール処理をまとめて行う。
        リセットが発生した場合はTrue, 戻り値として次状態を返す。
        """
        # 革命
        if self.rule_checker.check_revolution(card_objs):
            #self.log(f"革命発生! 現在の革命状態: {self.rule_checker.revolution}")
            pass  # コメントのみだと構文エラーになるため明示 no-op
        # 8切り
        if self.rule_checker.is_8cut(card_objs):
            self.last_player = self.turn
            self._reset_field()
            return True, (self.get_state(self.turn), False, True, "eight_cut")
        # ジョーカー流し（ジョーカー1枚出しのみ）
        if len(card_objs) == 1 and card_objs[0].is_joker:
            self.last_player = self.turn
            self._reset_field()
            return True, (self.get_state(self.turn), False, True, "joker_cut")
        return False, None

    def _check_agari(self, player, player_id):
        """
        上がり判定と順位付けを行う。
        プレイヤーが上がった場合や、全員の順位が確定した場合にTrueを返す。
        """
        if len(player.hand) == 0 and player_id not in self.rankings:
            self.rankings.append(player_id)
            # Zobrist: ランキングに入ったプレイヤーをマーク
            try:
                self._zkey ^= self._zobrist['RANKED'][int(player_id)]
            except Exception:
                pass
        if len(self.rankings) == self.num_players - 1:
            last_player = [i for i in range(self.num_players) if i not in self.rankings][0]
            self.rankings.append(last_player)
            try:
                self._zkey ^= self._zobrist['RANKED'][int(last_player)]
            except Exception:
                pass
            self.done = True
            if self.auto_dump_revolution_events:
                self.dump_revolution_events()
            return True
        return False

    # --- 革命イベントログ出力ユーティリティ -----------------------
    def dump_revolution_events(self):
        """現在のゲームに記録された革命イベントログを整形して出力する。"""
        events = self.rule_checker.get_revolution_events()
        if not events:
            #self.log("[REV_EVENTS] (none)")
            return
        if not self._rev_events_header_printed:
            #self.log("[REV_EVENTS] turn player reason old->new size cards meta")
            self._rev_events_header_printed = True
        for ev in events:
            self.log(
                #f"[REV_EVENTS] {ev['turn']} P{ev['player']} {ev['reason']} "
                #f"{ev['old_state']}->{ev['new_state']} {ev['size']} {ev['cards']} {ev['meta']}"
            )

    def print_revolution_events(self):
        """エイリアスメソッド (dump_revolution_events と同じ)。"""
        self.dump_revolution_events()

    # --- 補助メソッド ---
    def _find_hand_cards(self, player, action_cards):
        """
        手札から action_cards に該当する Card オブジェクトのリストを原子的に解決して返す。
        - action_cards は List[Card] もしくは List[str]（A-1 の前提）
        - すべての枚数が手札に存在しない場合は None を返し、手札を一切変更しない（原子的）
        """
        if not action_cards:
            return None

        # ケース1: すでに手札中の Card オブジェクトが渡されている場合（同一インスタンスかを確認）
        if all(isinstance(a, Card) for a in action_cards):
            # 同一オブジェクトが重複参照されていないか確認
            if len({id(a) for a in action_cards}) != len(action_cards):
                return None
            # 全カードが手札内のオブジェクトか確認（同一性 'is' で）
            for a in action_cards:
                if not any(h is a for h in player.hand):
                    return None
            # そのまま返す（呼び出し側で場に出す）
            return list(action_cards)

        # ケース2: 文字列指定の場合はマルチセットで厳密に確認
        if all(isinstance(a, str) for a in action_cards):
            # 手札側のカウント
            hand_counts = {}
            idx_map = {}
            for idx, h in enumerate(player.hand):
                key = str(h)
                hand_counts[key] = hand_counts.get(key, 0) + 1
                idx_map.setdefault(key, []).append(idx)

            # 要求側のカウント
            req_counts = {}
            for a in action_cards:
                req_counts[a] = req_counts.get(a, 0) + 1

            # 不足があれば原子的に不可
            for k, v in req_counts.items():
                if v > hand_counts.get(k, 0):
                    return None

            # 実体へマッピング（各文字列ごとに必要数のインデックスを取得）
            taken_indices = []
            for a in action_cards:
                taken_indices.append(idx_map[a].pop())
            taken_objs = [player.hand[i] for i in taken_indices]
            return taken_objs

        # 想定外の型が混在している場合は無効
        return None

    def _play_cards(self, player, card_objs):
        """
        カードを場に出し、手札から原子的に削除し、場の状態を更新。
        - card_objs は手札中の実オブジェクト（_find_hand_cards で解決済み）
        - 1枚でも存在しなければ削除は一切行わない（保険）
        """
        # まず手札内存在チェック（同一性）
        indices = []
        used = [False] * len(player.hand)
        for co in card_objs:
            found_idx = -1
            for i, h in enumerate(player.hand):
                if not used[i] and (h is co):
                    found_idx = i
                    used[i] = True
                    break
            if found_idx == -1:
                # 原子性のため、何もしない
                self.log("[WARNING] 原子的削除失敗: 指定カードが手札に見つかりませんでした。処理を中止します。")
                return
            indices.append(found_idx)

        # Zobrist: まず現在の場カードを外す
        if getattr(self, 'current_field', None):
            for c in self.current_field:
                try:
                    cid = self._card_id(c)
                    self._zkey ^= self._zobrist['FIELD'][cid]
                except Exception:
                    pass
        # 場更新（表示用に浅いコピー）
        self.current_field = card_objs[:]
        # Zobrist: 手札→場への移動を反映（手札から外し、場に追加）
        try:
            pid = player.player_id
        except Exception:
            pid = None
        for c in card_objs:
            try:
                cid = self._card_id(c)
                if pid is not None:
                    self._zkey ^= self._zobrist['HAND'][pid][cid]
                self._zkey ^= self._zobrist['FIELD'][cid]
            except Exception:
                pass
        # 実削除（後ろから）
        for idx in sorted(indices, reverse=True):
            del player.hand[idx]
        # パス情報リセット
        # Zobrist: pass を全解除
        for i, f in enumerate(self.passed):
            if f:
                try:
                    self._zkey ^= self._zobrist['PASSED'][i]
                except Exception:
                    pass
        self.passed = [False] * self.num_players

    def _reset_field(self):
        """場をリセットし、パス情報もリセット"""
        # Zobrist: 場カードを全て外す
        for c in self.current_field:
            try:
                cid = self._card_id(c)
                self._zkey ^= self._zobrist['FIELD'][cid]
            except Exception:
                pass
        self.current_field = []
        # Zobrist: pass を全解除
        for i, f in enumerate(self.passed):
            if f:
                try:
                    self._zkey ^= self._zobrist['PASSED'][i]
                except Exception:
                    pass
        self.passed = [False] * self.num_players
        self.turn_count += 1
        # 場リセット時は必ず最後に出したプレイヤーから再開
        if self.last_player is not None:
            if self.turn != self.last_player:
                # Zobrist: turn 切替
                try:
                    self._zkey ^= self._zobrist['TURN'][self.turn]
                    self._zkey ^= self._zobrist['TURN'][self.last_player]
                except Exception:
                    pass
            self.turn = self.last_player
        #self.log("--- 場がリセットされました ---")

    # --- Zobrist Hash 実装 ----------------------------------
    def _zobrist_init_tables(self):
        if self._zobrist is not None:
            return
        import random as _r
        _r.seed(0xC0FFEE)  # 安定性のため固定シード（必要に応じ変更）
        num_players = max(1, int(getattr(self, 'num_players', 4)))
        NUM_CARDS = 53  # 52枚 + Joker(=52)
        HAND = [[_r.getrandbits(64) for _ in range(NUM_CARDS)] for __ in range(num_players)]
        FIELD = [_r.getrandbits(64) for _ in range(NUM_CARDS)]
        PASSED = [_r.getrandbits(64) for _ in range(num_players)]
        TURN = [_r.getrandbits(64) for _ in range(num_players)]
        RANKED = [_r.getrandbits(64) for _ in range(num_players)]
        REVO = _r.getrandbits(64)
        self._zobrist = {
            'HAND': HAND,
            'FIELD': FIELD,
            'PASSED': PASSED,
            'TURN': TURN,
            'RANKED': RANKED,
            'REVO': REVO,
        }

    def _card_id(self, c):
        try:
            if getattr(c, 'is_joker', False):
                return 52
            suit = getattr(c, 'suit', '\u2660')
            suit_map = {'\u2660':0, '\u2665':1, '\u2666':2, '\u2663':3, '♠':0, '♥':1, '♦':2, '♣':3, 'S':0,'H':1,'D':2,'C':3}
            sid = suit_map.get(suit, 0)
            r = int(getattr(c, 'rank', 1)) - 1
            if r < 0: r = 0
            if r > 12: r = 12
            return sid * 13 + r
        except Exception:
            return 52

    def _zobrist_reset_full(self):
        # テーブルは lazy init 済み前提
        self._zkey = 0
        # hands
        for pidx, p in enumerate(getattr(self, 'players', [])):
            for c in getattr(p, 'hand', []) or []:
                try:
                    cid = self._card_id(c)
                    self._zkey ^= self._zobrist['HAND'][pidx][cid]
                except Exception:
                    pass
        # field
        for c in getattr(self, 'current_field', []) or []:
            try:
                cid = self._card_id(c)
                self._zkey ^= self._zobrist['FIELD'][cid]
            except Exception:
                pass
        # passed
        for i, f in enumerate(getattr(self, 'passed', []) or []):
            if f:
                self._zkey ^= self._zobrist['PASSED'][i]
        # rankings (フラグ化)
        ranks = set(getattr(self, 'rankings', []) or [])
        for i in range(len(getattr(self, 'players', []))):
            if i in ranks:
                self._zkey ^= self._zobrist['RANKED'][i]
        # revo
        try:
            if bool(getattr(getattr(self, 'rule_checker', None), 'revolution', False)):
                self._zkey ^= self._zobrist['REVO']
        except Exception:
            pass
        # turn
        try:
            self._zkey ^= self._zobrist['TURN'][int(getattr(self, 'turn', 0))]
        except Exception:
            pass
        self._zkey_dirty = False

    def get_zobrist_key(self, recompute_if_dirty=True) -> int:
        if recompute_if_dirty and getattr(self, '_zkey_dirty', False):
            self._zobrist_init_tables()
            self._zobrist_reset_full()
        return int(self._zkey)
