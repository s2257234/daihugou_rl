from functools import lru_cache


@lru_cache(maxsize=100000)
def _classify_combo_cached(revo_flag: bool, cards_key: tuple):
    """副作用なしの役分類をLRUキャッシュで提供する純関数。

    入力:
      - revo_flag: 革命状態 (strength 解釈に影響)
      - cards_key: 並び順非依存の正規化キー。
        形式は tuple( (is_joker:int, suit:str|None, rank:int|None), ... ) をソート済みで格納。

    出力:
      - dict(type,size,rank,ranks,jokers,strength) | None
    """
    cards = list(cards_key)
    n = len(cards)
    if n == 0:
        return None
    jokers = [c for c in cards if c[0] == 1]
    non_jokers = [c for c in cards if c[0] == 0]

    def _strength_from_rank(r: int) -> int:
        if r == 1:
            return 13
        if r == 2:
            return 14
        return r - 2

    # Joker単体
    if n == 1 and len(jokers) > 0:
        return {
            'type': 'joker_single',
            'size': 1,
            'rank': None,
            'ranks': [],
            'jokers': 1,
            'strength': 15,
        }

    # 階段: あるスートに対し、連続ランク列をジョーカーで補完して成立するか
    def _is_straight_and_ranks():
        """階段判定とランク列生成。

        方針:
          - まず非ジョーカーの最小ランク以上を始点とする階段を優先的に探索する。
            例: {8,9,JOKER} は 7-8-9 ではなく 8-9-10 を優先。
          - それで成立しない場合のみ、従来通り全ての始点を探索する（A-2-Joker などの特殊形維持）。
        """
        if n < 3:
            return None
        # スート候補: 非ジョーカーのスート集合（非ジョーカーが無いときは全スート）
        if non_jokers:
            suit_candidates = set(c[1] for c in non_jokers)
        else:
            suit_candidates = set(['\u2660','\u2665','\u2666','\u2663'])
        # 非ジョーカーの指定スートのランクリスト
        for suit in suit_candidates:
            hand_ranks = [c[2] for c in non_jokers if c[1] == suit and c[2] is not None]
            # 探索順を決定: 非ジョーカーの最小ランク以上 → それ以外
            if hand_ranks:
                min_non = min(hand_ranks)
                starts_pref = list(range(min_non, 14)) + list(range(1, min_non))
            else:
                starts_pref = list(range(1, 14))
            for start in starts_pref:
                expected = [((start + i - 1) % 13) + 1 for i in range(n)]
                # 2が含まれる場合は2が末尾でなければ不可（K,A,2 のみ許容）
                if 2 in expected and expected[-1] != 2:
                    continue
                temp = hand_ranks[:]
                jokers_left = len(jokers)
                match = 0
                for val in expected:
                    if val in temp:
                        temp.remove(val)
                        match += 1
                    else:
                        if jokers_left > 0:
                            jokers_left -= 1
                            match += 1
                        else:
                            match = -1
                            break
                if match == n:
                    return expected
        return None

    straight_ranks = _is_straight_and_ranks()
    if straight_ranks is not None:
        strength_ref = min(straight_ranks) if revo_flag else max(straight_ranks)
        return {
            'type': 'straight',
            'size': n,
            'rank': None,
            'ranks': straight_ranks,
            'jokers': len(jokers),
            'strength': strength_ref,
        }

    # 同ランク(ジョーカー含む)
    if len(non_jokers) == 0:
        # 全ジョーカー
        t = {1:'single',2:'pair',3:'triple'}.get(n, 'four')
        return {
            'type': t,
            'size': n,
            'rank': None,
            'ranks': [],
            'jokers': len(jokers),
            'strength': 15,
        }
    base_rank = non_jokers[0][2]
    same_rank = all((c[2] == base_rank) for c in non_jokers)
    if same_rank:
        t = {1:'single',2:'pair',3:'triple'}.get(n, 'four')
        if non_jokers:
            strengths = [_strength_from_rank(c[2]) for c in non_jokers]
            strength_ref = min(strengths) if revo_flag else max(strengths)
        else:
            strength_ref = 15
        return {
            'type': t,
            'size': n,
            'rank': base_rank,
            'ranks': [],
            'jokers': len(jokers),
            'strength': strength_ref,
        }

    return None


class RuleChecker:
    def __init__(self):
        self.revolution = False  # 革命フラグ
        # 追加ルール: 既存の階段より強い階段を出す際、
        #   1) ランク集合が一切重ならない (完全に上位の新しいブロック)
        #   2) その最小ランク(※Aは14として扱う) が 前の最大ランク より大きい
        # を要求する。これにより 9-10-J の後に J-Q-K や 10-J-Q は不可、最初に出せるのは Q-K-A。
        # デフォルト有効。従来挙動に戻したい場合 False に設定。
        self.strict_straight_progression = True
        # 革命イベントログ: トグル発生時のみ追加
        # 例: {'turn': 5, 'player': 2, 'reason': 'four_kind', 'cards': [...], 'old_state': False,
        #       'new_state': True, 'size': 4, 'meta': {'straight_len': None, 'rank': 7, 'jokers':1}}
        self.revolution_events = []
        # 革命発生条件の閾値設定（デフォルトは従来寄りの緩め）
        # - 4枚同ランク（ジョーカー混在可、4枚ちょうどでなく4枚以上）
        # - 階段は5枚以上（ジョーカー混在可）
        self.rev_enable_four_kind = True
        self.rev_four_kind_allow_joker = True
        self.rev_four_kind_exact = False
        self.rev_enable_straight = True
        self.rev_straight_min_len = 5
        self.rev_straight_allow_joker = True
        # 同一手番・同一カード集合での重複トグル防止用（idempotent 保護）
        # キー: (turn_count, player_id, tuple(sorted(str(c) for c in cards)))
        self._rev_seen_actions = set()
    
    # === 役分類 / 比較ユーティリティ =====================================
    def classify_combo(self, cards):
        """カード集合を役情報へ分類（LRUキャッシュ活用）。無効なら None。
        戻り値は従来フォーマットで 'raw_cards' を追加する。
        """
        if not cards:
            return None
        # 正規化キーを生成（順序非依存 + Jokerは (1,None,None)）
        try:
            key_elems = []
            for c in cards:
                if getattr(c, 'is_joker', False):
                    key_elems.append((1, None, None))
                else:
                    key_elems.append((0, getattr(c, 'suit', None), getattr(c, 'rank', None)))
            key_elems.sort()
            res = _classify_combo_cached(bool(self.revolution), tuple(key_elems))
            if res is None:
                return None
            out = dict(res)
            out['raw_cards'] = cards
            return out
        except Exception:
            # 予期せぬ型でも従来ロジックへフォールバック
            n = len(cards)
            jokers = [c for c in cards if getattr(c, 'is_joker', False)]
            non_jokers = [c for c in cards if not getattr(c, 'is_joker', False)]
            if n == 1 and jokers:
                jk = jokers[0]
                try:
                    jk.joker_as_rank = None
                    jk.joker_as_suit = None
                except Exception:
                    pass
                return {
                    'type': 'joker_single',
                    'size': 1,
                    'rank': None,
                    'ranks': [],
                    'jokers': 1,
                    'strength': 15,
                    'raw_cards': cards,
                }
            if self.is_straight(cards):
                straight_ranks = self.get_straight_ranks(cards)
                if not straight_ranks:
                    return None
                strength_ref = min(straight_ranks) if self.revolution else max(straight_ranks)
                return {
                    'type': 'straight',
                    'size': n,
                    'rank': None,
                    'ranks': straight_ranks,
                    'jokers': len(jokers),
                    'strength': strength_ref,
                    'raw_cards': cards,
                }
            if self.is_same_rank_or_joker(cards):
                base_rank = non_jokers[0].rank if non_jokers else None
                t = {1: 'single', 2: 'pair', 3: 'triple'}.get(n, 'four')
                if non_jokers:
                    strengths = [c.strength() for c in non_jokers]
                    strength_ref = min(strengths) if self.revolution else max(strengths)
                else:
                    strength_ref = 15
                return {
                    'type': t,
                    'size': n,
                    'rank': base_rank,
                    'ranks': [],
                    'jokers': len(jokers),
                    'strength': strength_ref,
                    'raw_cards': cards,
                }
            return None

    def compare_combos(self, challenger, field_combo):
        """challenger が field_combo を上回れるか。"""
        if challenger is None:
            return False
        if field_combo is None:  # 場が空
            return True
        # Joker 単体同士 -> 後出し不可(引き分け)
        if field_combo['type'] == 'joker_single':
            return challenger['type'] == 'joker_single'  # 同種なら許容(流し目的)
        if challenger['type'] != field_combo['type']:
            return False
        if challenger['size'] != field_combo['size']:
            return False
        # 階段の特別ルール (strict progression)
        if challenger['type'] == 'straight' and self.strict_straight_progression:
            old_ranks = field_combo.get('ranks', [])
            new_ranks = challenger.get('ranks', [])
            if not old_ranks or not new_ranks:
                return False
            # 1(A) は比較のため 14 に持ち上げ (循環を切って一方向比較)
            def norm(r):
                return 14 if r == 1 else r
            # 1) ランク集合が重なったら不可
            if set(old_ranks) & set(new_ranks):
                return False
            # 2) 新階段の“全ての”ランクが旧階段最大ランクより上になることを要求
            old_max = max(norm(r) for r in old_ranks)
            new_min = min(norm(r) for r in new_ranks)
            if new_min <= old_max:
                return False
            return True
        # それ以外 (従来通り) : strength 比較
        return self._compare_strength_value(challenger['strength'], field_combo['strength'])

    def _compare_strength_value(self, a, b):
        if self.revolution:
            return a < b
        return a > b

    def is_valid(self, current_field, cards):
        if cards is None or len(cards) == 0:
            return True
        return self.is_valid_move(cards, current_field)

    def is_valid_move(self, cards, current_field):
        # 新実装: classify & compare
        # 場が空
        if not current_field:
            return True
        challenger = self.classify_combo(cards)
        field_combo = self.classify_combo(current_field)
        return self.compare_combos(challenger, field_combo)

    def compare_strength(self, a, b):
        """
        革命フラグに応じた強さ比較。ただしジョーカーは常に最強
        a, bはstrength値
        """
        JOKER_STRENGTH = 15
        if a == JOKER_STRENGTH and b != JOKER_STRENGTH:
            return True
        if b == JOKER_STRENGTH and a != JOKER_STRENGTH:
            return False
        return a < b if self.revolution else a > b

    def is_same_rank_or_joker(self, cards):
        non_jokers = [card for card in cards if not card.is_joker]
        if not non_jokers:
            # 全部ジョーカーの場合は仮想ランク・スートをNoneにリセット
            for card in cards:
                if card.is_joker:
                    card.joker_as_rank = None
                    card.joker_as_suit = None
            return True
        rank = non_jokers[0].rank
        suit = non_jokers[0].suit
        valid = all(card.rank == rank or card.is_joker for card in cards)
        if valid:
            # ジョーカーが含まれる場合は仮想ランク・スートをセット
            for card in cards:
                if card.is_joker:
                    card.joker_as_rank = rank
                    card.joker_as_suit = suit
            return True
        else:
            # 無効な場合は代用情報をクリアして副作用を残さない
            for card in cards:
                if card.is_joker:
                    card.joker_as_rank = None
                    card.joker_as_suit = None
            return False

    def is_8cut(self, cards):
        """8が含まれていて、かつジョーカーだけではないとき、8切り発動"""
        has_8 = any(card.rank == 8 and not card.is_joker for card in cards)
        has_normal = any(not card.is_joker for card in cards)
        return has_8 and has_normal

    def is_straight(self, cards):
        """
        同じスートで連続したランクか判定（ジョーカーで間を埋めることも許可）
        例: 4,ジョーカー,6 や Q,ジョーカー,A など
        ただし2が末尾以外に来る階段（A,2,3や2,3,4等）は不可。K,A,2のみOK。
        ジョーカー補完後も厳密に判定。
        """
        if len(cards) < 3:
            return False
        jokers = [card for card in cards if card.is_joker]
        non_jokers = [card for card in cards if not card.is_joker]
        n = len(cards)
        suit_candidates = set([card.suit for card in non_jokers]) if non_jokers else set(['♠', '♥', '♦', '♣'])
        for suit in suit_candidates:
            # 手札のランクリスト
            hand_ranks = [card.rank for card in non_jokers if card.suit == suit]
            # 探索順: 非ジョーカーの最小ランク以上を優先
            if hand_ranks:
                min_non = min(hand_ranks)
                starts_pref = list(range(min_non, 14)) + list(range(1, min_non))
            else:
                starts_pref = list(range(1, 14))
            for start in starts_pref:
                expected = [(start + i - 1) % 13 + 1 for i in range(n)]
                # 2が含まれる場合は2が末尾でなければ不可
                if 2 in expected and expected[-1] != 2:
                    continue
                temp_ranks = hand_ranks[:]
                used_jokers = []
                jokers_left = jokers[:]
                match = 0
                for idx, val in enumerate(expected):
                    if val in temp_ranks:
                        temp_ranks.remove(val)
                        match += 1
                    else:
                        if jokers_left:
                            # ジョーカーをこのランク・スートに割り当て
                            joker = jokers_left.pop(0)
                            joker.joker_as_rank = val
                            joker.joker_as_suit = suit
                            used_jokers.append(joker)
                        else:
                            break
                if match + len(used_jokers) == n:
                    # 2が末尾以外に来る場合は不可
                    if 2 in expected and expected[-1] != 2:
                        continue
                    return True
        # 失敗時はリセット
        for joker in jokers:
            joker.joker_as_rank = None
            joker.joker_as_suit = None
        return False

    def get_straight_ranks(self, cards):
        """
        ジョーカーを補完した階段のランク列を返す（昇順）
        例: 4,ジョーカー,6 → [4,5,6]
        Q,ジョーカー,A → [12,13,1]
        ただし2が末尾以外に来る階段（A,2,3や2,3,4等）は不可。K,A,2のみOK。
        ジョーカー補完後も厳密に判定。
        """
        jokers = [c for c in cards if c.is_joker]
        non_jokers = [c for c in cards if not c.is_joker]
        n = len(cards)
        if not non_jokers:
            return []
        suit_candidates = set([c.suit for c in non_jokers]) if non_jokers else set(['♠', '♥', '♦', '♣'])
        for suit in suit_candidates:
            hand_ranks = [c.rank for c in non_jokers if c.suit == suit]
            # 探索順: 非ジョーカーの最小ランク以上を優先
            if hand_ranks:
                min_non = min(hand_ranks)
                starts_pref = list(range(min_non, 14)) + list(range(1, min_non))
            else:
                starts_pref = list(range(1, 14))
            for start in starts_pref:
                expected = [(start + i - 1) % 13 + 1 for i in range(n)]
                if 2 in expected and expected[-1] != 2:
                    continue
                temp_ranks = hand_ranks[:]
                used_jokers = []
                jokers_left = jokers[:]
                match = 0
                for idx, val in enumerate(expected):
                    if val in temp_ranks:
                        temp_ranks.remove(val)
                        match += 1
                    else:
                        if jokers_left:
                            joker = jokers_left.pop(0)
                            joker.joker_as_rank = val
                            joker.joker_as_suit = suit
                            used_jokers.append(joker)
                        else:
                            break
                if match + len(used_jokers) == n:
                    if 2 in expected and expected[-1] != 2:
                        continue
                    return expected
        for joker in jokers:
            joker.joker_as_rank = None
            joker.joker_as_suit = None
        return []

    def check_revolution(self, cards, player_id=None, turn_count=None):
        """
        革命トグル判定 (現在の仕様):
          - 条件: (1) 四枚同ランク(ジョーカー補完可, ちょうど4枚) または (2) 5枚以上の階段(ジョーカー可)
          - 条件成立ごとに revolution フラグを反転 (OFF→ON / ON→OFF)
          - 同一手番・同一カード集合での多重トグル防止 (idempotent)
        戻り値: トグルが発生した場合 True, それ以外 False
        """
        if not cards:
            return False
        # 重複発火保護キー
        action_key = None
        try:
            if turn_count is not None and player_id is not None:
                canonical_cards = tuple(sorted(str(c) for c in cards))
                action_key = (int(turn_count), int(player_id), canonical_cards)
                if action_key in self._rev_seen_actions:
                    return False
        except Exception:
            action_key = None

        non_jokers = [c for c in cards if not c.is_joker]
        jokers = [c for c in cards if c.is_joker]

        # 条件1: 四枚同ランク (ジョーカーで補完してよい) - ちょうど4枚
        four_kind_ok = False
        if len(cards) == 4:
            if len(non_jokers) > 0:
                base_rank = non_jokers[0].rank
                if all(c.rank == base_rank or c.is_joker for c in cards):
                    four_kind_ok = True

        # 条件2: 5枚以上の階段 (ジョーカー使用可) - is_straight 判定利用
        straight_ok = False
        if len(cards) >= 5:
            if self.is_straight(cards):
                straight_ok = True

        if not (four_kind_ok or straight_ok):
            return False

        # トグル
        old_state = self.revolution
        self.revolution = not self.revolution
        pure = (len(jokers) == 0)
        event = {
            'turn': turn_count,
            'player': player_id,
            'reason': 'four_kind' if four_kind_ok else 'long_straight',
            'cards': [str(c) for c in cards],
            'old_state': old_state,
            'new_state': self.revolution,
            'size': len(cards),
            'meta': {
                'straight_len': len(cards) if straight_ok else None,
                'rank': non_jokers[0].rank if four_kind_ok else None,
                'jokers': len(jokers),
                'pure': pure,
                'toggled': True,
            },
        }
        self.revolution_events.append(event)
        if action_key is not None:
            self._rev_seen_actions.add(action_key)
        return True

    def reset_revolution(self):
        """
        革命状態をリセット（場流し時など）
        """
        self.revolution = False
        # 場流しなどで状態をリセットするが、イベント履歴は残す（完全リセットしたいときは clear_revolution_events を呼ぶ）

    def clear_revolution_events(self):
        """革命イベント履歴をクリア（新ゲーム開始時など）。"""
        self.revolution_events.clear()
        # 重複トグル保護のキーもクリア
        self._rev_seen_actions.clear()

    def get_revolution_events(self):
        """革命イベント履歴を返す（参照用）。"""
        return list(self.revolution_events)

   
    def exchange_cards_by_rankings(self, players, rankings):
        """
        大富豪ルールの順位に応じたカード交換を行う。
        players: プレイヤーオブジェクトのリスト
        rankings: [1位, 2位, ..., n位]のplayer_idリスト（0-indexed, 1位=大富豪, 最下位=大貧民）
        デバッグ用に誰がどのカードをもらったかをprint出力
        """
        n = len(rankings)
        if n < 4:
            return  # 順位が確定していない場合は何もしない

        daifugo = rankings[0]
        fugo = rankings[1]
        hinmin = rankings[-2]
        dai_hinmin = rankings[-1]

        # --- 大貧民→大富豪（2枚） ---　自分の最も強いカードを2枚渡す。
        dai_hinmin_hand = sorted(players[dai_hinmin].hand, key=lambda c: c.strength(), reverse=True)
        dai_hinmin_give = dai_hinmin_hand[:2]
        for card in dai_hinmin_give:
            players[dai_hinmin].hand.remove(card)
        players[daifugo].hand.extend(dai_hinmin_give)
        #print(f"大貧民(Player {dai_hinmin})→大富豪(Player {daifugo}): {[str(c) for c in dai_hinmin_give]}")

        # --- 大富豪→大貧民（2枚） ---　自分の最も弱いカードを2枚渡す。
        daifugo_hand = sorted(players[daifugo].hand, key=lambda c: c.strength())
        daifugo_give = daifugo_hand[:2]
        for card in daifugo_give:
            players[daifugo].hand.remove(card)
        players[dai_hinmin].hand.extend(daifugo_give)
        #print(f"大富豪(Player {daifugo})→大貧民(Player {dai_hinmin}): {[str(c) for c in daifugo_give]}")

        # --- 貧民→富豪（1枚） ---　自分の最も強いカードを1枚渡す。
        hinmin_hand = sorted(players[hinmin].hand, key=lambda c: c.strength(), reverse=True)
        hinmin_give = hinmin_hand[0]
        players[hinmin].hand.remove(hinmin_give)
        players[fugo].hand.append(hinmin_give)
        #print(f"貧民(Player {hinmin})→富豪(Player {fugo}): {hinmin_give}")

        # --- 富豪→貧民（1枚） ---　自分の最も弱いカードを1枚渡す。
        fugo_hand = sorted(players[fugo].hand, key=lambda c: c.strength())
        fugo_give = fugo_hand[0]
        players[fugo].hand.remove(fugo_give)
        players[hinmin].hand.append(fugo_give)
        #print(f"富豪(Player {fugo})→貧民(Player {hinmin}): {fugo_give}")
