class RuleChecker:
    def __init__(self):
        self.revolution = False  # 革命フラグ

    def is_valid(self, current_field, cards):
        if cards is None or len(cards) == 0:
            return True
        return self.is_valid_move(cards, current_field)

    def is_valid_move(self, cards, current_field):
        play_count = len(cards)
        strengths = [card.strength() for card in cards if not card.is_joker]

        # 階段判定
        is_straight = self.is_straight(cards)

        # 場が空ならOK
        if not current_field:
            return True

        field_count = len(current_field)
        # ★場が空でなければ、出す枚数と場の枚数が一致しない場合はFalse
        if play_count != field_count:
            return False

        field_strengths = [card.strength() for card in current_field if not card.is_joker]
        field_is_straight = self.is_straight(current_field)

        # ジョーカー単独出し特別ルール
        if play_count == 1 and cards[0].is_joker:
            # ジョーカーの仮想ランク・スートをリセット
            cards[0].joker_as_rank = None
            cards[0].joker_as_suit = None
            # 場がジョーカー単独なら、次もジョーカー単独でしか出せない
            if field_count == 1 and current_field[0].is_joker:
                return True
            # 革命時は3より強い、通常時は2より強い
            if self.revolution:
                # 3の強さは1
                return (field_count == 1 and (not current_field[0].is_joker) and current_field[0].strength() < 2)
            else:
                # 2の強さは14
                return (field_count == 1 and (not current_field[0].is_joker) and current_field[0].strength() < 15)

        # 階段同士の比較
        if is_straight and field_is_straight:
            if play_count != field_count:
                return False
            if cards[0].suit != current_field[0].suit:
                return False
            # ジョーカーを補完した最大/最小ランクで比較
            play_ranks = self.get_straight_ranks(cards)
            field_ranks = self.get_straight_ranks(current_field)
            if not play_ranks or not field_ranks:
                return False
            if self.revolution:
                return self.compare_strength(min(play_ranks), min(field_ranks))
            else:
                return self.compare_strength(max(play_ranks), max(field_ranks))

        # 階段出し→通常出し、またはその逆は禁止
        if is_straight != field_is_straight:
            return False

        # 通常出し（同ランクのカード or ジョーカー）
        if not self.is_same_rank_or_joker(cards):
            return False

        if play_count != field_count:
            return False

        # 強さ比較
        if self.revolution:
            field_strength = min(field_strengths) if field_strengths else -1
            play_strength = min(strengths) if strengths else 0
        else:
            field_strength = field_strengths[0] if field_strengths else -1
            play_strength = max(strengths) if strengths else 0
        return self.compare_strength(play_strength, field_strength)

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
        # ジョーカーが含まれる場合は仮想ランク・スートをセット
        for card in cards:
            if card.is_joker:
                card.joker_as_rank = rank
                card.joker_as_suit = suit
        return valid

    def is_8cut(self, cards):
        """8が含まれていて、かつジョーカーだけではないとき、8切り発動"""
        has_8 = any(card.rank == 8 and not card.is_joker for card in cards)
        has_normal = any(not card.is_joker for card in cards)
        return has_8 and has_normal

    def is_straight(self, cards):
        """
        同じスートで連続したランクか判定（ジョーカーで間を埋めることも許可）
        例: 4,ジョーカー,6 や Q,ジョーカー,A など
        """
        if len(cards) < 3:
            return False
        jokers = [card for card in cards if card.is_joker]
        non_jokers = [card for card in cards if not card.is_joker]
        ranks = sorted([card.rank for card in non_jokers])
        num_jokers = len(jokers)
        n = len(cards)
        # スート候補を全て試す（non_jokersが空なら全スートを候補に）
        suit_candidates = set([card.suit for card in non_jokers]) if non_jokers else set(['♠', '♥', '♦', '♣'])
        for suit in suit_candidates:
            # このスートで階段が作れるか
            # ランクをループ（A,2,3...K,A,2...）として扱う
            for start in range(1, 14):
                expected = []
                for i in range(n):
                    val = (start + i - 1) % 13 + 1
                    expected.append(val)
                # 2が含まれる場合は2が末尾でなければ不可
                if 2 in expected and expected[-1] != 2:
                    continue
                temp_ranks = ranks[:]
                match = 0
                used_jokers = []
                for idx, val in enumerate(expected):
                    if idx < len(cards) and not cards[idx].is_joker and cards[idx].suit != suit:
                        break  # スートが違うカードが混じっている
                    if val in temp_ranks:
                        temp_ranks.remove(val)
                        match += 1
                    else:
                        used_jokers.append((idx, val))
                # 残りはジョーカーで埋められるか
                if n - match <= num_jokers:
                    # ジョーカーの仮想ランク・スートをセット
                    for j, (idx, val) in enumerate(used_jokers):
                        if j < len(jokers):
                            jokers[j].joker_as_rank = val
                            jokers[j].joker_as_suit = suit
                    return True
        # 失敗時はリセット
        for joker in jokers:
            joker.joker_as_rank = None
            joker.joker_as_suit = None
        return False

    def check_revolution(self, cards):
        """
        現在は4枚のペア出しのみで革命発生
        革命発生条件を判定し、該当すればself.revolutionをTrueにする。
        例: 同じランク4枚以上（ジョーカー含む場合は調整可）が出されたとき。
        """
        non_jokers = [c for c in cards if not c.is_joker]
        if len(cards) >= 4:
            # 4枚以上、かつ全て同じランク（ジョーカーは無視）
            if len(non_jokers) > 0:
                rank = non_jokers[0].rank
                if all(c.rank == rank or c.is_joker for c in cards):
                    self.revolution = not self.revolution  # 革命状態をトグル
                    return True
        return False

    def reset_revolution(self):
        """
        革命状態をリセット（場流し時など）
        """
        self.revolution = False

    def get_straight_ranks(self, cards):
        """
        ジョーカーを補完した階段のランク列を返す（昇順）
        例: 4,ジョーカー,6 → [4,5,6]
        Q,ジョーカー,A → [12,13,1]
        """
        jokers = [c for c in cards if c.is_joker]
        non_jokers = [c for c in cards if not c.is_joker]
        n = len(cards)
        if not non_jokers:
            return []
        ranks = sorted([c.rank for c in non_jokers])
        num_jokers = len(jokers)
        suit_candidates = set([c.suit for c in non_jokers]) if non_jokers else set(['♠', '♥', '♦', '♣'])
        for suit in suit_candidates:
            for start in range(1, 14):
                expected = []
                for i in range(n):
                    val = (start + i - 1) % 13 + 1
                    expected.append(val)
                # 2が含まれる場合は2が末尾でなければ不可
                if 2 in expected and expected[-1] != 2:
                    continue
                temp_ranks = ranks[:]
                match = 0
                used_jokers = []
                for idx, val in enumerate(expected):
                    if idx < len(cards) and not cards[idx].is_joker and cards[idx].suit != suit:
                        break
                    if val in temp_ranks:
                        temp_ranks.remove(val)
                        match += 1
                    else:
                        used_jokers.append((idx, val))
                if n - match <= num_jokers:
                    for j, (idx, val) in enumerate(used_jokers):
                        if j < len(jokers):
                            jokers[j].joker_as_rank = val
                            jokers[j].joker_as_suit = suit
                    return expected
        for joker in jokers:
            joker.joker_as_rank = None
            joker.joker_as_suit = None
        return ranks
