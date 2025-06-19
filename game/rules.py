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
        field_strengths = [card.strength() for card in current_field if not card.is_joker]
        field_is_straight = self.is_straight(current_field)

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
        non_jokers = [c for c in cards if not c.is_joker]
        if not non_jokers:
            return True
        rank = non_jokers[0].rank
        return all(c.rank == rank or c.is_joker for c in cards)

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
        suits = [card.suit for card in cards if not card.is_joker]
        if len(set(suits)) != 1:
            return False
        jokers = [c for c in cards if c.is_joker]
        non_jokers = [c for c in cards if not c.is_joker]
        ranks = sorted([c.rank for c in non_jokers])
        num_jokers = len(jokers)
        n = len(cards)
        # ランクをループ（A,2,3...K,A,2...）として扱う
        # 13→1→2のような連続も許可
        # まず全ての開始点を試す
        for start in range(1, 14):
            expected = []
            for i in range(n):
                val = (start + i - 1) % 13 + 1
                expected.append(val)
            # expectedの中でnon_jokersのrankが何個一致するか
            temp_ranks = ranks[:]
            match = 0
            for val in expected:
                if val in temp_ranks:
                    temp_ranks.remove(val)
                    match += 1
            # 残りはジョーカーで埋められるか
            if n - match <= num_jokers:
                return True
        return False

    def check_revolution(self, cards):
        """
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
        # 全ての開始点を試す
        for start in range(1, 14):
            expected = []
            for i in range(n):
                val = (start + i - 1) % 13 + 1
                expected.append(val)
            temp_ranks = ranks[:]
            match = 0
            for val in expected:
                if val in temp_ranks:
                    temp_ranks.remove(val)
                    match += 1
            if n - match <= num_jokers:
                return expected
        return ranks
