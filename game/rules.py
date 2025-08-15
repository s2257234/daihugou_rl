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

        # --- 2のペア＋ジョーカーの特殊判定 ---
        # 場が2,ジョーカー(2)のペアなら、何も上書きできない
        if (
            play_count == 2 and field_count == 2
            and self.is_same_rank_or_joker(current_field)
            and any(card.is_joker for card in current_field)
            and all((card.rank == 2 or card.is_joker) for card in current_field)
        ):
            # 2,ジョーカー(2)のペアが場に出ている場合は、どんなペアも不可
            return False

        # 強さ比較
        if self.revolution:
            field_strength = min(field_strengths) if field_strengths else -1
            play_strength = min(strengths) if strengths else 0
        else:
            field_strength = max(field_strengths) if field_strengths else -1
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
            for start in range(1, 14):
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
            for start in range(1, 14):
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

    def check_revolution(self, cards):
        """
        革命発生条件を判定し、該当すればself.revolutionをTrueにする。
        例: 同じランク4枚以上（ジョーカー含む場合は調整可）、または5枚以上の階段
        """
        non_jokers = [c for c in cards if not c.is_joker]
        # 4枚以上、かつ全て同じランク（ジョーカーは無視）
        if len(cards) >= 4:
            if len(non_jokers) > 0:
                rank = non_jokers[0].rank
                if all(c.rank == rank or c.is_joker for c in cards):
                    self.revolution = not self.revolution  # 革命状態をトグル
                    return True
        # 5枚以上の階段
        if len(cards) >= 5 and self.is_straight(cards):
            self.revolution = not self.revolution
            return True
        return False

    def reset_revolution(self):
        """
        革命状態をリセット（場流し時など）
        """
        self.revolution = False

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
        print(f"大貧民(Player {dai_hinmin})→大富豪(Player {daifugo}): {[str(c) for c in dai_hinmin_give]}")

        # --- 大富豪→大貧民（2枚） ---　自分の最も弱いカードを2枚渡す。
        daifugo_hand = sorted(players[daifugo].hand, key=lambda c: c.strength())
        daifugo_give = daifugo_hand[:2]
        for card in daifugo_give:
            players[daifugo].hand.remove(card)
        players[dai_hinmin].hand.extend(daifugo_give)
        print(f"大富豪(Player {daifugo})→大貧民(Player {dai_hinmin}): {[str(c) for c in daifugo_give]}")

        # --- 貧民→富豪（1枚） ---　自分の最も強いカードを1枚渡す。
        hinmin_hand = sorted(players[hinmin].hand, key=lambda c: c.strength(), reverse=True)
        hinmin_give = hinmin_hand[0]
        players[hinmin].hand.remove(hinmin_give)
        players[fugo].hand.append(hinmin_give)
        print(f"貧民(Player {hinmin})→富豪(Player {fugo}): {hinmin_give}")

        # --- 富豪→貧民（1枚） ---　自分の最も弱いカードを1枚渡す。
        fugo_hand = sorted(players[fugo].hand, key=lambda c: c.strength())
        fugo_give = fugo_hand[0]
        players[fugo].hand.remove(fugo_give)
        players[hinmin].hand.append(fugo_give)
        print(f"富豪(Player {fugo})→貧民(Player {hinmin}): {fugo_give}")
