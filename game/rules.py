class RuleChecker:
    def is_valid(self, current_field, cards):
        # カードが空なら（パス）、常に有効
        if cards is None or len(cards) == 0:
            return True

        return self.is_valid_move(cards, current_field)

    def is_valid_move(self, cards, current_field):
        # 出すカードの枚数
        play_count = len(cards)

        # 同じランクのカードでなければ無効（ジョーカーを除いて）
        ranks = [card.rank for card in cards if not card.is_joker]
        if len(set(ranks)) > 1:
            return False

        # 場が空なら出せる（ジョーカー含んでも可）
        if not current_field:
            return True

        # 場のカード情報
        field_count = len(current_field)
        field_ranks = [card.rank for card in current_field if not card.is_joker]

        # ジョーカー単体では出せない（場が複数枚ある場合）
        if any(card.is_joker for card in cards):
            if play_count < field_count:
                return False  # 出すカード枚数が足りない
            if not any(not c.is_joker for c in cards):
                return False  # 全部ジョーカーは無効

        # 場のカードにジョーカー含まれていたら通常のカードでは出せない
        if any(card.is_joker for card in current_field):
            return False

        # 出すカードと場のカードの枚数が一致しないと出せない
        if play_count != field_count:
            return False

        # 出すカードが同じランクでなければ無効（ジョーカー除いて）
        if len(set(ranks)) > 1:
            return False

        # ランクが場より上なら出せる（ジョーカー含む場合は最大のランクで比較）
        field_rank = field_ranks[0] if field_ranks else -1
        max_rank = max(ranks) if ranks else 0
        return max_rank > field_rank
