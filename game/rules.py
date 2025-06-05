class RuleChecker:
    def __init__(self):
        self.revolution = False  # 革命があるなら必要

    def is_valid(self, current_field, cards):
        if cards is None or len(cards) == 0:
            return True
        return self.is_valid_move(cards, current_field)

    def is_valid_move(self, cards, current_field):
        play_count = len(cards)
        ranks = [card.rank for card in cards if not card.is_joker]

        if len(set(ranks)) > 1:
            return False

        if not current_field:
            return True

        field_count = len(current_field)
        field_ranks = [card.rank for card in current_field if not card.is_joker]

        if any(card.is_joker for card in cards):
            if play_count < field_count:
                return False
            if not any(not c.is_joker for c in cards):
                return False

        if any(card.is_joker for card in current_field):
            return False

        if play_count != field_count:
            return False

        if len(set(ranks)) > 1:
            return False

        field_rank = field_ranks[0] if field_ranks else -1
        max_rank = max(ranks) if ranks else 0
        return max_rank > field_rank

    def is_8cut(self, cards):
        """8が含まれていて、かつジョーカーだけではないとき、8切り発動"""
        has_8 = any(card.rank == 8 and not card.is_joker for card in cards)
        has_normal = any(not card.is_joker for card in cards)
        return has_8 and has_normal
    
