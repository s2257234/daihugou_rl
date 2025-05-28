# game/rules.py

class RuleChecker:
    def is_valid(self, current_field, card):
        # 現在の場の最後のカードを取得
        current_card = current_field[-1] if current_field else None
        return self.is_valid_move(card, current_card)

    def is_valid_move(self, card, current):
        # 場にカードがない（リセットされている）場合は何でも出せる
        if not current:
            return True

        # 出そうとしているカードがジョーカーなら必ず出せる
        if card.is_joker:
            return True

        # 現在の場のカードがジョーカーなら、出せない
        if current.is_joker:
            return False

        # 通常は、場のカードのランクより上でないと出せない
        return card.rank > current.rank
