class RuleBasedAgent:
    def select_action(self, observation, legal_actions):
        if not legal_actions:
            return None  # パス

        # 最もランクの低いカードセットを選ぶ（ジョーカーは避ける）
        def card_set_score(card_set):
            # ジョーカーが含まれていれば高スコア（避ける）
            if any(c.is_joker for c in card_set):
                return 100
            # 階段で出されたら高スコア
            if self.rule_checker.is_straight(card_set):
                return 100
            return card_set[0].rank  # 同ランクなので先頭だけでOK

        return min(legal_actions, key=card_set_score)
