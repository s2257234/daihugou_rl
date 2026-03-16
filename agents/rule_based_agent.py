class RuleBasedAgent:
    def __init__(self, player_id=None):
        self.player_id = player_id
        # AlphaZeroAgentとの互換性のためのダミー属性
        self.total_positive = 0
        self.total_value_samples = 0

    def select_action(self, observation, legal_actions=None, training=False, **kwargs):
        # training引数は互換性のため受け取るが使用しない（ルールベースは学習しない）
        if legal_actions is None:
            return None  # legal_actionsがなければパス
        if not legal_actions:
            return None  # パス

        # 最もランクの低いカードセットを選ぶ（ジョーカーは避ける）
        def card_set_score(card_set):
            if card_set is None:
                return float('inf')  # パスは最も避ける
            # ジョーカーが含まれていれば高スコア（避ける）
            if any(c.is_joker for c in card_set):
                return 100
            # 階段で出されたら高スコア
            # self.rule_checkerがない場合は階段判定をスキップ
            return card_set[0].rank  # 同ランクなので先頭だけでOK

        return min(legal_actions, key=card_set_score)

    def finalize_phase(self, winner_player_id=None, was_active=None, **kwargs):
        """互換性のためのダミーメソッド（ルールベースは学習しないため何もしない）"""
        pass

    def flush_unfinished_phase(self, **kwargs):
        """互換性のためのダミーメソッド（ルールベースは学習しないため何もしない）"""
        pass

    def finalize_game(self, **kwargs):
        """互換性のためのダミーメソッド（ルールベースは学習しないため何もしない）"""
        pass
