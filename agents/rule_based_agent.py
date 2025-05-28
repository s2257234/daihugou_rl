# agents/rule_based_agent.py

class RuleBasedAgent:
    def select_action(self, observation, legal_actions):
        if not legal_actions:
            return None  # パス

        # 最もランクの低いカードを出す（温存志向）
        return min(legal_actions, key=lambda c: c.rank if not c.is_joker else 100)
