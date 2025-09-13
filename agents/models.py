"""AlphaZero 用 Policy-Value ネットワーク (初期版)

目的:
  - 環境状態(簡易特徴) -> (policy_logits, value) を出力
  - policy_logits は固定長 (config 想定 max_policy_size) を出す。
    実際の合法手リスト legal_actions が N 個なら、呼び出し側 (drl_agent.py) で
    先頭 N 要素のみを利用して確率に正規化する運用を想定。

制約 / 今後の拡張ポイント:
  - 現在の state は dict 形式 {"hand_size", "field_size", "turn"} を想定 (簡易)。
  - 本番ではカード種別 / 残り枚数 / 連番 / 階段 etc. を多チャネル one-hot に拡張する。
  - マルチプレイヤー (4人大富豪) なので turn は one-hot (4次元) 埋め込み。
  - value: 現在は単一スカラー (root 視点)。多人数報酬を分離したい場合はベクトル出力に変更可能。

使用方法:
  model = PolicyValueNet(max_policy_size=128)
  logits, value = model.forward(state_dict)
  -> logits: Tensor(shape=[max_policy_size])
     value : Tensor(shape=[1])  (tanh で -1~1)

PyTorch 依存: torch が未インストールの場合は ImportError を送出する。
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch がインストールされていません。pip install torch で導入してください.") from e


class PolicyValueNet(nn.Module):
    def __init__(self,
                 max_policy_size: int = 128,
                 hidden_size: int = 128,
                 num_players: int = 4,
                 device: Optional[str] = None):
        super().__init__()
        self.max_policy_size = max_policy_size
        self.num_players = num_players
        self.device = torch.device(device) if device else torch.device("cpu")

        # 入力特徴量: hand_size(1) + field_size(1) + turn_onehot(num_players)
        input_dim = 2 + num_players
        h = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(h, max_policy_size)
        # value: 各プレイヤーが「次に上がる」確率 (multi-label) を 0~1 で出力 (Sigmoid)
        self.value_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, num_players),
            nn.Sigmoid(),  # shape: (num_players,)
        )
        self.to(self.device)

    # -----------------------------------------------------
    # 状態エンコード
    # -----------------------------------------------------
    def _encode_state(self, state: Dict[str, Any]):
        """辞書形式 state をテンソル特徴に変換。
        想定キー: hand_size(int), field_size(int), turn(int)
        未存在キーは 0 として扱う。
        """
        hand_size = float(state.get("hand_size", 0))
        field_size = float(state.get("field_size", 0))
        turn = int(state.get("turn", 0))
        turn_onehot = [0.0] * self.num_players
        if 0 <= turn < self.num_players:
            turn_onehot[turn] = 1.0
        feat = [hand_size, field_size] + turn_onehot
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        return x

    # -----------------------------------------------------
    # 推論
    # -----------------------------------------------------
    def forward(self, state: Dict[str, Any]):  # state は単一局面 (バッチ拡張は未対応)
        """(policy_logits, value_vec) を返す。

        policy_logits: Tensor (max_policy_size,)
        value_vec:    Tensor (num_players,) 各プレイヤーの『次に上がる』確率
        """
        x = self._encode_state(state)
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value_vec = self.value_head(h)
        return policy_logits, value_vec

    # -----------------------------------------------------
    # 保存 / 読込ユーティリティ
    # -----------------------------------------------------
    def save(self, path: str):
        ckpt = {
            "state_dict": self.state_dict(),
            "max_policy_size": self.max_policy_size,
            "hidden_size": self.policy_head.in_features,  # 便宜上
            "num_players": self.num_players,
        }
        torch.save(ckpt, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "PolicyValueNet":
        ckpt = torch.load(path, map_location=map_location or "cpu")
        model = PolicyValueNet(
            max_policy_size=ckpt.get("max_policy_size", 128),
            hidden_size=ckpt.get("hidden_size", 128),
            num_players=ckpt.get("num_players", 4),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


__all__ = ["PolicyValueNet"]

# =============================================================
# Stage2: 可変長アクション対応 Policy-Value ネットワーク
# =============================================================


class ActionPolicyValueNet(nn.Module):
    """可変長合法手に対して (state_emb, action_feat) から逐次ロジットを算出するモデル。

    特徴:
      - state 埋め込みは従来同様 hand_size / field_size / turn one-hot
      - action 特徴は以下:
          [is_pass, num_cards, is_pair, is_straight, has_joker,
           min_rank_norm, max_rank_norm, avg_rank_norm, span_norm]
        (必要に応じて拡張可能)
      - ロジット: joint_mlp(concat(state_emb, action_emb)) -> 1
      - value: state_emb から算出

    使用方法:
        model = ActionPolicyValueNet()
        logits, value = model.evaluate(state_dict, legal_actions)
    """

    def __init__(self,
                 state_hidden: int = 128,
                 action_hidden: int = 64,
                 joint_hidden: int = 128,
                 num_players: int = 4,
                 device: Optional[str] = None):
        super().__init__()
        self.num_players = num_players
        self.device = torch.device(device) if device else torch.device("cpu")
        self.supports_variable_actions = True  # 検出用フラグ

        state_in = 2 + num_players  # hand_size, field_size, turn_onehot
        self.state_backbone = nn.Sequential(
            nn.Linear(state_in, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(),
        )

        self.action_backbone = nn.Sequential(
            nn.Linear(9, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, action_hidden),
            nn.ReLU(),
        )

        self.joint = nn.Sequential(
            nn.Linear(state_hidden + action_hidden, joint_hidden),
            nn.ReLU(),
            nn.Linear(joint_hidden, 1),  # ロジット
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, 1),
            nn.Sigmoid(),  # probability 0~1
        )
        self.to(self.device)

        # ランク順 (昇順) 3..A 2 Joker (大富豪標準強さ: 3弱, 2強, Joker 最強想定)
        self._rank_order = ["3","4","5","6","7","8","9","10","J","Q","K","A","2","JOKER"]
        self._rank_index = {r:i for i,r in enumerate(self._rank_order)}
        self._max_rank_idx = len(self._rank_order)-1

    # ---------------- Public API ----------------
    def evaluate(self, state: Dict[str, Any], legal_actions: List[Any]):
        state_vec = self._encode_state(state)
        state_emb = self.state_backbone(state_vec)
        logits = []
        for act in legal_actions:
            feat_vec = self._encode_action(act)
            act_emb = self.action_backbone(feat_vec)
            joint = self.joint(torch.cat([state_emb, act_emb], dim=-1))
            logits.append(joint.squeeze(-1).item())
        value = self.value_head(state_emb).squeeze(-1).item()
        return logits, value

    # ---------------- Encoders ----------------
    def _encode_state(self, state: Dict[str, Any]):
        hand_size = float(state.get("hand_size", 0))
        field_size = float(state.get("field_size", 0))
        turn = int(state.get("turn", 0))
        turn_onehot = [0.0]*self.num_players
        if 0 <= turn < self.num_players:
            turn_onehot[turn] = 1.0
        feat = [hand_size, field_size] + turn_onehot
        return torch.tensor(feat, dtype=torch.float32, device=self.device)

    def _encode_action(self, action: Any):
        # pass
        if action in (None, "pass", "PASS"):
            return torch.tensor([1,0,0,0,0,0,0,0,0], dtype=torch.float32, device=self.device)
        cards = action
        # 正規化: list[str] 想定 / そうでなければ文字列化
        if not isinstance(cards, (list, tuple)):
            cards = [str(cards)]
        cards = [str(c) for c in cards]
        ranks_idx = []
        has_joker = 0
        for c in cards:
            r = self._extract_rank(c)
            if r == "JOKER":
                has_joker = 1
            idx = self._rank_index.get(r, None)
            if idx is not None:
                ranks_idx.append(idx)
        num_cards = len(cards)
        is_pair = 1 if num_cards==2 and self._all_same_rank(cards) else 0
        is_straight = 1 if (num_cards>=3 and self._is_straight(ranks_idx)) else 0
        if ranks_idx:
            mn = min(ranks_idx)
            mx = max(ranks_idx)
            avg = sum(ranks_idx)/len(ranks_idx)
            span = mx - mn
            mn_n = mn/self._max_rank_idx
            mx_n = mx/self._max_rank_idx
            avg_n = avg/self._max_rank_idx
            span_n = span/max(1,self._max_rank_idx)
        else:
            mn_n=mx_n=avg_n=span_n=0.0
        feat = [
            0,  # is_pass
            float(num_cards),
            float(is_pair),
            float(is_straight),
            float(has_joker),
            float(mn_n), float(mx_n), float(avg_n), float(span_n)
        ]
        return torch.tensor(feat, dtype=torch.float32, device=self.device)

    # ---------------- Helpers ----------------
    def _extract_rank(self, card_str: str) -> str:
        s = card_str.upper()
        if "JOKER" in s:
            return "JOKER"
        # remove suit symbols / letters
        suits = ['S','H','D','C','♠','♥','♦','♣']
        # keep digits and letters forming rank
        filtered = ''.join(ch for ch in s if ch not in suits)
        # map face cards
        mapping = {"11":"J","12":"Q","13":"K","1":"A"}  # 1 -> A 対応保険
        if filtered in mapping:
            return mapping[filtered]
        return filtered

    def _all_same_rank(self, cards: List[str]) -> bool:
        ranks = [self._extract_rank(c) for c in cards]
        return len(set(ranks)) == 1

    def _is_straight(self, idx_list: List[int]) -> bool:
        if not idx_list:
            return False
        if len(idx_list) < 3:
            return False
        idx_list = sorted(idx_list)
        # Joker を単純に無視 (高度な柔軟ストレート補完は後続)
        for i in range(1, len(idx_list)):
            if idx_list[i] - idx_list[i-1] != 1:
                return False
        return True

    # ---------------- Save/Load ----------------
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "num_players": self.num_players}, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or "cpu")
        model = ActionPolicyValueNet(num_players=ckpt.get("num_players",4))
        model.load_state_dict(ckpt["state_dict"])
        return model


__all__.append("ActionPolicyValueNet")
