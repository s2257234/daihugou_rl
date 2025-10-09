
"""PolicyValueNet 実装概要 (フル特徴専用 / model_format_version=3)

このファイルは AlphaZero 系大富豪エージェント用の Policy-Value ネットワークを提供する。

====================================
設計方針 / 現状仕様
====================================
1. 入力特徴 (full_input のみ / 簡易入力廃止)
    - 形式: 1 次元ベクトル (float32) 長さ full_feature_dim
    - 最新レイアウト v4 (v3 + FieldCards + PlayHistory) : 59N + 125
        * Self 55
        * OppSummary 5(N-1)
        * Field 22
        * FieldCards 53
        * PlayHistory 53
        * Belief 53(N-1)
        * Turn N
        合計: (55 + 22 + 53 + 53) + 58(N-1) + N = 59N + 125
    - 直前レイアウト v3 (部分観測 + belief, rank4) : 59N + 19
        * Self: 53 card bits + pass + remain = 55
        * OppSummary: (remain + rank4 one-hot) * (N-1) = 5(N-1)
        * Field: 1 revolution + 7 combo + 13 rank base + 1 field_size_norm = 22
        * Belief: 53 * (N-1)
        * Turn: N  → 合計 59N + 19
    - 旧レイアウト v2 (rank5) : 60N + 18 （互換読み込みのみ。新規生成しない）
    - さらに旧フル情報 v1 : 56N + 22 （全プレイヤー手札ビットを含む完全情報）
    - 生成は `agents.drl_agent._extract_state` が行い、full_input_dim を常に期待値に揃える。

2. 圧縮形式 (full_compact / cfv1)
    - 辞書キー: {'packed_bits','floats','binary_len','num_players','format','full_input_dim'}
    - packed_bits: binary_len 個のビット列を packbits した bytes
    - floats: 残りの連続値 (float16 配列)
    - 復元: unpackbits → float32 変換 → floats(float16→float32) を結合し full_input
    - モデルは state に full_input が無い場合 compact を自動展開 (展開失敗は無視)

3. 入力検証 / 自動補正
    - モデル内部 _encode_state で in_features と長さが異なる場合: pad または truncate
    - 初回のみ警告 [WARN] full_input dim mismatch ...
    - 根本的には生成側で常に正しい長さを保証することが前提 (補正は最終防衛線)

4. 出力
    - policy_head: shape [max_policy_size] (合法手リスト側で先頭 K 要素を切り出し softmax)
    - value_head: shape [num_players] 各プレイヤーが「次に上がる / 勝利フェーズ達成」確率 (Sigmoid)
    - value の意味合いは学習戦略に応じて再定義可能 (順位/報酬ベクトル拡張など)

5. 保存 / ロード
    - save() で以下メタ情報を付与: max_policy_size, hidden_size, num_players,
      use_full_features=True, full_feature_dim, model_format_version=3
    - load():
         * state_dict ラップ形式 (現行 / v2 以前) と state_dict 直保存 Legacy を判別
         * full_feature_dim 欠落時は backbone.0.weight の in_features から推定
         * 簡易入力互換コードは削除済み (常にフル特徴モデルへ再構築)

6. 後方互換ポリシー
    - model_format_version < 3 でも weight shape から full_feature_dim 推定が成功すればロード可
    - 旧 ckpt が簡易入力用に極小 in_features を持っている場合はロード時に ValueError で通知

7. 例外戦略
    - state に full_input / full_compact が無い場合は即 ValueError (簡易入力廃止を明示)
    - compact 展開失敗は握りつぶし (フォーマット不正時) → その後 full_input 無ければ例外

8. 想定拡張
    - 追加特徴 (履歴, 役回数, 制約フラグ) は full_input 生成側で末尾に拡張し full_feature_dim を増加
    - ネットワーク側は in_features 増分再学習で対応 (ロード互換は別途変換スクリプトで支援)

9. 注意点
    - pad/truncate が頻出する状態はデータ生成側のバグを示唆 → 早期修正を推奨
    - value_head の解釈が将来変更される場合、学習済み ckpt の互換性に留意

"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self,
                 max_policy_size: int = 128,
                 hidden_size: int = 128,
                 num_players: int = 4,
                 device: Optional[str] = None,
                 use_full_features: bool = True,
                 full_feature_dim: Optional[int] = None):
        """Policy-Value Net (フル特徴専用)

        簡易入力 (hand_size/field_size/turn_onehot) のフォールバックを廃止し、常に
        full_input もしくは full_compact (cfv1) を要求する。旧 ckpt との互換性維持のため
        use_full_features 引数は残すが False 指定は例外とする。
        """
        super().__init__()
        if not use_full_features:
            raise ValueError("簡易入力モードは廃止されました。必ず full_feature_dim を指定してください。")
        if full_feature_dim is None:
            raise ValueError("full_feature_dim が必須です。")
        self.max_policy_size = max_policy_size
        self.num_players = num_players
        self.device = torch.device(device) if device else torch.device("cpu")
        self.use_full_features = True
        input_dim = full_feature_dim

        h = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(h, max_policy_size)
        # value_head: ロジット出力 (Sigmoid は呼び出し側で適用 / BCEWithLogitsLoss 用)
        self.value_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, num_players),
        )
        self.to(self.device)

    def _encode_state(self, state: Dict[str, Any]):
        # フル特徴必須: full_input か full_compact が無ければ例外
        if ("full_input" in state) or ("full_compact" in state):
            import numpy as _np
            if 'full_compact' in state and 'full_input' not in state:
                try:
                    cf = state['full_compact']
                    if isinstance(cf, dict) and cf.get('format') == 'cfv1':
                        bin_len = int(cf.get('binary_len', 0))
                        packed = cf.get('packed_bits', b'')
                        floats = cf.get('floats')
                        if isinstance(packed, (bytes, bytearray)) and bin_len > 0:
                            bits_arr = _np.unpackbits(_np.frombuffer(packed, dtype=_np.uint8))[:bin_len].astype(_np.float32)
                        else:
                            bits_arr = _np.zeros(bin_len, dtype=_np.float32)
                        if floats is not None:
                            try:
                                floats_arr = _np.asarray(floats, dtype=_np.float16).astype(_np.float32)
                            except Exception:
                                floats_arr = _np.asarray(list(floats), dtype=_np.float32)
                        else:
                            floats_arr = _np.zeros(0, dtype=_np.float32)
                        state['full_input'] = _np.concatenate([bits_arr, floats_arr])
                except Exception:
                    pass
            arr = state.get('full_input')
            if arr is None:
                arr = []
            expected_dim: Optional[int] = None
            try:
                expected_dim = self.backbone[0].in_features  # type: ignore[index]
            except Exception:
                expected_dim = None

            def _finalize_vec(seq_like):
                # seq_like -> torch tensor with optional pad/truncate
                import numpy as _np
                vec = _np.asarray(list(seq_like), dtype=_np.float32)
                if expected_dim is not None and vec.shape[0] != expected_dim:
                    if vec.shape[0] < expected_dim:
                        padded = _np.zeros(expected_dim, dtype=_np.float32)
                        padded[:vec.shape[0]] = vec
                        vec = padded
                    else:
                        vec = vec[:expected_dim]
                    if not hasattr(self, '_warned_full_dim_mismatch'):
                        print(f"[WARN] full_input dim mismatch (got={len(seq_like)}, expected={expected_dim}) -> auto pad/truncate")
                        self._warned_full_dim_mismatch = True  # type: ignore[attr-defined]
                return torch.tensor(vec, dtype=torch.float32, device=self.device)

            if isinstance(arr, (list, tuple)):
                return _finalize_vec(arr)
            try:
                import numpy as _np
                if isinstance(arr, _np.ndarray):
                    return _finalize_vec(arr)
            except Exception:
                pass
            if hasattr(arr, 'detach'):
                t = arr.detach().float().to(self.device)
                if expected_dim is not None and t.numel() != expected_dim:
                    if t.numel() < expected_dim:
                        padded = torch.zeros(expected_dim, dtype=torch.float32, device=self.device)
                        padded[:t.numel()] = t
                        t = padded
                    else:
                        t = t[:expected_dim]
                    if not hasattr(self, '_warned_full_dim_mismatch'):
                        print(f"[WARN] full_input tensor dim mismatch (got={t.numel()}, expected={expected_dim}) -> auto pad/truncate")
                        self._warned_full_dim_mismatch = True  # type: ignore[attr-defined]
                return t

        raise ValueError("state に full_input / full_compact が存在しません (簡易入力廃止)。")

    def forward(self, state: Dict[str, Any]):
        x = self._encode_state(state)
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value_vec = self.value_head(h)
        return policy_logits, value_vec

    def forward_batch(self, states: List[Dict[str, Any]]):
        if not states:
            x = self._encode_state({})
            _ = self.backbone(x)
            return torch.empty(0, self.max_policy_size, device=self.device), torch.empty(0, self.num_players, device=self.device)
        xs = torch.stack([self._encode_state(s) for s in states], dim=0)
        h = self.backbone(xs)
        policy_logits = self.policy_head(h)
        value_vec = self.value_head(h)
        return policy_logits, value_vec

    def save(self, path: str):
        try:
            input_dim = self.backbone[0].in_features  # type: ignore[index]
        except Exception:
            input_dim = None
        ckpt = {
            "state_dict": self.state_dict(),
            "max_policy_size": self.max_policy_size,
            "hidden_size": self.policy_head.in_features,
            "num_players": self.num_players,
            "use_full_features": True,
            "full_feature_dim": int(input_dim) if input_dim else None,
            "model_format_version": 3,
        }
        # 非同期 I/O 設定が利用可能ならオフロード
        try:
            from agents.config import ALPHA_ZERO_CONFIG as _CFG
        except Exception:
            _CFG = {}
        enable_async = bool(_CFG.get("enable_async_io", False))
        if enable_async:
            try:
                from utils.async_io import get_async_io
                aio = get_async_io(_CFG)
                if aio:
                    aio.enqueue_torch_save(ckpt, path)
                    return
            except Exception:
                pass
        torch.save(ckpt, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "PolicyValueNet":
        try:
            ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=map_location or "cpu")
        except Exception:
            ckpt = torch.load(path, map_location=map_location or "cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            max_policy_size = ckpt.get("max_policy_size", 128)
            hidden_size = ckpt.get("hidden_size", 128)
            num_players = ckpt.get("num_players", 4)
            full_dim = ckpt.get("full_feature_dim")
            if full_dim is None:
                # 旧 ckpt 推定: backbone.0.weight から in_features
                w = state_dict.get("backbone.0.weight", None)
                if w is None:
                    raise ValueError("旧チェックポイントから full_feature_dim を推定できません。")
                full_dim = w.shape[1]
        else:  # 完全旧形式 (state_dict 直保存)
            state_dict = ckpt
            max_policy_size = 128
            hidden_size = 128
            num_players = 4
            w = state_dict.get("backbone.0.weight", None)
            if w is None:
                raise ValueError("旧形式 ckpt に backbone.0.weight が存在しません。")
            full_dim = w.shape[1]

        model = PolicyValueNet(max_policy_size=max_policy_size, hidden_size=hidden_size,
                               num_players=num_players, use_full_features=True, full_feature_dim=int(full_dim))
        model.load_state_dict(state_dict)
        # map_location を指定していた場合はモデル本体をそのデバイスへ移動し device 属性を同期
        if map_location:
            try:
                dev = torch.device(map_location)
                model.to(dev)
                model.device = dev  # type: ignore[attr-defined]
            except Exception:
                pass
        return model


__all__ = ["PolicyValueNet"]
