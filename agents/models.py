
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
import warnings

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

        # ---- Feature partition (v4 layout) ----
        # Self: 55
        self.self_dim = 55
        # Belief: 53 * (N-1)
        self.belief_dim = 53 * (num_players - 1)
        # Context: OppSummary 5*(N-1) + Field 22 + FieldCards 53 + PlayHistory 53 + Turn N
        self.context_dim = 5 * (num_players - 1) + 22 + 53 + 53 + num_players
        # Expected full feature dimension (v4): 59N + 125
        self.full_feature_dim = self.self_dim + self.belief_dim + self.context_dim

        # 互換: 引数の full_feature_dim が与えられ、計算値と異なる場合は警告の上で採用
        if full_feature_dim is not None and int(full_feature_dim) != int(self.full_feature_dim):
            # 最終防衛線として、明示指定を優先して各セクションの配分は既定値のままにし、パディング/切り詰めで吸収する
            # ここでは metadata 用に保持のみ行う
            self.full_feature_dim = int(full_feature_dim)

        # ---- Small encoders for each component ----
        # 出力次元は例に倣って Self=32, Belief=64, Context=32
        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_dim, 32),
            nn.ReLU(),
        )
        self.belief_encoder = nn.Sequential(
            nn.Linear(self.belief_dim, 64),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 32),
            nn.ReLU(),
        )

        # ---- Backbone & Heads ----
        h = hidden_size
        backbone_input_dim = 32 + 64 + 32  # encoders' output dims
        self.backbone = nn.Sequential(
            nn.Linear(backbone_input_dim, h),
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
            # 期待次元は raw full_input 長（バックボーン入力ではない）
            expected_dim: Optional[int] = int(getattr(self, 'full_feature_dim', 0)) or None

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

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        """1D ベクトルを (1, D) に整形し、デバイス/型を合わせる補助。"""
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _pad_or_truncate(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """x の最終次元を dim に合わせて pad/truncate する。"""
        cur = x.size(-1)
        if cur == dim:
            return x
        if cur < dim:
            pad = torch.zeros(x.size(0), dim - cur, device=self.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        # truncate
        return x[:, :dim]

    def _forward_from_tensor(self, x: torch.Tensor):
        """full_input テンソルからエンコーダ/バックボーン/ヘッドを通す共通経路。

        入力: x shape = [B, full_feature_dim] or [full_feature_dim]
        出力: policy_logits [B, max_policy_size], value_vec [B, num_players]
        """
        x = self._ensure_2d(x)
        # safety: pad/truncate to expected dimension
        x = self._pad_or_truncate(x, self.full_feature_dim)

        # Split into components
        parts = torch.split(x, [self.self_dim, self.belief_dim, self.context_dim], dim=1)
        if len(parts) != 3:
            # 極端な不一致時はフォールバックで全体を backbone へ（安全側）
            h = self.backbone(x.new_zeros(x.size(0), 32 + 64 + 32))
            return self.policy_head(h), self.value_head(h)
        self_feat, belief_feat, context_feat = parts

        # Encode each component
        self_emb = self.self_encoder(self_feat)
        belief_emb = self.belief_encoder(belief_feat)
        context_emb = self.context_encoder(context_feat)
        combined = torch.cat([self_emb, belief_emb, context_emb], dim=1)

        # Backbone + Heads
        h = self.backbone(combined)
        policy_logits = self.policy_head(h)
        value_vec = self.value_head(h)
        return policy_logits, value_vec

    def forward(self, state_or_x: Any):
        """互換維持のため、辞書(state) か 1D/2D テンソルの両方を受け付ける。

        - 辞書: _encode_state で full_input ベクトルへ変換
        - テンソル: そのまま full_input として扱う
        単一サンプル入力時は旧APIと同様、1D を返す。
        """
        if isinstance(state_or_x, dict):
            x = self._encode_state(state_or_x)
        else:
            x = state_or_x  # assume tensor-like
        pol, val = self._forward_from_tensor(x)
        # 旧 forward は単一サンプルで 1D を返していたため互換のため squeeze
        if pol.dim() == 2 and pol.size(0) == 1:
            pol = pol.squeeze(0)
        if val.dim() == 2 and val.size(0) == 1:
            val = val.squeeze(0)
        return pol, val

    def forward_batch(self, states: List[Dict[str, Any]]):
        if not states:
            # 空バッチ互換
            return (
                torch.empty(0, self.max_policy_size, device=self.device),
                torch.empty(0, self.num_players, device=self.device),
            )
        xs = torch.stack([self._encode_state(s) for s in states], dim=0)
        return self._forward_from_tensor(xs)

    # --- Optional variable-length API (compat shim) ---
    # 可変長アクション対応モデルが未実装の場合の互換用 evaluate。
    # 既存の固定ヘッド forward を内部で呼び出し、legal_actions の長さ n に合わせて
    # 先頭 n 要素のロジットを返す。value はプレイヤー数ぶんのベクター（確率; Sigmoid 適用済み）。
    # 注意: この実装は supports_variable_actions を自動では有効化しません（既存の
    #       バッチ推論経路を温存するため）。必要な場合は呼び出し側で
    #       model.supports_variable_actions = True を設定してください。
    def evaluate(self, state: Dict[str, Any], legal_actions: List[Any]):
        """Return (policy_logits_for_legal, value_vector) for variable-length actions.

        - policy_logits_for_legal: 長さ len(legal_actions) のロジット配列（softmax は呼び出し側）
        - value_vector: 長さ num_players の確率ベクター（Sigmoid 済み）

        このメソッドは固定ヘッドモデルの便宜用ラッパーであり、可変長モデルの厳密な
        行動エンコード（例: アクションID辞書）を行いません。既存の方針（legal の並びに
        対応して先頭から切り出す）に合わせます。
        """
        try:
            import torch as _t
            _use_amp = bool(getattr(self.device, 'type', None) == 'cuda')
            with _t.no_grad():
                with _t.amp.autocast('cuda', enabled=_use_amp):
                    logits_full, value_vec_logits = self.forward(state)
        except Exception:
            # フォールバック（AMP なし / 例外吸収）
            logits_full, value_vec_logits = self.forward(state)

        n = len(legal_actions) if legal_actions is not None else 0
        # policy ロジットを legal 数に合わせて切り出し（不足は 0-padding）
        if hasattr(logits_full, 'shape'):
            import torch as _t
            if logits_full.shape[0] < n:
                pad = _t.zeros(n - logits_full.shape[0], device=getattr(logits_full, 'device', None), dtype=logits_full.dtype)
                logits_sel = _t.cat([logits_full, pad], dim=0)
            else:
                logits_sel = logits_full[:n]
            policy_logits = logits_sel.detach().cpu().tolist()
        else:
            policy_logits = list(logits_full)[:n]
            if len(policy_logits) < n:
                policy_logits += [0.0] * (n - len(policy_logits))

        # value を Sigmoid で確率化してベクター返却
        try:
            v_probs = value_vec_logits.sigmoid().detach().cpu().tolist()
        except Exception:
            try:
                raw = value_vec_logits.tolist() if hasattr(value_vec_logits, 'tolist') else list(value_vec_logits)
                import math as _m
                v_probs = [float(1.0/(1.0+_m.exp(-float(x)))) for x in raw]
            except Exception:
                # 予期せぬ型の場合は安全側に均一分布
                v_probs = [1.0 / float(self.num_players)] * int(self.num_players)

        return policy_logits, v_probs

    def save(self, path: str, *, force_sync: bool = False):
        ckpt = {
            "state_dict": self.state_dict(),
            "max_policy_size": self.max_policy_size,
            "hidden_size": self.policy_head.in_features,
            "num_players": self.num_players,
            "use_full_features": True,
            # 重要: full_feature_dim はバックボーン入力ではなく raw 特徴の全長
            "full_feature_dim": int(self.full_feature_dim),
            "model_format_version": 3,
        }
        # 非同期 I/O 設定が利用可能ならオフロード
        try:
            from agents.config import ALPHA_ZERO_CONFIG as _CFG
        except Exception:
            _CFG = {}
        enable_async = bool(_CFG.get("enable_async_io", False))
        # 原子的保存 (tmp -> replace) の際は同期保存が必要。
        # path が .tmp で終わる場合や force_sync=True の場合は async を無効化。
        if enable_async and (not force_sync) and (not str(path).endswith('.tmp')):
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
            # 古い PyTorch では weights_only 引数が存在しないため、従来ロードにフォールバック
            ckpt = torch.load(path, map_location=map_location or "cpu")
        except Exception:
            # weights_only=True で失敗するレガシー ckpt 用フォールバック。
            # weights_only=False による FutureWarning を一時的に抑制する。
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*weights_only=False.*",
                    category=FutureWarning,
                )
                ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)

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
