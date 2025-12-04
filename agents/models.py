
"""PolicyValueNet 実装概要 (フル特徴専用 / model_format_version=3)

このファイルは AlphaZero 系大富豪エージェント用の Policy-Value ネットワークを提供する。

====================================
設計方針 / 現状仕様
====================================
1. 入力特徴 (full_input のみ / 簡易入力廃止)
    - 形式: 1 次元ベクトル (float32) 長さ full_feature_dim
    - 最新レイアウト v5 (Belief/PlayHistory を除去 + OpponentDiscards + PassMatrix 拡張): 72N + 59
        * Self 55
        * OppSummary 5(N-1)
        * Field 22
        * FieldCards 53
        * Turn N
        * OpponentDiscards 53(N-1)
        * PassMatrix 13(N-1)
        合計: 55 + 22 + 53 + [5(N-1) + N + 53(N-1) + 13(N-1)] = 72N + 59
    - 旧レイアウト (v4: Belief除去後 / PlayHistoryあり) : 59N + 125 + 53(N-1) + 13(N-1) - 53
    - 直前レイアウト v3 (部分観測 + belief, rank4) : 59N + 19
        * Self: 53 card bits + pass + remain = 55
        * OppSummary: (remain + rank4 one-hot) * (N-1) = 5(N-1)
        * Field: 1 revolution + 7 combo + 13 rank base + 1 field_size_norm = 22
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

from typing import Dict, Any, Optional, List, Tuple
import warnings

import torch
import torch.nn as nn
from torch import autograd


class SignalAugmentFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, std: float, dims: int = 1) -> torch.Tensor:
        # ノイズは forward 時のみ乗せる。backward ではノイズを無かったものとして扱う。
        if std != 0:
            size = list(x.shape[:dims]) + [1] * (len(x.shape) - dims)
            noise = torch.randn(size, device=x.device, requires_grad=False) * float(std) + 1.0
            # ノイズを乗せたテンソルを返すが、勾配はノイズ無視で伝播させる（see backward）
            return x * noise
        else:
            return x

    @staticmethod
    def backward(ctx: Any, *gs: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[None], Optional[None]]:
        # 誤差逆伝播時はノイズが無かったものとして扱うため、入力に対する勾配はそのまま返す
        g_x = gs[0] if len(gs) > 0 else None
        return g_x, None, None


signal_augment = SignalAugmentFunction.apply


class SignalAugmentation(nn.Module):
    def __init__(self, std: float, dims: int = 1) -> None:
        super().__init__()
        self.std = float(std)
        self.dims = int(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std != 0:
            return signal_augment(x, float(self.std), int(self.dims))
        else:
            return x

    def extra_repr(self) -> str:
        return 'std={}, dim={}'.format(self.std, self.dims)



class PolicyValueNet(nn.Module):
    def __init__(self,
                 max_policy_size: int = 128,
                 hidden_size: int = 256,
                 num_players: int = 4,
                 device: Optional[str] = None,
                 use_full_features: bool = True,
                 full_feature_dim: Optional[int] = None,
                 enable_hand_prediction_head: bool = True,
                 context_out_dim: int = 256,
                 signal_noise_std: float = 0.2):
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

        # ---- Feature partition (v5 layout: no PlayHistory, Belief removed, with OpponentDiscards/PassMatrix) ----
        # Self: 55
        self.self_dim = 55
        # Context (v6 base): OppSummary 5*(N-1) + Field 22 + FieldCards 53 + Turn N
        base_context_dim = 5 * (num_players - 1) + 22 + 53 + num_players
        # New extensions:
        #   OpponentDiscards: 53*(N-1)
        #   PassMatrix: 13*(N-1)
        #   RankRemain: 13 (各ランク残枚数の正規化 (4-occ)/4)
        #   JokerRemain: 1 (ジョーカー残枚数の正規化 (1-occ))
        self.opponent_discards_dim = 53 * (num_players - 1)
        self.pass_matrix_dim = 13 * (num_players - 1)
        self.rank_remain_dim = 13
        self.joker_remain_dim = 1
        self.context_dim = base_context_dim + self.opponent_discards_dim + self.pass_matrix_dim + self.rank_remain_dim + self.joker_remain_dim
        # Expected full feature dimension (v6):
        # 55 + [5*(N-1) + 22 + 53 + N + 53*(N-1) + 13*(N-1) + 13 + 1] = 72N + 73
        self.full_feature_dim = int(self.self_dim + self.context_dim)
        # 先に context_out_dim を確定させておく（旧 ckpt 分岐で利用するため）
        self.context_out_dim = int(context_out_dim)
        # Signal augmentation strength: 0.0 disables augmentation
        self.signal_noise_std = float(signal_noise_std)
        # Prepare augmentation module (dims=1 for (N, HiddenSize) style tensors)
        self.signal_augmentation = SignalAugmentation(self.signal_noise_std, dims=1)
        # 互換: 旧 ckpt などで full_feature_dim が異なる場合は context 部分長を動的再計算
        if full_feature_dim is not None and int(full_feature_dim) != int(self.full_feature_dim):
            legacy_dim = int(full_feature_dim)
            new_context_dim = legacy_dim - self.self_dim
            if new_context_dim <= 0:
                raise ValueError(f"invalid full_feature_dim {legacy_dim}: must be > self_dim({self.self_dim})")
            self.full_feature_dim = legacy_dim
            self.context_dim = new_context_dim
            # context_encoder を再構築
            self.context_encoder = nn.Sequential(
                nn.Linear(self.context_dim, self.context_out_dim),
                nn.ReLU(),
            )
            # legacy ckpt branch: no runtime prints here

        # ---- Small encoders for each component ----
        # 出力次元: Self=32, Context=可変 (デフォルト256)。旧 ckpt 互換のため可変化。
        self.self_out_dim = 32
        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_dim, self.self_out_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_out_dim),
            nn.ReLU(),
        )

        # ---- Backbone & Heads ----
        h = hidden_size
        self.hidden_size = h  # メタ保存用
        self.backbone_in_dim = self.self_out_dim + self.context_out_dim  # encoders' output dims (self + context)
        # ---- Residual Blocks (MLP ResNet style) ----
        # 要望: backbone 内に ResBlock x2 (Norm→ReLU→Linear→Norm→ReLU→Linear + Skip)
        # 互換のため旧重み (backbone.0, backbone.2) は load() 時に投影へ移植。

        class ResBlock(nn.Module):
            def __init__(self, dim: int, aug: Optional[nn.Module] = None):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.lin1 = nn.Linear(dim, dim)
                self.norm2 = nn.LayerNorm(dim)
                self.lin2 = nn.Linear(dim, dim)
                self.aug = aug

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                out = self.norm1(x)
                out = torch.relu(out)
                out = self.lin1(out)
                out = self.norm2(out)
                out = torch.relu(out)
                out = self.lin2(out)
                # Apply signal augmentation (only during training)
                out = self.aug(out)
                return residual + out

        self.backbone_proj = nn.Linear(self.backbone_in_dim, h)
        self.backbone = nn.Sequential(
            ResBlock(h, self.signal_augmentation),
            ResBlock(h, self.signal_augmentation),
        )
        # Policy head: 旧 Linear(h->max_policy_size) から MLP 化 (Linear->ReLU->Linear)
        # 旧 ckpt 互換: load() 側で 'policy_head.weight' が存在する場合は最終層へ移植
        self.policy_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, max_policy_size),
        )
        # value_head: ロジット出力 (Sigmoid は呼び出し側で適用 / BCEWithLogitsLoss 用)
        self.value_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, num_players),
        )
        # 追加ヘッド: 相手手札予測（belief）
        self.enable_hand_prediction_head = bool(enable_hand_prediction_head)
        # belief 入力は削除したが、出力は (num_players-1)*53 を維持
        self.hand_pred_dim = 53 * (num_players - 1)
        if self.enable_hand_prediction_head:
            self.hand_head = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                # 拡張分（OpponentDiscards, PassMatrix）は context に含まれており、backbone 経由で hand_head 入力に反映される
                nn.Linear(h, self.hand_pred_dim),  # (num_players-1)*53
            )
        else:
            self.hand_head = None  # type: ignore[assignment]
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
                # Ensure tensor is created on the same device as model parameters
                try:
                    dev = getattr(self, 'device', None)
                    if dev is None:
                        dev = next(self.parameters()).device
                except Exception:
                    dev = torch.device('cpu')
                return torch.tensor(vec, dtype=torch.float32, device=dev)

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
                return t

        raise ValueError("state に full_input / full_compact が存在しません (簡易入力廃止)。")

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        """1D ベクトルを (1, D) に整形し、デバイス/型を合わせる補助。"""
        try:
            dev = getattr(self, 'device', None)
            if dev is None:
                dev = next(self.parameters()).device
        except Exception:
            dev = torch.device('cpu')
        x = x.to(dev).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _pad_or_truncate(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """x の最終次元を dim に合わせて pad/truncate する。"""
        cur = x.size(-1)
        if cur == dim:
            return x
        if cur < dim:
            try:
                dev = getattr(self, 'device', None)
                if dev is None:
                    dev = next(self.parameters()).device
            except Exception:
                dev = torch.device('cpu')
            pad = torch.zeros(x.size(0), dim - cur, device=dev, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        # truncate
        return x[:, :dim]

    def _forward_from_tensor(self, x: torch.Tensor):
        """full_input テンソルからエンコーダ/バックボーン/ヘッドを通す共通経路。

        入力: x shape = [B, full_feature_dim] or [full_feature_dim]
        出力: policy_logits [B, max_policy_size], value_vec [B, num_players]
        """
        x = self._ensure_2d(x)
        x = self._pad_or_truncate(x, self.full_feature_dim)
        # Ensure input tensor is on the same device as model parameters to avoid cpu/cuda mix
        try:
            dev = getattr(self, 'device', None)
            if dev is None:
                dev = next(self.parameters()).device
        except Exception:
            dev = torch.device('cpu')
        x = x.to(dev)
        parts = torch.split(x, [self.self_dim, self.context_dim], dim=1)
        if len(parts) != 2:
            h = self.backbone(x.new_zeros(x.size(0), self.backbone_in_dim))
            return self.policy_head(h), self.value_head(h)
        self_feat, context_feat = parts
        # Defensive: ensure each part is on model device
        # ensure parts are on model device
        try:
            self_feat = self_feat.to(dev)
            context_feat = context_feat.to(dev)
        except Exception:
            # best-effort move; if it fails, let subsequent ops raise
            pass
        self_emb = self.self_encoder(self_feat)
        context_emb = self.context_encoder(context_feat)
        combined = torch.cat([self_emb, context_emb], dim=1)
        h0 = self.backbone_proj(combined)
        h = self.backbone(h0)
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

    def forward_with_belief(self, state_or_x: Any):
        """拡張 forward.

        戻り値:
            policy_logits: shape [..., max_policy_size]
            value_vec:     shape [..., num_players] (ロジット; Sigmoid は呼び出し側)
            hand_logits:   shape [..., hand_pred_dim] or None (ロジット; Sigmoid は呼び出し側)

        以前の実装では hand_probs (Sigmoid 済み確率) を返していたが、
        学習を BCEWithLogitsLoss で統一するためロジットを直接返す仕様に変更。
        下位互換のため呼び出し側は hand_logits を受け取り必要に応じて torch.sigmoid。
        """
        if isinstance(state_or_x, dict):
            x = self._encode_state(state_or_x)
        else:
            x = state_or_x
        x = self._ensure_2d(x)
        x = self._pad_or_truncate(x, self.full_feature_dim)
        parts = torch.split(x, [self.self_dim, self.context_dim], dim=1)
        if len(parts) != 2:
            h = self.backbone(x.new_zeros(x.size(0), self.backbone_in_dim))
            policy_logits = self.policy_head(h)
            value_vec = self.value_head(h)
            hand_logits = None
        else:
            self_feat, context_feat = parts
            self_emb = self.self_encoder(self_feat)
            context_emb = self.context_encoder(context_feat)
            combined = torch.cat([self_emb, context_emb], dim=1)
            h0 = self.backbone_proj(combined)
            h = self.backbone(h0)
            policy_logits = self.policy_head(h)
            value_vec = self.value_head(h)
            if self.enable_hand_prediction_head and self.hand_head is not None:
                hand_logits = self.hand_head(h)
            else:
                hand_logits = None
        if policy_logits.dim() == 2 and policy_logits.size(0) == 1:
            policy_logits = policy_logits.squeeze(0)
        if value_vec.dim() == 2 and value_vec.size(0) == 1:
            value_vec = value_vec.squeeze(0)
        if hand_logits is not None and hand_logits.dim() == 2 and hand_logits.size(0) == 1:
            hand_logits = hand_logits.squeeze(0)
        return policy_logits, value_vec, hand_logits

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
            "hidden_size": int(getattr(self, "hidden_size", 256)),
            "num_players": self.num_players,
            "use_full_features": True,
            # 重要: full_feature_dim はバックボーン入力ではなく raw 特徴の全長
            "full_feature_dim": int(self.full_feature_dim),
            "model_format_version": 3,
            "has_hand_head": bool(self.enable_hand_prediction_head),
            # 互換のため、エンコーダ出力次元も保存
            "context_out_dim": int(getattr(self, "context_out_dim", 256)),
            # self_out_dim は固定32だが将来の変更に備える
            "self_out_dim": int(getattr(self, "self_out_dim", 32)),
        }
        # 非同期 I/O 設定が利用可能ならオフロード
        try:
            from agents.config import ALPHA_ZERO_CONFIG as _CFG
        except Exception:
            _CFG = {}
        # Respect config override: skip actual disk writes when disabled
        try:
            if bool(_CFG.get("disable_checkpoint_saving", False)):
                try:
                    # best-effort event log entry for visibility
                    import datetime, os
                    log_dir = _CFG.get("log_dir", "logs")
                    os.makedirs(log_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(os.path.join(log_dir, 'events.log'), 'a', encoding='utf-8') as f:
                        f.write(f"[{ts}] [save] checkpoint saving disabled by config -> {path}\n")
                except Exception:
                    pass
                return
        except Exception:
            pass
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
            # 互換: context/self 出力次元の検出
            context_out_dim = ckpt.get("context_out_dim")
            self_out_dim = ckpt.get("self_out_dim", 32)
            if context_out_dim is None:
                w_ctx = state_dict.get("context_encoder.0.weight")
                if w_ctx is not None and hasattr(w_ctx, "shape") and len(w_ctx.shape) == 2:
                    context_out_dim = int(w_ctx.shape[0])
                else:
                    w_bb = state_dict.get("backbone.0.weight")
                    if w_bb is not None and hasattr(w_bb, "shape") and len(w_bb.shape) == 2:
                        context_out_dim = max(int(w_bb.shape[1]) - int(self_out_dim), 1)
                    else:
                        context_out_dim = 32
        else:  # 完全旧形式 (state_dict 直保存)
            state_dict = ckpt
            max_policy_size = 128
            hidden_size = 128
            num_players = 4
            w = state_dict.get("backbone.0.weight", None)
            if w is None:
                raise ValueError("旧形式 ckpt に backbone.0.weight が存在しません。")
            full_dim = w.shape[1]
            context_out_dim = 32
            self_out_dim = 32

        model = PolicyValueNet(max_policy_size=max_policy_size,
                               hidden_size=hidden_size,
                               num_players=num_players,
                               use_full_features=True,
                               full_feature_dim=int(full_dim),
                               enable_hand_prediction_head=ckpt.get("has_hand_head", False),
                               context_out_dim=int(context_out_dim))
        # 旧 ckpt の単層 policy_head (policy_head.weight) を MLP 最終層へ移植
        try:
            if ("policy_head.weight" in state_dict and "policy_head.2.weight" in model.state_dict()):
                w_old = state_dict.get("policy_head.weight")
                b_old = state_dict.get("policy_head.bias")
                w_new = model.state_dict().get("policy_head.2.weight")
                if w_old is not None and w_new is not None and w_old.shape == w_new.shape:
                    state_dict["policy_head.2.weight"] = w_old
                    if b_old is not None and "policy_head.2.bias" in model.state_dict():
                        state_dict["policy_head.2.bias"] = b_old
        except Exception:
            pass
        try:
            model.load_state_dict(state_dict, strict=True)  # 追加ヘッド後互換
        except TypeError:
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
