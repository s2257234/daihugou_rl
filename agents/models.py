
"""PolicyValueNet 実装概要 (フル特徴専用 / model_format_version=3)

このファイルは AlphaZero 系大富豪エージェント用の Policy-Value ネットワークを提供する。

====================================
設計方針 / 現状仕様
====================================
1. 入力特徴 (full_input のみ / 簡易入力廃止)
    - 形式: 1 次元ベクトル (float32) 長さ full_feature_dim
    - 最新レイアウト v9 (しばり特徴追加): 73N + 79
        * Self 55
        * OppSummary 5(N-1)
        * Field 27 (revolution 1 + combo 7 + rank_base 13 + field_size_norm 1 + shibari 5)
        * FieldCards 53
        * Turn N
        * OpponentDiscards 53(N-1)
        * PassMatrix 13(N-1)
        * RankRemain 13
        * JokerRemain 1
        * is_leader 1
        * last_actor_onehot N
        合計: 55 + 27 + 53 + 13 + 1 + 1 + [5(N-1) + N + 53(N-1) + 13(N-1) + N] = 73N + 79
    - 旧レイアウト v8 (is_leader + last_actor_onehot 追加): 73N + 74
    - 旧レイアウト v5 (Belief/PlayHistory を除去 + OpponentDiscards + PassMatrix 拡張): 72N + 59
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
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import autograd

from game.action_vocab import canonical_action_keys as _canonical_action_keys


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



_MODEL_FALLBACK_LOGGED = set()


def _log_model_fallback_once(key: str, msg: str, exc: Exception | None = None) -> None:
    if key in _MODEL_FALLBACK_LOGGED:
        return
    _MODEL_FALLBACK_LOGGED.add(key)
    try:
        if exc is not None:
            print(f"{msg} ({type(exc).__name__}: {exc})")
        else:
            print(msg)
    except Exception:
        pass


class PolicyValueNet(nn.Module):
    def __init__(self,
                 max_policy_size: int = 128,
                 hidden_size: int = 128,
                 num_players: int = 4,
                 device: Optional[str] = None,
                 use_full_features: bool = True,
                 full_feature_dim: Optional[int] = None,
                 enable_hand_prediction_head: bool = True,
                 context_out_dim: int = 128,
                 signal_noise_std: float = 0.0):
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

        # ---- Feature partition (v9 layout: added shibari features) ----
        # Self: 55
        self.self_dim = 55
        # Context (v6 base): OppSummary 5*(N-1) + Field 27 + FieldCards 53 + Turn N
        # Note: Field expanded from 22 to 27 (added shibari: 1 active bit + 4 suit bits)
        base_context_dim = 5 * (num_players - 1) + 27 + 53 + num_players
        # Extensions:
        #   OpponentDiscards: 53*(N-1)
        #   PassMatrix: 13*(N-1)
        #   RankRemain: 13 (各ランク残枚数の正規化 (4-occ)/4)
        #   JokerRemain: 1 (ジョーカー残枚数の正規化 (1-occ))
        #   is_leader: 1 (場が空なら 1.0)
        #   last_actor_onehot: N (直前に出したプレイヤーの one-hot)
        self.opponent_discards_dim = 53 * (num_players - 1)
        self.pass_matrix_dim = 13 * (num_players - 1)
        self.rank_remain_dim = 13
        self.joker_remain_dim = 1
        self.is_leader_dim = 1
        self.last_actor_dim = num_players
        self.context_dim = (base_context_dim + self.opponent_discards_dim + self.pass_matrix_dim
                            + self.rank_remain_dim + self.joker_remain_dim
                            + self.is_leader_dim + self.last_actor_dim)
        # Expected full feature dimension (v9):
        # 55 + [5*(N-1) + 27 + 53 + N + 53*(N-1) + 13*(N-1) + 13 + 1 + 1 + N] = 73N + 79
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
            # legacy ckpt branch: no runtime prints here

        # ---- Small encoders for each component ----
        # 出力次元: Self=32, Context=128 (hidden_sizeに合わせる)。旧 ckpt 互換のため可変化。
        self.self_out_dim = 32
        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_dim, self.self_out_dim),
            nn.ReLU(),
        )
        # context_encoder は上記の条件分岐で context_dim が調整されている場合があるため、
        # ここで統一して定義（重複定義を回避）
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_out_dim),
            nn.ReLU(),
        )

        # ---- Backbone & Heads ----
        h = hidden_size
        self.hidden_size = h  # メタ保存用
        self.backbone_in_dim = self.self_out_dim + self.context_out_dim  # encoders' output dims (self + context)
        # ---- Residual Blocks (MLP ResNet style) ----
        # backbone 内に ResBlock x6 (Norm→ReLU→Linear→Norm→ReLU→Linear + Skip)
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
            ResBlock(h, self.signal_augmentation),
            ResBlock(h, self.signal_augmentation),
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
        # Value head: single scalar logit per sample (Sigmoid applied by caller when needed)
        # LayerNormを削除: value_head.0.weightがゼロになる問題を回避
        self.value_head = nn.Sequential(
            # 1層目: 特徴抽出 & 圧縮
            nn.Linear(h, h),
            nn.ReLU(),
            
            # 2層目: スカラー出力 (BCEWithLogitsLoss用ロジット)
            nn.Linear(h, 1)
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
                except Exception as e:
                    # full_compactからfull_inputへの変換に失敗した場合は、後続の処理でエラーになる
                    # 無意味に例外を握り潰さず、適切に処理する
                    import warnings
                    warnings.warn(f"Failed to convert full_compact to full_input: {e}", RuntimeWarning)
                    _log_model_fallback_once(
                        "full_compact_convert_failed",
                        "[model-fallback] full_compact conversion failed; will rely on full_input or raise",
                        e,
                    )
                    # full_inputが設定されなかった場合、後続の処理で適切にエラーが発生する
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
        # parts are already on the same device as `x` (moved above), no-op
        self_emb = self.self_encoder(self_feat)
        context_emb = self.context_encoder(context_feat)
        combined = torch.cat([self_emb, context_emb], dim=1)
        h0 = self.backbone_proj(combined)
        h = self.backbone(h0)
        policy_logits = self.policy_head(h)
        # value_head now outputs shape [B, 1] -> convert to [B] (squeeze last dim)
        value_logits = self.value_head(h)
        try:
            value = value_logits.squeeze(-1)
        except Exception:
            value = value_logits
        return policy_logits, value

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
        # 互換性: 単一サンプル時はバッチ次元を取り除く
        if hasattr(pol, 'dim') and pol.dim() == 2 and pol.size(0) == 1:
            pol = pol.squeeze(0)
        # val について: _forward_from_tensor は [B] またはスカラを返す
        try:
            if hasattr(val, 'dim') and val.dim() == 1 and val.size(0) == 1:
                val = val.squeeze(0)
        except Exception:
            pass
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
            value_logits = self.value_head(h)
            try:
                value_vec = value_logits.squeeze(-1)
            except Exception:
                value_vec = value_logits
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
        try:
            if hasattr(value_vec, 'dim') and value_vec.dim() == 1 and value_vec.size(0) == 1:
                value_vec = value_vec.squeeze(0)
        except Exception:
            pass
        if hand_logits is not None and hand_logits.dim() == 2 and hand_logits.size(0) == 1:
            hand_logits = hand_logits.squeeze(0)
        return policy_logits, value_vec, hand_logits

    def forward_batch(self, states: List[Dict[str, Any]]):
        if not states:
            # 空バッチ互換
            return (
                torch.empty(0, self.max_policy_size, device=self.device),
                torch.empty(0, device=self.device),
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
        except Exception as e:
            # フォールバック（AMP なし / 例外吸収）
            _log_model_fallback_once(
                "evaluate_amp_fallback",
                "[model-fallback] evaluate AMP failed; fallback to non-AMP forward",
                e,
            )
            logits_full, value_vec_logits = self.forward(state)

        n = len(legal_actions) if legal_actions is not None else 0
        # GPU 上で policy を softmax 処理し、確率値のみを CPU 転送（最適化）
        if hasattr(logits_full, 'shape'):
            import torch as _t
            if logits_full.shape[0] < n:
                pad = _t.zeros(n - logits_full.shape[0], device=getattr(logits_full, 'device', None), dtype=logits_full.dtype)
                logits_sel = _t.cat([logits_full, pad], dim=0)
            else:
                logits_sel = logits_full[:n]
            # GPU 上で softmax を実行してから CPU へ転送（1回のみ）
            policy_probs = _t.softmax(logits_sel, dim=0)
            policy_logits = policy_probs.cpu().tolist()
        else:
            policy_logits = list(logits_full)[:n]
            if len(policy_logits) < n:
                policy_logits += [0.0] * (n - len(policy_logits))

        # GPU 上で value を sigmoid 処理してから CPU 転送（最適化）
        try:
            import torch as _t
            if isinstance(value_vec_logits, _t.Tensor):
                v = value_vec_logits
                # GPU 上で sigmoid を実行
                v_prob_tensor = _t.sigmoid(v)
                # スカラー値を取得（1回の CPU 転送）
                if v.dim() == 0:
                    v_prob = float(v_prob_tensor.item())
                elif v.dim() == 1:
                    v_prob = float(v_prob_tensor[0].item()) if v.numel() > 0 else float(v_prob_tensor.item())
                else:
                    v_prob = float(v_prob_tensor.mean().item())
            else:
                raw = value_vec_logits.tolist() if hasattr(value_vec_logits, 'tolist') else list(value_vec_logits)
                import math as _m
                first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
                v_prob = float(1.0 / (1.0 + _m.exp(-float(first))))
        except Exception as e:
            # フォールバック: 中立値 0.5
            _log_model_fallback_once(
                "evaluate_value_fallback",
                "[model-fallback] evaluate value decode failed; using 0.5",
                e,
            )
            v_prob = 0.5

        return policy_logits, v_prob

    def canonical_action_keys(self) -> List[str]:
        """Return the full canonical action vocabulary in deterministic order."""
        keys = list(_canonical_action_keys(include_pass=True))
        max_k = int(getattr(self, 'max_policy_size', len(keys)) or len(keys))
        if max_k < len(keys):
            _log_model_fallback_once(
                "canonical_action_keys_truncated",
                f"[model-fallback] canonical_action_keys truncated: max_policy_size={max_k} < vocab={len(keys)}",
            )
            return keys[:max_k]
        if max_k > len(keys):
            # pad with PASS if larger than vocab
            keys.extend(['PASS'] * (max_k - len(keys)))
        return keys

    def save(self, path: str, *, force_sync: bool = False, logger=None):
        # state_dict()を取得
        state_dict = self.state_dict()
        
        # 保存前の検証: 重要なパラメータがゼロでないことを確認
        critical_params = [
            'self_encoder.0.weight',
            'context_encoder.0.weight',
            'backbone_proj.weight',
            'backbone.0.lin1.weight',
            'backbone.0.lin2.weight',
            'policy_head.0.weight',
            'value_head.0.weight',
        ]
        
        # 保存前の検証: 重要なパラメータがゼロでないことを確認
        # この検証は保存を中断する可能性があるため、例外処理を慎重に行う
        zero_params = []
        validation_error = None
        
        try:
            for key in critical_params:
                if key in state_dict:
                    param = state_dict[key]
                    if hasattr(param, 'abs'):
                        try:
                            abs_max = param.abs().max().item()
                            if abs_max < 1e-8:
                                zero_params.append(key)
                        except Exception as e:
                            # パラメータの検証中にエラーが発生した場合
                            validation_error = f"Failed to validate parameter {key}: {e}"
                            if logger:
                                try:
                                    logger.log_text(f"[save] WARNING: {validation_error}")
                                except Exception:
                                    pass
                            print(f"[save] WARNING: {validation_error}")
        except Exception as e:
            # 検証処理自体でエラーが発生した場合
            validation_error = f"Parameter validation process failed: {e}"
            if logger:
                try:
                    logger.log_text(f"[save] ERROR: {validation_error}")
                except Exception:
                    pass
            print(f"[save] ERROR: {validation_error}")
            # 検証に失敗した場合は保存を中断（安全性のため）
            raise ValueError(f"Cannot save checkpoint due to validation failure: {e}")
        
        # ゼロパラメータの処理
        if zero_params:
            # 重みパラメータのみをチェック（バイアスはゼロでも問題ない）
            weight_zero_params = [p for p in zero_params if 'weight' in p]
            bias_zero_params = [p for p in zero_params if 'bias' in p]
            
            if weight_zero_params:
                # 重みパラメータがゼロの場合は詳細な情報を収集して保存を中断
                # ゼロパラメータの詳細情報を収集
                zero_details = []
                for param_name in weight_zero_params[:10]:  # 最初の10個のみ
                    if param_name in state_dict:
                        param = state_dict[param_name]
                        try:
                            shape = param.shape if hasattr(param, 'shape') else 'unknown'
                            abs_max = param.abs().max().item() if hasattr(param, 'abs') else 0.0
                            abs_mean = param.abs().mean().item() if hasattr(param, 'abs') else 0.0
                            zero_details.append(f"{param_name}: shape={shape}, max={abs_max:.2e}, mean={abs_mean:.2e}")
                        except Exception:
                            zero_details.append(f"{param_name}: (failed to get details)")
                
                warning_msg = f"[save] WARNING: Found zero weight parameters before save: {weight_zero_params[:10]}{'...' if len(weight_zero_params) > 10 else ''}"
                error_msg = f"[save] ERROR: Cannot save checkpoint with zero weight parameters. Checkpoint save aborted."
                details_msg = f"[save] Zero parameter details:\n" + "\n".join(f"  {d}" for d in zero_details)
                
                if logger:
                    try:
                        logger.log_text(warning_msg)
                        logger.log_text(error_msg)
                        logger.log_text(details_msg)
                    except Exception:
                        pass
                
                print(warning_msg)
                print(error_msg)
                print(details_msg)
                
                # 保存を中断（ValueErrorを発生）
                raise ValueError(
                    f"Cannot save checkpoint with zero weight parameters: {weight_zero_params[:5]}{'...' if len(weight_zero_params) > 5 else ''}. "
                    f"This indicates a serious training issue. Please investigate why these parameters became zero."
                )
            elif bias_zero_params:
                # バイアスパラメータのみがゼロの場合は警告のみ（問題ない）
                if logger:
                    try:
                        logger.log_text(f"[save] INFO: Bias parameters are zero (this is normal): {bias_zero_params[:5]}{'...' if len(bias_zero_params) > 5 else ''}")
                    except Exception:
                        pass
        
        ckpt = {
            "state_dict": state_dict,
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
        # 防御的措置: 学習中に複数回上書きされがちな 'policy_value_best.pt' への
        # 中間保存を抑止する。最終保存（force_sync=True）や明示的に許可された場合は実行。
        try:
            best_defer = bool(_CFG.get('defer_best_mid_save', True)) if isinstance(_CFG, dict) else True
        except Exception:
            best_defer = True
        try:
            if best_defer and (not force_sync) and str(path).endswith('policy_value_best.pt'):
                try:
                    # ログは残すが保存はスキップ
                    if logger:
                        try:
                            logger.log_text(f"[save] INFO: Best-model save deferred by config: {path}")
                        except Exception:
                            pass
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
        # torch.save()実行
        try:
            torch.save(ckpt, path)
            
            # 保存後の検証: 設定でスキップ指定されていれば検証を行わない
            try:
                skip_verify = bool(_CFG.get("skip_checkpoint_verify", False)) if isinstance(_CFG, dict) else False
            except Exception:
                skip_verify = False

            if skip_verify:
                # 検証をスキップする代わりに簡易ログを残す（重複抑止あり）
                try:
                    _now_ts = __import__('time').time()
                    last = globals().get('_LAST_SAVE_LOG', None)
                    if last is None or not isinstance(last, dict):
                        last = {'path': None, 'ts': 0.0}
                    if last.get('path') != path or (_now_ts - float(last.get('ts', 0.0))) > 10.0:
                        if logger:
                            try:
                                logger.log_text(f"[save] INFO: Checkpoint saved (verification skipped by config): {path}")
                            except Exception:
                                pass
                        globals()['_LAST_SAVE_LOG'] = {'path': path, 'ts': _now_ts}
                except Exception:
                    try:
                        if logger:
                            logger.log_text(f"[save] INFO: Checkpoint saved (verification skipped by config): {path}")
                    except Exception:
                        pass
                
            else:
                # デフォルト: 保存後に再ロードして簡易検証を行う（元の挙動）
                try:
                    verify_ckpt = torch.load(path, map_location='cpu', weights_only=False)
                    verify_state_dict = verify_ckpt.get('state_dict', verify_ckpt) if 'state_dict' in verify_ckpt else verify_ckpt
                    
                    # 重要なパラメータが保存されているか確認
                    missing_critical = []
                    for key in critical_params:
                        if key not in verify_state_dict:
                            missing_critical.append(key)
                    
                    if missing_critical:
                        warning_msg = f"[save] WARNING: Critical parameters missing in saved checkpoint: {missing_critical[:5]}{'...' if len(missing_critical) > 5 else ''}"
                        if logger:
                            try:
                                logger.log_text(warning_msg)
                            except Exception:
                                pass
                        print(warning_msg)
                    else:
                        # 値が正しく保存されているか確認（最初の数個のパラメータのみ）
                        verify_ok = True
                        for key in critical_params[:3]:  # 最初の3つだけ確認
                            if key in state_dict and key in verify_state_dict:
                                orig = state_dict[key]
                                saved = verify_state_dict[key]
                                if hasattr(orig, 'abs') and hasattr(saved, 'abs'):
                                    orig_max = orig.abs().max().item()
                                    saved_max = saved.abs().max().item()
                                    if abs(orig_max - saved_max) > 1e-6:
                                        verify_ok = False
                                        break
                        
                        # Throttle duplicate save-log entries: avoid repeating identical messages
                        try:
                            _now_ts = __import__('time').time()
                            last = globals().get('_LAST_SAVE_LOG', None)
                            if last is None or not isinstance(last, dict):
                                last = {'path': None, 'ts': 0.0}
                            # log if path changed or more than 10s passed since last identical message
                            if last.get('path') != path or (_now_ts - float(last.get('ts', 0.0))) > 10.0:
                                if logger:
                                    try:
                                        logger.log_text(f"[save] Checkpoint saved and verified successfully: {path}")
                                    except Exception:
                                        pass
                                globals()['_LAST_SAVE_LOG'] = {'path': path, 'ts': _now_ts}
                        except Exception:
                            try:
                                if logger:
                                    logger.log_text(f"[save] Checkpoint saved and verified successfully: {path}")
                            except Exception:
                                pass
                except Exception as verify_e:
                    # 検証エラーは保存を失敗としない（警告のみ）
                    warning_msg = f"[save] WARNING: Checkpoint verification failed: {verify_e}"
                    if logger:
                        try:
                            logger.log_text(warning_msg)
                        except Exception:
                            pass
                    print(warning_msg)
                
        except Exception as e:
            error_msg = f"[save] ERROR: Exception during torch.save(): {e}"
            if logger:
                try:
                    logger.log_text(error_msg)
                except Exception:
                    pass
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise  # 例外を再発生させる

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "PolicyValueNet":
        """チェックポイントをロード（古いフォーマット対応、strict=False）"""
        try:
            ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        except TypeError as e:
            # 古い PyTorch では weights_only 引数が存在しないため、従来ロードにフォールバック
            _log_model_fallback_once(
                "load_weights_only_unsupported",
                "[model-fallback] torch.load weights_only unsupported; fallback to legacy load",
                e,
            )
            ckpt = torch.load(path, map_location=map_location or "cpu")
        except Exception as e:
            # weights_only=True で失敗するレガシー ckpt 用フォールバック。
            # weights_only=False による FutureWarning を一時的に抑制する。
            _log_model_fallback_once(
                "load_weights_only_failed",
                "[model-fallback] torch.load weights_only failed; fallback to weights_only=False",
                e,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*weights_only=False.*",
                    category=FutureWarning,
                )
                ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)

        # ckptの内容を安全に取得（Tensorの真偽値判定を回避）
        is_dict = isinstance(ckpt, dict)
        has_state_dict = False
        if is_dict:
            try:
                has_state_dict = "state_dict" in ckpt
            except Exception:
                has_state_dict = False
        
        if is_dict and has_state_dict:
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

        # Signal Augmentationを無効化（勾配爆発対策）
        model = PolicyValueNet(max_policy_size=max_policy_size,
                               hidden_size=hidden_size,
                               num_players=num_players,
                               use_full_features=True,
                               full_feature_dim=int(full_dim),
                               enable_hand_prediction_head=ckpt.get("has_hand_head", False),
                               context_out_dim=int(context_out_dim),
                               signal_noise_std=0.0)  # 常に無効化
        
        # hidden_sizeの不一致を検出して警告
        try:
            from agents.config import ALPHA_ZERO_CONFIG
            config_hidden_size = ALPHA_ZERO_CONFIG.get("hidden_size", hidden_size)
            if hidden_size != config_hidden_size:
                print(f"[load] WARNING: Checkpoint hidden_size ({hidden_size}) differs from config hidden_size ({config_hidden_size}). "
                      f"This may cause shape mismatches when loading parameters (strict=False).")
        except Exception:
            pass  # configの読み込みに失敗しても続行
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
        def _convert_legacy_value_head(sd, ms):
            """Convert old LayerNorm付き value_head to current 2-layer MLP safely.

            実際に旧形式の構造が検出された場合のみキーを書き換える。誤検知による
            正常キー削除を防ぐため、以下を満たすときだけ変換する:
              - モデルが現行の value_head.2.* を持つ
              - チェックポイントが LayerNorm 付き旧構造を示唆する
                (value_head.1.* が LayerNorm 用に存在する、もしくは value_head.3.weight のみ存在し
                 value_head.2.weight が欠落している)
            戻り値: 変換を実行した場合 True
            """
            has_layernorm = ('value_head.1.weight' in sd) or ('value_head.1.bias' in sd)
            has_old_final = 'value_head.3.weight' in sd
            missing_new_final = 'value_head.2.weight' not in sd
            model_wants_new = 'value_head.2.weight' in ms  # 現行構造

            if not model_wants_new:
                return False  # 現行モデルでなければ何もしない

            # 旧構造と判定できる場合のみ変換を実施
            if not (has_layernorm or (has_old_final and missing_new_final)):
                return False

            old_w = sd.get('value_head.3.weight')
            old_b = sd.get('value_head.3.bias')
            if old_w is None:
                return False

            import torch as _t
            ow = _t.as_tensor(old_w) if not isinstance(old_w, _t.Tensor) else old_w
            target_shape = tuple(ms['value_head.2.weight'].shape)

            if ow.shape == target_shape:
                sd['value_head.2.weight'] = ow.cpu().numpy() if not isinstance(old_w, _t.Tensor) else ow
                if old_b is not None and 'value_head.2.bias' in ms:
                    ob = _t.as_tensor(old_b) if not isinstance(old_b, _t.Tensor) else old_b
                    if ob.shape == tuple(ms['value_head.2.bias'].shape):
                        sd['value_head.2.bias'] = ob.cpu().numpy() if not isinstance(old_b, _t.Tensor) else ob
            elif ow.dim() == 2 and target_shape[0] == 1:
                # 旧: 複数出力 → 新: 単一出力の平均化
                nw = ow.mean(dim=0, keepdim=True)
                sd['value_head.2.weight'] = nw.cpu().numpy() if not isinstance(old_w, _t.Tensor) else nw
                if old_b is not None and 'value_head.2.bias' in ms:
                    ob = _t.as_tensor(old_b) if not isinstance(old_b, _t.Tensor) else old_b
                    nb = _t.tensor([float(ob.mean())], dtype=ob.dtype)
                    sd['value_head.2.bias'] = nb.cpu().numpy() if not isinstance(old_b, _t.Tensor) else nb
            else:
                return False  # 形状不一致で変換不可

            # 旧 LayerNorm / 旧最終層のキーを安全に削除
            for k in ['value_head.3.weight', 'value_head.3.bias', 'value_head.1.weight', 'value_head.1.bias']:
                if k in sd:
                    del sd[k]
            return True

        def _reinit_critical_if_zero(m: nn.Module):
            """Re-init critical layers if they ended up all-zero after load."""
            critical_params = [
                'self_encoder.0.weight',
                'context_encoder.0.weight',
                'backbone_proj.weight',
                'backbone.0.lin1.weight',
                'policy_head.0.weight',
                'value_head.0.weight',
            ]
            zero_params = []
            loaded_state_dict = m.state_dict()
            for key in critical_params:
                if key in loaded_state_dict:
                    param = loaded_state_dict[key]
                    if hasattr(param, 'abs'):
                        abs_max = param.abs().max().item()
                        if abs_max < 1e-8:
                            zero_params.append(key)

            def _reinit_linear(layer: nn.Linear, nonlinearity: str = 'relu'):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity=nonlinearity)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(layer.bias, -bound, bound)

            if 'value_head.0.weight' in zero_params:
                try:
                    _reinit_linear(m.value_head[0], nonlinearity='relu')
                    print(f"[load] INFO: Re-initialized value_head.0.weight (was zero)")
                    zero_params.remove('value_head.0.weight')
                except Exception as reinit_e:
                    print(f"[load] WARNING: Failed to re-initialize value_head.0.weight: {reinit_e}")
            if 'policy_head.0.weight' in zero_params:
                try:
                    _reinit_linear(m.policy_head[0], nonlinearity='relu')
                    print(f"[load] INFO: Re-initialized policy_head.0.weight (was zero)")
                    zero_params.remove('policy_head.0.weight')
                except Exception as reinit_e:
                    print(f"[load] WARNING: Failed to re-initialize policy_head.0.weight: {reinit_e}")
            if 'backbone_proj.weight' in zero_params:
                try:
                    _reinit_linear(m.backbone_proj, nonlinearity='relu')
                    print(f"[load] INFO: Re-initialized backbone_proj.weight (was zero)")
                    zero_params.remove('backbone_proj.weight')
                except Exception as reinit_e:
                    print(f"[load] WARNING: Failed to re-initialize backbone_proj.weight: {reinit_e}")
            if zero_params:
                warning_msg = f"[load] WARNING: Found zero parameters after load: {zero_params[:5]}{'...' if len(zero_params) > 5 else ''}"
                print(warning_msg)

        try:
            ms = model.state_dict()
            # value_head構造の互換性処理（旧LayerNorm付き ckpt → 現行2層MLP）
            if _convert_legacy_value_head(state_dict, ms):
                _log_model_fallback_once(
                    "convert_legacy_value_head",
                    "[model-fallback] converted legacy value_head to current layout",
                )

            # strict=False で互換性のあるパラメータのみロード（古いチェックポイント対応）
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # 形状不一致によるスキップを検出
            shape_mismatch_keys = []
            if missing_keys:
                # missing_keysに含まれるパラメータが、state_dictには存在するが形状が異なる可能性がある
                for key in missing_keys:
                    if key in state_dict:
                        ckpt_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else None
                        model_shape = model.state_dict().get(key)
                        if model_shape is not None and hasattr(model_shape, 'shape'):
                            model_shape = model_shape.shape
                            if ckpt_shape != model_shape:
                                shape_mismatch_keys.append((key, ckpt_shape, model_shape))
            
            if shape_mismatch_keys:
                print(f"[load] WARNING: Shape mismatches detected (parameters skipped due to strict=False):")
                for key, ckpt_shape, model_shape in shape_mismatch_keys[:10]:
                    print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")
                if len(shape_mismatch_keys) > 10:
                    print(f"  ... and {len(shape_mismatch_keys) - 10} more")
            
            if missing_keys or unexpected_keys:
                # デバッグ用に警告を出すが、エラーにはしない
                if missing_keys:
                    # 形状不一致でないmissing_keysのみ表示
                    non_shape_mismatch = [k for k in missing_keys if k not in [item[0] for item in shape_mismatch_keys]]
                    if non_shape_mismatch:
                        print(f"[INFO] Missing keys when loading checkpoint: {non_shape_mismatch[:5]}{'...' if len(non_shape_mismatch) > 5 else ''}")
                if unexpected_keys:
                    print(f"[INFO] Unexpected keys when loading checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
            # ロード後の検証: 重要パラメータのゼロ埋まりを検出し再初期化
            try:
                _reinit_critical_if_zero(model)
            except Exception as verify_e:
                # 検証エラーはロードを失敗としない（警告のみ）
                print(f"[load] WARNING: Load verification failed: {verify_e}")
        except TypeError as e:
            # 古いPyTorchでstrict引数がない場合
            _log_model_fallback_once(
                "load_strict_unsupported",
                "[model-fallback] load_state_dict strict unsupported; retry without strict",
                e,
            )
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Handle potential shape mismatch for value_head when migrating from
            # multi-player value outputs to single-scalar output.
            try:
                sd = state_dict
                ms = model.state_dict()
                # possible keys for final linear in checkpoint: 'value_head.2.weight'/'value_head.3.weight'/'value_head.weight'
                # 現在のモデル構造: value_head.0 (Linear), value_head.1 (ReLU), value_head.2 (Linear)
                # 古いチェックポイント（LayerNormあり）: value_head.0 (Linear), value_head.1 (LayerNorm), value_head.2 (ReLU), value_head.3 (Linear)
                old_w = sd.get('value_head.3.weight', None) or sd.get('value_head.2.weight', None) or sd.get('value_head.weight', None)
                old_b = sd.get('value_head.3.bias', None) or sd.get('value_head.2.bias', None) or sd.get('value_head.bias', None)
                # pick target keys present in model (現在の構造では value_head.2 が最終層)
                if 'value_head.2.weight' in ms:
                    new_w_key = 'value_head.2.weight'
                    new_b_key = 'value_head.2.bias'
                elif 'value_head.3.weight' in ms:
                    new_w_key = 'value_head.3.weight'
                    new_b_key = 'value_head.3.bias'
                else:
                    new_w_key = 'value_head.weight'
                    new_b_key = 'value_head.bias'
                if old_w is not None and new_w_key in ms:
                    import torch as _t
                    ow = _t.as_tensor(old_w) if not isinstance(old_w, _t.Tensor) else old_w
                    target_shape = tuple(ms[new_w_key].shape)
                    # If old had multiple outputs and new has single output, average rows
                    if ow.dim() == 2 and target_shape[0] == 1:
                        nw = ow.mean(dim=0, keepdim=True)
                        sd[new_w_key] = nw.cpu().numpy() if not isinstance(old_w, _t.Tensor) else nw
                        if old_b is not None and new_b_key in ms:
                            ob = _t.as_tensor(old_b) if not isinstance(old_b, _t.Tensor) else old_b
                            nb = _t.tensor([float(ob.mean())], dtype=ob.dtype)
                            sd[new_b_key] = nb.cpu().numpy() if not isinstance(old_b, _t.Tensor) else nb
                    # 形状が一致する場合はそのままコピー
                    elif ow.shape == target_shape:
                        sd[new_w_key] = ow.cpu().numpy() if not isinstance(old_w, _t.Tensor) else ow
                        if old_b is not None and new_b_key in ms:
                            ob = _t.as_tensor(old_b) if not isinstance(old_b, _t.Tensor) else old_b
                            if ob.shape == tuple(ms[new_b_key].shape):
                                sd[new_b_key] = ob.cpu().numpy() if not isinstance(old_b, _t.Tensor) else ob
                # try load again permissively
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception as e2:
                    _log_model_fallback_once(
                        "load_state_dict_retry",
                        "[model-fallback] load_state_dict strict=False failed; retrying strict",
                        e2,
                    )
                    model.load_state_dict(sd)
            except Exception:
                # fallback: re-raise original
                raise
        # どのロード経路でも最後にゼロチェックと再初期化を実行
        try:
            _reinit_critical_if_zero(model)
        except Exception as verify_e:
            print(f"[load] WARNING: Post-load verification failed: {verify_e}")
        # Signal Augmentationを確実に無効化（既存チェックポイントからロードした場合でも）
        # load_state_dict()後でも、signal_augmentationモジュールのstdを0.0に設定
        if hasattr(model, 'signal_augmentation') and model.signal_augmentation is not None:
            model.signal_augmentation.std = 0.0
            model.signal_noise_std = 0.0
            # ResBlock内のaugmentationも無効化（すべて同じインスタンスを参照しているが念のため）
            if hasattr(model, 'backbone'):
                for block in model.backbone:
                    if hasattr(block, 'aug') and block.aug is not None:
                        if hasattr(block.aug, 'std'):
                            block.aug.std = 0.0
        
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
