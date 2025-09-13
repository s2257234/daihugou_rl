"""AlphaZero / 大富豪 強化学習 設定ファイル

本ファイルはハイパーパラメータを集中管理するための辞書を提供します。
必要に応じて trainer 側や CLI から override してください。

カテゴリ:
- MCTS / 探索
- 学習 / 最適化
- モデル
- データ / パス

値は初期チューニング用の仮値です。ゲーム特性や計算資源に合わせて調整してください。
"""
from __future__ import annotations

ALPHA_ZERO_CONFIG = {
    # ---------------------------
    # MCTS / 探索
    # ---------------------------
    # MCTS シミュレーション回数 (初期は重いので軽量値。性能向上後に再調整) 
    "num_simulations": 32,           # 1手あたりのシミュレーション回数 (以前:128)
    "puct_c": 1.4,                   # PUCT 探索定数
    "dirichlet_alpha": 0.3,          # Dirichlet ノイズ α (ルート)
    "dirichlet_epsilon": 0.25,       # ノイズ混合率 ε
    "temperature": 1.0,              # 方策サンプリング温度 (序盤高く終盤低くする調整可)
    "temperature_decay_moves": 20,   # この手数以降は温度を 0 (argmax) にする等のスケジューリング用目安

    # ---------------------------
    # モデル
    # ---------------------------
    "max_policy_size": 128,          # policy ログits の固定長 (合法手数 <= この値)
    "hidden_size": 128,              # MLP 隠れ層次元
    "num_players": 4,                # 大富豪 4人

    # ---------------------------
    # 学習 / 最適化
    # ---------------------------
    "buffer_size": 50_000,           # リプレイバッファ最大サイズ
    "batch_size": 256,               # 学習バッチサイズ (train_step 実装時に利用)
    "lr": 1e-4,                      # 学習率
    "weight_decay": 1e-4,            # L2 正則化
    "value_loss_coef": 1.0,          # 価値損失係数
    "policy_loss_coef": 1.0,         # 方策損失係数
    "entropy_coef": 1e-3,            # エントロピー正則化 (過学習防止 / 探索促進)
    "epochs_per_update": 1,          # 1回の train 呼び出しで何エポック回すか

    # ---------------------------
    # データ / 保存パス
    # ---------------------------
    "checkpoint_dir": "checkpoints",           # モデル保存ディレクトリ
    "checkpoint_path": "checkpoints/policy_value_latest.pt",  # 直近モデル
    "replay_path": "replay_buffer.joblib",     # リプレイバッファ保存先

    # ---------------------------
    # ログ / デバッグ
    # ---------------------------
    "log_interval": 50,              # 何手 or 何エピソードごとにログ出力するか (trainer 実装で使用想定)
    "seed": 42,                      # 乱数シード (再現性確保)
    # ---------------------------
    # 実行安全性 / 時間制御
    # ---------------------------
    "max_episode_steps": 800,        # 1エピソードのステップ上限 (無限長防止 / 強制打ち切り)
}

__all__ = ["ALPHA_ZERO_CONFIG"]
