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
    "num_simulations": 96,           # 1手あたりのシミュレーション回数 (以前:128)
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
    "buffer_size": 200_000,           # リプレイバッファ最大サイズ
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
    # 周期保存: 大量エピソード実行時にエピソード間隔で世代チェックポイントを残す
    # 例) 100000 エピソードで 2000 間隔 -> 50 個保存
    "checkpoint_interval_episodes": 2000,       # 0 / None なら無効
    "keep_previous_model_opponent": True,       # 直前世代モデルを一部プレイヤーに割当てて多様性確保
    "previous_model_mix_players": 2,            # 学習プレイヤー以外から2人を過去モデル化
    "past_model_pool_size": 6,             # 過去6世代保持
    "opponent_mix_interval_episodes": 500, # 500エピソードごとに再割当
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
    # ---------------------------
    # ログ / 可視化
    # ---------------------------
    "log_dir": "logs",              # ログ出力ディレクトリ (CSV / TensorBoard)
    "enable_tensorboard": True,      # TensorBoard 出力を有効化
    "mcts_log_sample_rate": 0.15,    # MCTS ルート統計のサンプリング率
    "disable_mcts_log": True,        # True で mcts_samples.jsonl へ出力しない
    "clear_logs_on_start": True,     # 起動時に既存ログを消去 (Falseで残す)
    # ETA 表示調整
    "eta_smoothing_alpha": 0.25,     # エピソード時間 EMA 係数 (0=平均,1=最新のみ)
    "monotonic_eta": True,           # 残り時間推定を単調減少にクランプ
    # ---------------------------
    # リプレイ共有 / 構造
    # ---------------------------
    "use_shared_replay": True,       # 全エージェントで単一の共有リプレイバッファを使用
    "replay_recent_sample_ratio": 0.0,  # >0 なら直近一定割合を優先サンプリング (未実装placeholder)
    # ---------------------------
    # ハードウェア / デバイス
    # ---------------------------
    # 'auto' -> torch.cuda.is_available() なら 'cuda'、それ以外は 'cpu'
    # 明示的に 'cpu' / 'cuda' / 'cuda:0' などを指定することも可能
    "device": "auto",
}

__all__ = ["ALPHA_ZERO_CONFIG"]
