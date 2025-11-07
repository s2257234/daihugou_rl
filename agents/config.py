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
    # 学習 / 最適化
    # ---------------------------
    "buffer_size": 250000,            # リプレイバッファ最大サイズ 
    "batch_size": 256,               # 学習バッチサイズ (train_step 実装時に利用)
    "lr": 7e-5,                      # 学習率　初期値0.0001
    "weight_decay": 7e-5,          # L2 正則化　初期値1e-4
    "value_loss_coef": 1.0,          # 価値損失係数
    "policy_loss_coef": 1.2,         # 方策損失係数
    "entropy_coef": 1e-3,             # エントロピー正則化
    "epochs_per_update": 1,          # 1回の train 呼び出しで何エポック回すか

    # --- 学習率スケジューラ ---
    # デフォルト: Warmup + Cosine (再開なし)
    # none | warmup_cosine
    "lr_scheduler": "warmup_cosine",
    "lr_warmup_steps": 3000,
    "lr_min": 3e-6,
    # ウォームアップ後、ここまでの更新ステップで lr_min へ到達
    "lr_cosine_T_max_updates": 500000, #目標ステップ数に応じて変更する
    # 再開直後に一度だけウォームアップをやり直す（プロセス内一回限り）
    # True にすると、optimizer/scheduler 復元後にスケジューラを last_epoch=-1 で再初期化し、
    # param_group['lr'] を基準学習率に戻します。その後このフラグは False に戻されます。
    "resume_reset_warmup_once": False,

    # ---------------------------
    # MCTS / 探索
    # ---------------------------
    # MCTS シミュレーション回数 (96で速度重視、tau=0.3の強力シャープ化でカバー) 
    "num_simulations": 96,       # 1手あたりのシミュレーション回数 (速度とバランス)　初期値96
    "puct_c": 1.1,                  # PUCT 探索定数 (1.4→1.2で探索を抑制、訪問集中を促進) 初期値1.0
    "dirichlet_alpha": 0.3,          # Dirichlet ノイズ α (ルート) 初期値0.3
    "dirichlet_epsilon": 0.05,       # ノイズ混合率 ε (0.15→0.10でノイズを削減、評価に基づく集中) 
    "temperature": 0.8,              # 方策サンプリング温度 (序盤高く終盤低くする調整可) 初期値1.0
    "temperature_decay_moves": 20,   # この手数以降は温度を 0 (argmax) にする等のスケジューリング用目安
    # 温度スケジュール（序盤高温→後半低温、自己対戦エピソード進行で高温手数を短縮）
    # デフォルト: 最初の10手は τ=0.6、それ以降は τ=0.1。エピソードが進むと高温手数を段階的に短縮（最低2手を維持）
    "temp_high_value": 0.8,            # 高温 τ (0.8→0.6に下げて過度なランダム性を抑制)
    "temp_low_value": 0.1,              # 低温 τ
    "temp_high_moves_initial": 4,      # 高温適用の初期手数
    "temp_high_moves_min": 2,           # 高温適用の最低手数
    "temp_high_moves_decay_every": 500, # 何エピソードごとに高温手数を1手短縮するか
    # 序盤ランダム化: 指定手数までは完全ランダムに行動 (探索温度の代替オプション)
    "opening_random_enable": True,    # True で有効化
    "opening_random_moves": 3,          # >0 で有効。例: 3 なら最初の3手をランダム行動
    "opening_random_include_pass": False,  # True なら pass もランダム候補に含める
    # 学習ターゲットπの温度（行動サンプリングとは分離）
    # 行動選択は高温（多様性確保）でも、学習用πは強力にシャープ化してエントロピーを劇的に下げる
    # 0.3でv^(1/0.3)=v^3.33正規化 → 訪問数格差を大幅に強調、低エントロピー教師信号を生成
    "policy_target_tau": 0.25,           # 学習ターゲット用温度 (1.0→0.3で強力シャープ化、高entropy問題に対処)
    # 追加: MCTS 高速化オプション（デフォルト有効化）
    "mcts_batch_eval_size": 64,      # 葉ノードのバッチ評価サイズ（1で無効同等）
    "enable_mcts_tt": True,          # トランスポジションテーブル有効化（品質不変で再計算を削減）
    "mcts_tt_capacity": 10000,       # キャッシュ上限を削減（簡易LRUでエビクション）

    # --- 不完全情報処理 / デターミニゼーション関連 ---
    # 不完全情報 (相手手札非公開) 前提で opponent hand をサンプリングするか
    # 学習方針: 学習時は fixed_once（1回サンプル固定）、推論時は stochastic（毎シム再サンプル）
    "enable_determinization": True,
    # 学習時/推論時のデターミニゼーションモード
    #   - "fixed_once": ルートで一度だけ割当をサンプリングし、MCTS全シミュレーションへ固定適用
    #   - "stochastic": 各シミュレーション毎に割当を再サンプル
    #   - "none": デターミニゼーションを行わず、環境の完全情報（真の手札）をそのまま使用
    # 自己対戦では「完全情報で1度だけ（= 各手番ごとに固定の完全情報でMCTS）」にするため、train は none を既定にする
    "determinization_mode_train": "none",
    "determinization_mode_eval": "stochastic",
    # 推論時は Dirichlet ノイズを無効化（安定した選択のため）
    "inference_dirichlet": False,
    # 並列 determinization プールを有効化 (True でバックグラウンドスレッドが割当候補を生成)
    "enable_parallel_determinization": True,
    # プール容量 (生成済み割当の最大保持数) メモリ削減: 128→64
    "det_pool_capacity": 64,
    # 再生成のための補充閾値 (容量 * ratio を下回ると生成ループが活発化)
    "det_pool_refill_threshold": 0.30,
    # プールからのサンプリング方式: fifo | random
    "det_pool_sampling": "fifo",
    # インライン/プール生成時の最大リトライ回数 (パス制約矛盾解消目的)
    "det_retry_max": 8,
    # 生成ワーカー数（0で無効＝インラインのみ）。Windows環境ではまず0で安全運用。
    "det_workers": 1,
    # ゲーム終了毎に determinization プールを停止しメモリ/署名ミスマッチを抑制するか
    # True: 各ゲーム開始時に必要なら再起動 (安定性/メモリ優先) / False: ゲーム間で継続 (微小性能最適化)
    "reset_det_pool_each_game": True,
    # per-move パフォーマンスログを抑制したい場合 False に (デバッグ用途 True 推奨)
    "enable_perf_log": False,
    # Early Stop (MCTS 収束早期打ち切り)
    # 有効化するとシミュレーション途中で十分収束した場合に打ち切り計算量を節約
    "mcts_early_stop_enable": True,          # False で完全無効化
    "mcts_early_stop_min_sims": 64,          # この回数までは必ず実行 (速度重視で32に戻す)
    "mcts_early_stop_visit_ratio": 0.65,     # ルート最大訪問子 / 総訪問 >= ratio で候補 (0.70→0.80で厳格化)
    "mcts_early_stop_gap_ratio": 0.10,       # (top - second)/総訪問 >= gap なら確定 (0.12→0.15で厳格化)
    "mcts_early_stop_log_sample_rate": 0.005,# 早期停止ログのサンプリング率 (ノイズ抑制)
    # min_sims 到達後に早期停止判定の粒度を細かくするためのバッチ縮小サイズ。
    # 例: 通常 batch_eval_size=64 だと判定が64刻みになり早期停止タイミングを逃す可能性がある。
    # min_sims を超えて以降は batch をこのサイズ以下に縮め、より細かく収束判定する。
    # 0 / 1 以下や未設定で無効化 (既存挙動)。
    "mcts_early_stop_post_min_batch": 8,
    # 早期停止デバッグ: 判定チェック毎に top_ratio / gap_ratio を低頻度で events.log 出力
    # True で有効化。高頻度になり過ぎないよう内部でサンプリング。
    "mcts_early_stop_debug": True,
    # Early Stop 計測ログ出力間隔 (何手ごとに平均シミュレーション数を events.log へ書くか)
    "mcts_sims_log_interval": 50,
    # 学習対象プレイヤー以外の MCTS 探索間引き設定
    # 学習プレイヤー (learning_player_id) は num_simulations を使用。
    # opponent_num_simulations > 0 なら絶対値で上書き。
    # それ以外は num_simulations * opponent_sim_scale を丸め、最低 opponent_sim_min を保証。
    "learning_player_id": 0,
    "opponent_num_simulations": 32,
    "opponent_sim_scale": 0.125,  # 約1/8
    "opponent_sim_min": 8,

    # ---------------------------
    # モデル
    # ---------------------------
    "max_policy_size": 128,          # policy ログits の固定長 (合法手数 <= この値)
    "hidden_size": 128,              # MLP 隠れ層次元
    "num_players": 4,                # 大富豪 4人

    

    # ---------------------------
    # 検証用バッファ / 検証評価
    # ---------------------------
    # リプレイを学習用/検証用に確率分割する比率 (0.0〜0.5 程度を推奨)
    # サンプルが確定(value 付与)したタイミングで一度だけ split を付与します。
    "val_split_ratio": 0.1,
    # 学習更新に対して何回に1回、検証損失を計算するか (0/None で検証無効)
    "val_eval_every_updates": 100,
    # 検証時に使用する最大サンプル数 (過大計算防止)。0/None で全件。
    "val_max_samples": 4096,
    # 検証時のバッチサイズ (未指定で学習バッチと同一)
    "val_batch_size": None,

    # ---------------------------
    # モデル更新前 評価ゲート
    # ---------------------------
    # 学習で得た候補モデルを採用する前に、直前モデルに対して同一配牌・先後交代で
    # 厳しめの勝率しきい値で判定する仕組み。
    # True で有効化。しきい値は 0.6 (60%)、対局数は100（= 同一配牌のペア×10）。
    "eval_gate_enable": True,
    "eval_gate_games": 40,
    "eval_gate_threshold": 0.55,
    # 非同期ゲート: 学習/自己対局を止めずにバックグラウンドで評価し、合格時のみ昇格
    "eval_gate_async": True,
    "eval_gate_workers": 2,
    # ゲート評価の乱数シード（Noneでランダム）。同一配牌実現のため random / numpy を固定。
    "eval_gate_seed": 20251031,
    # 評価中は探索ノイズ/序盤ランダムをOFFにして純粋実力を比較
    "eval_gate_disable_dirichlet": True,
    "eval_gate_disable_opening_random": True,
    # 評価ゲートの起動制御（学習更新回数ベース）
    # train_it がこの回数に到達するまで評価を開始しない（0/未設定で無効）
    "eval_gate_start_after_updates": 1000,
    # 直近の評価開始からこの更新数に達するまで次の評価を起動しない（0/未設定で無効）
    "eval_gate_every_updates": 1000,

    # ---------------------------
    # データ / 保存パス
    # ---------------------------
    "checkpoint_dir": "checkpoints",           # モデル保存ディレクトリ
    "checkpoint_path": "checkpoints/policy_value_latest.pt",  # 直近モデル
    # 周期保存: 大量エピソード実行時にエピソード間隔で世代チェックポイントを残す
    # 例) 100000 エピソードで 2000 間隔 -> 50 個保存
    "checkpoint_interval_episodes": 10000,       # 0 / None なら無効
    "updates_per_iter": 50,                 # ステップだけ学習
    "keep_previous_model_opponent": True,       # 直前世代モデルを一部プレイヤーに割当てて多様性確保
    "previous_model_mix_players": 2,            # 学習プレイヤー以外から2人を過去モデル化
    "past_model_pool_size": 1,             # メモリ削減: 過去1世代のみ保持
    "opponent_mix_interval_episodes": 500, # 500エピソードごとに再割当
    "replay_path": "replay_buffer.joblib",     # リプレイバッファ保存先
    "strict_lossless": False,        # True なら 圧縮しない
    # full_input をリプレイサンプルに保持するか (False で state.full_input / full_compact を破棄しメモリ節約)
    "store_full_input": True,

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
    # ログバッファリング制御 (False で即時書き込み、True でバッファ経由)
    "log_buffer_enabled": True,     # events.log が即座に表示されるよう無効化
    # メモリスナップショット (events.log へ 1h 毎など)
    "memory_log_interval_sec": 3600,
    # ETA 表示調整
    "eta_smoothing_alpha": 0.25,     # エピソード時間 EMA 係数 (0=平均,1=最新のみ)
    "monotonic_eta": True,           # 残り時間推定を単調減少にクランプ

    "use_progress_bar": False,
    "minimal_progress": False,  # 進捗表示を最小限に (ログ行数抑制)

    # ---------------------------
    # リプレイ共有 / 構造
    # ---------------------------
    "use_shared_replay": True,       # 全エージェントで単一の共有リプレイバッファを使用
    "replay_recent_sample_ratio": 0.0,  # >0 なら直近一定割合を優先サンプリング (未実装placeholder)

    # =====================================================
    # プロセス分離 / 非同期アーキテクチャ向け追加キー (初期版)
    # =====================================================
    # Self-Play 専用プロセスがサンプルをファイルシャードとして吐き出し
    # Learner (学習) 専用プロセスがそれを取り込む最小構成。
    # Windows でも動作するようシンプルなファイルベース (rename による原子化) を採用。
    # true で有効化しても既存の train_concurrent は維持される (併存可能)。
    "enable_process_decoupling": True,
    # サンプルシャード保存ディレクトリ (self-play プロセス側が生成)。
    "sample_shard_dir": "sample_shards",
    # 取り込み済みシャードの退避先 (None なら削除)。
    "sample_shard_ingested_dir": "sample_shards/_ingested",
    # 1 シャードに詰める最大サンプル数 (到達/エピソード終了でファイルへ flush)。
    "sample_shard_max_samples": 2000,
    # Learner がシャードをポーリングする間隔 (秒)。低すぎるとI/O増。
    "learner_poll_interval_sec": 5.0,
    # Learner が一度に取り込む最大シャード数 (負荷平準化)。0/None で無制限。
    "learner_max_shards_per_poll": 10,
    # モデル最新ファイルポーリング間隔 (Self-Play が最新モデルへ追随する周期)。
    "selfplay_model_reload_interval_sec": 30.0,
    # Self-Play プロセスの無限ループ安全停止フラグ (True で一定エピソード後終了)。
    "selfplay_max_episodes": 0,  # 0 で無限
    # シャードファイル拡張子 (衝突回避 & grep 用)。
    "sample_shard_ext": ".shard.joblib",
    # Learner が取り込み後に保持する ReplayBuffer サイズ上限 (既存 buffer_size と同義だが分離運用時に再確認のため)。
    "learner_buffer_size": 250000,
    # 取り込み時に古いサンプルをどれだけ優先削除するかの比率 (0.0～1.0)。 >0 で FIFO 削減を強制。
    "learner_ingest_purge_ratio": 0.0,
    # 評価 (ゲート) 用独立プロセスで使用する対局周期 (秒) 0 で毎ポーリング時評価判定。
    "gate_poll_interval_sec": 60.0,
    # 評価結果を learner へ通知する簡易ファイル (JSON)。
    "gate_result_path": "gate_results/latest_gate.json",
    # 候補モデル保存ディレクトリ (learner が世代 ckpt を置く)。
    "candidate_model_dir": "checkpoints/candidates",
    # ベストモデルファイル (self-play / evaluator が参照)。
    "best_model_path": "checkpoints/policy_value_best.pt",
    # 候補昇格しきい値 (ゲートプロセスが判定)。既存 eval_gate_threshold と同義だが分離簡易化。
    # ゲート昇格しきい値: gate_daemon も含め全体で eval_gate_threshold を唯一のキーとして使用
    # NOTE: gate_daemon / trainer 双方このキーを参照
    # 例: 0.55 -> 55% 超で採用
    "eval_gate_threshold": 0.55,
    # プロセス間簡易シグナルファイル (learner が生成し self-play が再読込を即時誘発)。空ファイルで可。
    "model_refresh_flag_path": "checkpoints/_refresh.flag",

    # ---------------------------
    # 自己対局 並列実行
    # ---------------------------
    # 並列ワーカー数 (0/1 で無効 = 単一プロセス)。Windows の spawn に対応。
    "selfplay_workers": 6,            # CPU 28 論理スレッドに合わせ並列度を拡大（様子を見て 12 まで）
    # ワーカープロセスでの推論デバイス。通常は CPU を推奨 (GPU 共有は非推奨)。

    "selfplay_worker_device": "cpu",
    # 並行学習トリガ: 新規サンプルがこの数だけ取り込まれたら学習を1バースト起動
    # 小さすぎると学習バーストが細切れになり効率低下。大きすぎると応答が遅れる
    "concurrent_min_new_samples_before_train": 1000,  # メモリ削減: 学習頻度を下げて蓄積抑制
    # Learner 分離プロセス用: 最低新規サンプル蓄積数（この数以上 ingest されたら学習バーストを開始）
    "learner_min_new_samples_before_train": 8000,
    # Learner がメモリ保持せずファイルスナップショットのみを使うモード (True で ingest 後メモリ即 purge)
    "learner_file_only_replay": True,
    # Learner が統合スナップショット replay_buffer.joblib を保存する間隔(秒)
    "learner_replay_snapshot_interval_sec": 900,
    # 学習後の最新チェックポイント保存の最短間隔(秒)。0以下で毎回保存（高I/O）
    "concurrent_latest_save_every_sec": 300.0,
    # ワーカー配布用モデル(pt)保存の最短間隔(秒)。0以下で毎回保存
    "concurrent_blob_save_every_sec": 900.0,

    # ---------------------------
    # ハードウェア / デバイス
    # ---------------------------
    # 'auto' -> torch.cuda.is_available() なら 'cuda'、それ以外は 'cpu'
    # 明示的に 'cpu' / 'cuda' / 'cuda:0' などを指定することも可能
    "device": "auto",
    # CPU 利用効率の最適化（並列ワーカーのスレッド数制御）
    # 既定では各ワーカー内の PyTorch が CPU スレッドを多く占有し、
    # 複数プロセス間で過剰スレッド競合が起こり総合スループットが伸びないことがあります。
    # torch_num_threads_workers を 1〜2 程度に下げると、ワーカー数×スレッドで
    # 物理コアに収まりやすく、CPU 使用率の頭打ちを改善できる場合があります。
    "torch_num_threads_workers": 1,   # 0/未設定で既定のまま。1〜2 を推奨値として用意
    "torch_num_threads_main": 3,      # メインプロセス（学習側）の CPU スレッド数（0で既定）
    # PyTorch inter-op 並列スレッド（オペレータ間並列）。未指定/0 で既定。
    # CPU 環境では 1 に下げるとオーバーヘッドが減ることがあります。
    "torch_num_interop_threads_workers": 1,
    "torch_num_interop_threads_main": 1,

    # ---------------------------
    # ログ最適化 (大量学習向け)
    # ---------------------------
    # TensorBoard 及び CSV への書き込み頻度を制御し I/O/ディスク負荷を軽減
    # 例: train 100 ステップに 1 回 / episode 10 回に 1 回
    "tensorboard_train_log_every": 200,      # 1 なら毎ステップ
    "tensorboard_episode_log_every": 200,
    "tensorboard_flush_seconds": 300,        # 最低この秒数ごとに flush (0/None なら都度 flush)
    # CSV 出力間引き (1=毎回)。間引いた行は欠番になる
    "csv_train_log_every": 200,
    "csv_episode_log_every": 200,
    # MCTS ルート統計 JSONL を更に抑制したい場合 (disable_mcts_log と組み合わせ)
    "mcts_jsonl_max_bytes": 50_000_000,     # 上限超過で以降追記停止 (約50MB)。0/None で無効

    # ---------------------------
    # CSV ログ制御
    # ---------------------------
    # True なら episodes.csv / train_updates.csv を一切生成しない
    "disable_csv_logging": False,
    # True なら逐次書き込みをせず、最後に 1 行だけ (最終エピソード指標 / 最終学習指標) を保存
    # disable_csv_logging が True の場合は無視される
    "csv_summary_only": False,
    # ---------------------------
    # 拡張特徴量 (フル状態入力) 設定
    # ---------------------------
    # True の場合、各プレイヤーの 53枚カード所持ビット + パスフラグ + 残枚数、
    # 場の役分類フラグ (single/pair/triple/four/straight/joker_single/empty)、
    # 革命フラグ、場枚数、場ランク(one-hot 13) と手番 one-hot を結合した
    # 高次元ベクトル (full_input) を PolicyValueNet へ入力する。
    # False の場合は従来の hand_size / field_size / turn one-hot の簡易入力。
    "use_full_features": True,
    # ---------------------------
    # 追加: 保存・ステータス可視化 / デバッグ
    # ---------------------------
    # setup() 直後に初期モデル/空リプレイを checkpoint_path へ保存するか
    "initial_checkpoint_on_setup": True,
    # 並行モード train_concurrent で一定秒ごとに events.log へ進捗とバッファ統計を出す間隔 (0/None で無効)
    "concurrent_status_log_sec": 300,
    # 追加ステータスログにメモリスナップショットも含めるか
    "status_log_include_memory": True,
    # 追加詳細デバッグ (Queue drain / train trigger) を標準出力へ都度出すフラグ
    "concurrent_debug_logging": False,

    # ---------------------------
    # メモリ最適化フラグ
    # ---------------------------
    # True の場合 legal_actions の元リストを各サンプルに残す (学習時の再構築コスト回避用)。
    # False なら ID 化された legal_ids のみ保持しメモリ削減 (推奨)。
    # drl_agent._store_sample 内で参照。
    "enable_legal_actions_backup": False,
    # True なら量子化 pi_q がある時は生の pi を捨てる (デフォルト: True)。
    # False で raw pi も保持 (デバッグ/分析用途)。格納時は float16 へ変換しメモリ節約。
    "drop_raw_pi": True,
    # train_step の指標を events.log に行単位で書き出す頻度 (update 毎)。0/None で無効。
    "events_log_train_every": 200,
    # フル特徴量モデルで full_input / full_compact を欠いたサンプル (ゼロパディング対象) を学習から除外するか
    # True: train_step でスキップ (推奨) / False: ゼロベクトルで学習に含める
    "skip_zero_padded_full_samples": True,

    # ---------------------------
    # 重複サンプルフィルタ設定
    # ---------------------------
    # True で有効化: _store_sample で (policy top1, value_pred(量子化), legal_ids数) 等から
    # 簡易シグネチャを作り直近ウィンドウ内の過剰出現(>duplicate_signature_max_count)時に破棄。
    # 情報量の低い連続同型局面の氾濫を抑制しメモリ圧縮と多様性向上を狙う。
    "enable_duplicate_filter": True,
    # シグネチャの保持ウィンドウサイズ (FIFO)。大きくし過ぎると計数コスト増。
    "duplicate_window_size": 3000,
    # 同一シグネチャを許容する最大回数 (この回数を超えると以降スキップ)。
    "duplicate_signature_max_count": 200,
    # スキップ数を一定間隔でログ出力するか (0/None で無効, >0 でその間隔毎に [dup] 行)。
    "duplicate_log_interval": 0,
    # フィルタのシグネチャ構成: 'top_value_len' で (policy_top_idx, value_u8, legal_count)
    # 将来拡張ポイント (例: 'hash_pi')。
    "duplicate_signature_type": "top_value_len",

    # ---------------------------
    # 追加: チェックポイント保存時にリプレイを purge するオプション
    # ---------------------------
    # True: _save_checkpoint 内で replay を save(purge=True) しメモリ解放。
    # False: 保存後もメモリに残す (従来挙動)。
    "purge_replay_after_checkpoint": False,
    # True: train_concurrent / train_updates の各 train_step 後に即座に checkpoint 保存を行い
    #       (purge_replay_after_checkpoint が True なら) リプレイを空にする超省メモリ運用。
    # False: まとまったバースト後に保存 (推奨)。
    "purge_replay_after_each_update": False,
    # ---------------------------
    # 自動メモリベース High/Low Water リプレイ制御 
    # ---------------------------------------------------------------------
    # True で有効化: プロセス RSS が high 基準を超過したら low 目標付近になるまで
    # 古いサンプルを間引く (FIFO)。purge_replay_after_each_update が True の場合は
    # そもそも巨大化しないため自動制御は実質発火しない想定。
    "auto_replay_water_enabled": True,
    # 物理メモリ(RAM)に対する RSS 高水位比率。0<high<1。
    "replay_memory_high_ratio": 0.85,
    # 低水位ターゲット比率。削減後はおおむねこの比率以下になるまで削る。
    "replay_memory_low_ratio": 0.35,
    
    # 親+子RSS合計が15GBを超えたら必ず purge したい要求に合わせ、デフォルトを 15360 MB に設定。
    # 0 のままにしたい場合はユーザ側で override してください。
    # ※ 既存キーを上書きしないため、新たに high_abs_mb_override を用意し trainer 側で優先する実装でも可。
    # ここではシンプルに既存値を直接 15360 に変更する。
    "replay_memory_high_abs_mb": 2000,   # メモリ削減: 14GB→6GB で発火
    # 低水位を絶対MBで指定したい場合のオプション (0/None で無効)。
    # high_abs_mb 発火時、low_abs_mb > 0 なら ratio計算の代わりに low_abs_mb へ近づくよう target_size を計算。
    "replay_memory_low_abs_mb": 500,    # メモリ削減: 5GB→3GB
    # 再発火クールダウン(秒)。直近 purge からこの秒数は再度メモリ水位判定をスキップ。
    "replay_memory_cooldown_sec": 300,
        # 親+子プロセスRSS合算で判定するか (並列 self-play 時に必須)
        "replay_memory_include_children": False,
        # 子RSS再計算の最短間隔(秒) (頻繁すぎる psutil 呼び出しを抑制)
        "replay_memory_children_recalc_sec": 15,
        # purge 後に gc.collect() を実行して RSS 解放を促すか
        "replay_memory_force_gc": True,
        # purge 詳細ログ (判定毎の mem_check / before/after) を出すか
    "replay_memory_debug_log": False,  # 一時的に mem_check デバッグ出力を無効化
        # purge ログで after_delete と GC 後の2段階を出すか
        "replay_memory_log_before_after": True,
        # コンパクト化モード: none|rebuild (rebuild で残存要素を新しい deque に詰め替え断片化軽減)
        "replay_memory_compact_mode": "rebuild",
        # 緊急高水位 (通常 high_ratio を更に越えた場合) 0/None で無効
        "replay_memory_emergency_ratio": 0.0,
        # 緊急時 target_size を さらに * factor で深く削る (0<factor<1)
        "replay_memory_aggressive_factor": 0.8,
        # 発火時最低削除件数 (微小削減ノイズ抑制)。0/None で無効
        "replay_memory_min_purge_rows": 0,

    # ---------------------------
    # Worker RSS 再起動 (最小安全版)
    # ---------------------------
    # 有効化スイッチ
    "worker_restart_enable": True,
    # 1ワーカーRSS(MB)がこの高水位閾値を連続観測回数分超えたら graceful 再起動要求
    "worker_restart_rss_high_mb":1700,
    # 再起動判定用ヒステリシス下限 (未使用: 今回は単純連続超過のみ、将来拡張用)
    "worker_restart_rss_low_mb": 1200,
    # 閾値超過を何回連続観測したら発火するか
    "worker_restart_consecutive_required": 3,
    # 同一ワーカーの再起動間隔(秒) 下回る場合は保留 (スラッシング防止)
    "worker_restart_min_interval_sec": 1200,
    # 再起動要求時に付与するジッター最大秒数 (0 で無効)。worker_id を元に安定ジッター。
    "worker_restart_jitter_sec": 30,
    # 緊急全体RSS閾値 (MB)。0/None で無効。超過時は最大RSSワーカー即 graceful 要求 (将来kill拡張余地)。
    "worker_restart_emergency_total_mb": 20000,
    # graceful 再起動でエピソード終了待ちする最大秒数
    "worker_restart_grace_timeout_sec": 180,
    # grace timeout 超過後の強制 kill までの猶予秒数
    "worker_restart_force_kill_sec": 240,
    # flush (サンプル送信) 完了待ちタイムアウト
    "worker_restart_flush_timeout_sec": 20,
    # 終了直前に最終 objtypes ログを送るか
    "worker_restart_log_object_types_on_exit": False,
    # シード戦略 (base+wid+gen を文字列管理。実装側で解釈) 今回は informational
    "worker_restart_seed_strategy": "base+wid+gen",

    # ---------------------------
    # ---------------------------
    # 追加メモリ削減オプション
    # ---------------------------
    # ワーカープロセスでローカルリプレイバッファを持たない（Queue へ即座に送信）
    "worker_zero_buffer": True,
    # エピソード終了時に即座に行動履歴をクリア
    "clear_action_history_per_episode": True,
    # Phase サンプル確定後に即座にリストをクリア
    "aggressive_phase_clear": True,
    
    # ---------------------------
    # 非同期 I/O (torch.save / joblib.dump を専用プロセスへオフロード)
    # ---------------------------
    # True で有効化。モデル / リプレイ保存のディスク書込みをメイン計算ループから分離し I/O 待ちでの停滞を軽減。
    "enable_async_io": True,
    # I/O ワーカーのキュー最大長 (溢れた場合は同期フォールバックし警告)
    "async_io_queue_maxsize": 32,
    # 終了時に全ジョブ完了を待つ最大秒数 (0/None で待たない)
    "async_io_flush_timeout_sec": 30,
    # True で queue full 時に 1 度だけ WARN ログ
    "async_io_warn_queue_full": True,
    # True で I/O ジョブ完了時に簡易イベントログ ([async-io] ...) を events.log へ (低頻度デバッグ用)
    "async_io_log_events": False,
}

__all__ = ["ALPHA_ZERO_CONFIG"]
