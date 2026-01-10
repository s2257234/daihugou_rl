import os, json, csv, time, atexit
import threading
import queue as _queue
from typing import Dict, Any, Optional
import glob
import joblib
import numpy as np

class TrainingLogger:
    """Collects and writes training / episode / MCTS root statistics.

    Outputs:
      logs/train_updates.csv   : per train_step aggregate losses & accuracies
      logs/episodes.csv        : per episode aggregate performance
      logs/mcts_samples.jsonl  : sampled MCTS root statistics per move
      TensorBoard (logs/tb)    : scalar summaries (optional)
    """
    def __init__(self, log_dir: str = "logs", use_tensorboard: bool = True, clear_existing: bool = False, log_mcts_samples: bool = True,
                 config: dict | None = None):
        self.log_dir = log_dir
        if clear_existing and os.path.exists(log_dir):
            try:
                import shutil
                for name in os.listdir(log_dir):
                    if '.tmp.' in name:
                        continue
                    p = os.path.join(log_dir, name)
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            except Exception:
                pass
        os.makedirs(log_dir, exist_ok=True)
        self.use_tb = use_tensorboard
        self.tb = None
        self._tb_disabled_reason: Optional[str] = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                tb_dir = os.path.join(log_dir, "tb")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=tb_dir)
                # 初期イベント: 起動確認 (step=0)
                try:
                    self.tb.add_text("meta/info", f"logger_initialized_at={time.strftime('%Y-%m-%d %H:%M:%S')}" , 0)
                    self.tb.add_scalar("meta/initialized", 1, 0)
                    self.tb.flush()
                    #print(f"[TrainingLogger] TensorBoard 有効: イベント出力先 = {tb_dir}")
                except Exception:
                    pass
            except Exception:
                self.tb = None
                # 失敗理由をファイルに残す (例: tensorboard 未インストール)
                try:
                    with open(os.path.join(log_dir, "tensorboard_disabled.txt"), "w", encoding="utf-8") as f:
                        f.write("TensorBoard SummaryWriter 初期化失敗。pip install tensorboard が必要な可能性があります。\n")
                    print("[TrainingLogger] TensorBoard 初期化失敗。'logs/tensorboard_disabled.txt' を確認してください。")
                except Exception:
                    pass
        # MCTS サンプル記録可否
        self.log_mcts_samples = log_mcts_samples
        # files
        self.train_csv = os.path.join(log_dir, "train_updates.csv")
        self.episode_csv = os.path.join(log_dir, "episodes.csv")
        self.mcts_jsonl = os.path.join(log_dir, "mcts_samples.jsonl")
        # コンフィグ由来: ログ頻度制御 (存在しなければデフォルト)
        self.cfg = config or {}
        self.disable_csv = bool(self.cfg.get("disable_csv_logging", False))
        self.csv_summary_only = bool(self.cfg.get("csv_summary_only", False)) and not self.disable_csv
        self.tb_train_every = max(1, int(self.cfg.get("tensorboard_train_log_every", 1) or 1))
        self.tb_ep_every = max(1, int(self.cfg.get("tensorboard_episode_log_every", 1) or 1))
        self.tb_flush_seconds = float(self.cfg.get("tensorboard_flush_seconds", 0) or 0)
        self.csv_train_every = max(1, int(self.cfg.get("csv_train_log_every", 1) or 1))
        self.csv_episode_every = max(1, int(self.cfg.get("csv_episode_log_every", 1) or 1))
        self.mcts_jsonl_max_bytes = int(self.cfg.get("mcts_jsonl_max_bytes", 0) or 0)
        self._last_tb_flush_time = time.time()
        self._mcts_size_capped = False
        # --- バッファリング設定 (新規) ---
        self._buffer_enabled = bool(self.cfg.get("log_buffer_enabled", True))
        # CSV 共通: レコード数閾値 / 時間間隔
        self._buf_flush_interval = float(self.cfg.get("log_buffer_flush_interval_sec", 30.0) or 5.0)
        self._buf_max_records = int(self.cfg.get("log_buffer_max_records", 64) or 64)
        # テキスト(events.log) 用: 行数閾値 / 時間間隔
        self._text_buf_max_lines = int(self.cfg.get("log_text_buffer_max_lines", 20) or 200)
        self._text_buf_flush_interval = float(self.cfg.get("log_text_flush_interval_sec", 30.0) or 5.0)

        # 内部バッファ
        self._train_buf = []  # list[list]
        self._episode_buf = []
        self._text_buf = []  # list[str]
        # 検証ロス一時保持（train_updates.csvへ同梱/即時書込の両方で使用）
        # 最終フラッシュ時刻
        now_ts = time.time()
        self._last_train_flush = now_ts
        self._last_episode_flush = now_ts
        self._last_text_flush = now_ts
        self._val_losses_by_step = {}
        # 直近の検証ロス（毎トレイン行に同梱するためのフォールバック）
        self._last_val_pair = (None, None)
        # 検証手札予測損失（hand_pred_loss）最新値
        self._last_val_hand = None
        # 検証手札予測再現率（recall）最新値
        self._last_val_hand_recall = None

        # headers (既存挙動維持: ただし即時ファイル生成はバッファ有効時も保持)
        if not self.disable_csv and not self.csv_summary_only:
            if not os.path.exists(self.train_csv):
                with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "update_step","policy_loss","value_loss","hand_pred_loss","entropy","train_count",
                        "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","hand_label_pos_rate","samples",
                        # 検証ロス: policy/value/hand
                        "val_policy_loss","val_value_loss","val_hand_pred_loss","val_hand_recall",
                        # ValueTarget statistics (computed from latest selfplay joblib)
                        "vt_count","vt_mean","vt_std","vt_min","vt_max","vt_clip_neg","vt_clip_pos"
                    ])
            # 既存行の有無を記録（初回1行は必ず出すための判定に利用）
            try:
                self._train_rows_written = 0
                if os.path.exists(self.train_csv):
                    with open(self.train_csv, 'r', encoding='utf-8') as rf:
                        # ヘッダを除くデータ行数を概算
                        self._train_rows_written = max(0, sum(1 for _ in rf) - 1)
            except Exception:
                self._train_rows_written = 0
            if not os.path.exists(self.episode_csv):
                with open(self.episode_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "episode","avg_rank","first_rate","episode_len","phase_acc",
                        "phase_win_rate","phase_wins","phase_attempts","cum_phase_win_rate"
                    ])
        # summary only 用に最後の値を保持
        self._last_train_metrics = None  # type: ignore[assignment]
        self._last_episode_metrics = None  # type: ignore[assignment]
        # config スナップショット (一度だけ) 既に存在しなければ
        self._config_dump_path = os.path.join(log_dir, "config_snapshot.json")
        self._config_written = False
        self.update_step = 0
        self.episode_idx = 0
        # ディスクフル検知フラグ
        self._disk_full = False
        self._disk_full_reason = None
        # メモリスナップショット制御
        self._last_mem_log_time = 0.0

        # --- 非同期テキスト追記（[perf-ep]向け） ---
        # 設定: async_text_enabled=True なら有効化し、[perf-ep]行は即時に専用スレッドで events.log に追記する
        # バッファフラッシュの間隔待ちを避け、可視性を高める目的
        self._async_text_enabled = bool(self.cfg.get('async_text_enabled', True))
        self._async_text_perf_only = bool(self.cfg.get('async_text_perf_only', True))  # True の場合、[perf-ep] のみ対象
        self._async_text_q = None
        self._async_text_thread = None
        self._async_text_stop = False
        if self._async_text_enabled:
            try:
                self._async_text_q = _queue.Queue(maxsize=int(self.cfg.get('async_text_queue_size', 2048) or 2048))
                self._async_text_thread = threading.Thread(target=self._async_text_writer_loop, name="TrainingLoggerAsyncText", daemon=True)
                self._async_text_thread.start()
            except Exception:
                self._async_text_q = None
                self._async_text_thread = None

        # atexit で強制 flush (プロセス終了前に残バッファを吐き出す)
        try:
            atexit.register(self._atexit_flush)
        except Exception:
            pass

        # バッファリング有効時でも events.log をすぐに見えるように空ファイルを作成
        # (ファイルがないと「ログが出ていない」と誤解されやすいため)
        try:
            if self._buffer_enabled:
                ev = os.path.join(self.log_dir, 'events.log')
                if not os.path.exists(ev):
                    os.makedirs(self.log_dir, exist_ok=True)
                    with open(ev, 'a', encoding='utf-8'):
                        pass
        except Exception:
            pass

        # バッファリング有効時はバックグラウンドで周期 flush
        # (log_text が低頻度でも一定秒で events.log / CSV が更新されるようにする)
        self._bg_stop = False
        self._bg_thread = None
        if self._buffer_enabled:
            try:
                self._bg_thread = threading.Thread(target=self._bg_flush_loop, name="TrainingLoggerFlush", daemon=True)
                self._bg_thread.start()
            except Exception:
                self._bg_thread = None

    # ---------------- Runtime config update (new) ----------------
    def update_config(self, new_cfg: dict | None = None):
        """ランタイムでログ関連設定を更新するためのヘルパー。

        既存の logger インスタンス生成後に config 辞書を変更しても反映されない問題を解消します。
        必要に応じて trainer 側で設定値を変更した直後に呼び出してください。

        更新対象:
          - disable_csv_logging / csv_summary_only
          - csv_train_log_every / csv_episode_log_every
          - tensorboard_train_log_every / tensorboard_episode_log_every
          - log_buffer_enabled および flush 間隔関連
        """
        if not isinstance(new_cfg, dict):
            return
        # マージ
        try:
            self.cfg.update(new_cfg)
        except Exception:
            pass
        # 基本フラグ
        self.disable_csv = bool(self.cfg.get("disable_csv_logging", self.disable_csv))
        self.csv_summary_only = bool(self.cfg.get("csv_summary_only", self.csv_summary_only)) and not self.disable_csv
        # 頻度 (1 以上)
        try:
            self.csv_train_every = max(1, int(self.cfg.get("csv_train_log_every", self.csv_train_every) or 1))
        except Exception:
            pass
        try:
            self.csv_episode_every = max(1, int(self.cfg.get("csv_episode_log_every", self.csv_episode_every) or 1))
        except Exception:
            pass
        try:
            self.tb_train_every = max(1, int(self.cfg.get("tensorboard_train_log_every", self.tb_train_every) or 1))
        except Exception:
            pass
        try:
            self.tb_ep_every = max(1, int(self.cfg.get("tensorboard_episode_log_every", self.tb_ep_every) or 1))
        except Exception:
            pass
        # バッファリング関連
        try:
            self._buffer_enabled = bool(self.cfg.get("log_buffer_enabled", self._buffer_enabled))
        except Exception:
            pass
        try:
            self._buf_flush_interval = float(self.cfg.get("log_buffer_flush_interval_sec", self._buf_flush_interval) or self._buf_flush_interval)
        except Exception:
            pass
        try:
            self._buf_max_records = int(self.cfg.get("log_buffer_max_records", self._buf_max_records) or self._buf_max_records)
        except Exception:
            pass
        try:
            self._text_buf_max_lines = int(self.cfg.get("log_text_buffer_max_lines", self._text_buf_max_lines) or self._text_buf_max_lines)
        except Exception:
            pass
        try:
            self._text_buf_flush_interval = float(self.cfg.get("log_text_flush_interval_sec", self._text_buf_flush_interval) or self._text_buf_flush_interval)
        except Exception:
            pass
        # 非同期テキスト設定 (キュー再生成は避ける。必要なら再初期化ロジックを別途実装)
        try:
            self._async_text_perf_only = bool(self.cfg.get('async_text_perf_only', self._async_text_perf_only))
        except Exception:
            pass
        # 必要に応じてヘッダ再生成 (頻度のみの変更では不要)
        if not self.disable_csv and not self.csv_summary_only:
            try:
                if not os.path.exists(self.train_csv):
                    with open(self.train_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "update_step","policy_loss","value_loss","hand_pred_loss","entropy","train_count",
                            "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","hand_label_pos_rate","samples",
                            "val_policy_loss","val_value_loss","val_hand_pred_loss","val_hand_recall",
                            "vt_count","vt_mean","vt_std","vt_min","vt_max","vt_clip_neg","vt_clip_pos"
                        ])
            except Exception:
                pass

    # ---------------- Memory snapshot ----------------
    def log_memory_snapshot(self, sample_count: int | None = None, force: bool = False):
        """(無効化) 以前は RSS / per-sample メモリをログに出力していたが要求により出力停止。"""
        return

    # ---------------- Internal helpers ----------------
    def _disable_tensorboard(self, reason: str):
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
        self.tb = None
        self._tb_disabled_reason = reason
        marker = os.path.join(self.log_dir, "tensorboard_disabled.txt")
        try:
            with open(marker, "a", encoding="utf-8") as f:
                f.write(f"DISABLED at step={self.update_step or self.episode_idx} reason={reason}\n")
        except Exception:
            pass
        print(f"[TrainingLogger] TensorBoard無効化: {reason}")

    def _mark_disk_full(self, exc: Exception, context: str):
        """ディスクフル(OSError:28)検知時にロギングを停止し以降静かにスキップ。

        context: 'train_csv','episode_csv','mcts','text','tensorboard' など呼び出し元識別
        """
        if not self._disk_full:
            self._disk_full = True
            self._disk_full_reason = f"{type(exc).__name__}:{exc}"
            marker = os.path.join(self.log_dir, "disk_full_disabled.txt")
            try:
                with open(marker, 'a', encoding='utf-8') as f:
                    f.write(f"DISABLED {time.strftime('%Y-%m-%d %H:%M:%S')} ctx={context} reason={self._disk_full_reason}\n")
            except Exception:
                pass
            print(f"[TrainingLogger] ディスク容量不足検知 (ctx={context}) -> 以降のログ出力を停止: {self._disk_full_reason}")
        # TensorBoard も停止
        if self.tb is not None:
            self._disable_tensorboard(f"disk_full({context})")

    # ---------------- Train metrics ----------------
    def log_train(self, metrics: Dict[str, Any]):
        if self._disk_full:
            return
        # 検証ロスを取得してからupdate_stepをインクリメント
        # これにより、log_validationが保存した値（update_step + 1）を正しく取得できる
        current_step_before_inc = self.update_step
        self.update_step += 1
        # 旧バージョンの train_updates.csv に追記しようとして列不足になるケースを検出し再生成 (一度だけ)
        try:
            if os.path.exists(self.train_csv):
                with open(self.train_csv, 'r', encoding='utf-8') as rf:
                    header_line = rf.readline().strip()
                # 新仕様: total_loss を廃止し、train_count 列を追加
                needs_rebuild = False
                if ('cum_pos_rate' not in header_line) or ('total_loss' in header_line) or ('train_count' not in header_line):
                    needs_rebuild = True
                # 新規列: val_policy_loss / val_value_loss が無ければ再生成
                if ('val_policy_loss' not in header_line) or ('val_value_loss' not in header_line):
                    needs_rebuild = True
                # 新規列: hand_pred_loss / val_hand_pred_loss が無ければ再生成
                if ('hand_pred_loss' not in header_line) or ('val_hand_pred_loss' not in header_line):
                    needs_rebuild = True
                # 新規列: hand_label_pos_rate が無ければ再生成
                if ('hand_label_pos_rate' not in header_line):
                    needs_rebuild = True
                # 新規列: vt_mean 等の ValueTarget 列が無ければ再生成
                if ('vt_mean' not in header_line) or ('vt_count' not in header_line):
                    needs_rebuild = True
                if needs_rebuild:
                    # バックアップして新ヘッダで再生成
                    bak = self.train_csv + '.bak'
                    if not os.path.exists(bak):
                        os.replace(self.train_csv, bak)
                        with open(self.train_csv, 'w', newline='', encoding='utf-8') as wf:
                            writer = csv.writer(wf)
                            writer.writerow([
                                  "update_step","policy_loss","value_loss","hand_pred_loss","entropy","train_count",
                                "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","hand_label_pos_rate","samples",
                                "val_policy_loss","val_value_loss","val_hand_pred_loss","val_hand_recall",
                                "vt_count","vt_mean","vt_std","vt_min","vt_max","vt_clip_neg","vt_clip_pos"
                            ])
        except Exception:
            pass
        # 初回に config スナップショットを保存 (遅延書き込み)
        if not self._config_written and isinstance(metrics, dict) and 'config' in metrics:
            try:
                with open(self._config_dump_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics['config'], f, ensure_ascii=False, indent=2)
                self._config_written = True
            except Exception:
                pass
        # CSV 書き込み間引き
        if self.csv_summary_only:
            self._last_train_metrics = dict(metrics)
        # 初回1行は必ず出す: 既存データ行が0の場合は間引き条件を無視
        should_write = (not self.disable_csv) and (not self.csv_summary_only) and (
            (self.update_step % self.csv_train_every) == 0 or getattr(self, '_train_rows_written', 0) == 0
        )
        if should_write:
            # 毎トレイン行に現在のステップに対応する検証ロスを必ず同梱（検証が未実施なら None）
            # 優先順位: 1) metricsに直接含まれている値、2) _val_losses_by_stepから取得、3) _last_val_pair
            # これにより、log_trainが呼ばれる直前に計算された検証ロスを確実に使用できる
            vpl = metrics.get('val_policy_loss')
            vvl = metrics.get('val_value_loss')
            vhl = metrics.get('val_hand_pred_loss')
            vhr = metrics.get('val_hand_recall')
            
            # metricsに含まれていない場合、_val_losses_by_stepから取得
            if vpl is None and vvl is None:
                vpl, vvl = self._val_losses_by_step.get(self.update_step, self._last_val_pair)
            # hand_pred_lossとhand_recallがmetricsに含まれていない場合、最新値を使用
            if vhl is None:
                vhl = self._last_val_hand
            if vhr is None:
                vhr = self._last_val_hand_recall

            # train_count を常に update_step と一致させる（直列モードでの二重カウント差異解消）
            row = [
                self.update_step,                    # update_step (cumulative)
                metrics.get("policy_loss"),
                metrics.get("value_loss"),
                metrics.get("hand_pred_loss"),
                metrics.get("entropy"),
                self.update_step,                    # train_count forced to update_step
                metrics.get("value_acc"),
                metrics.get("value_brier"),
                metrics.get("policy_kl"),
                metrics.get("policy_top1_match"),
                metrics.get("pos_rate"),
                metrics.get("cum_pos_rate"),
                metrics.get("hand_label_pos_rate"),
                metrics.get("samples"),
                vpl, vvl, vhl, vhr,
            ]
            # ValueTarget stats (computed from latest selfplay joblib)
            try:
                vt_stats = self._compute_value_target_stats()
            except Exception:
                vt_stats = {}
            row.extend([
                vt_stats.get('vt_count'), vt_stats.get('vt_mean'), vt_stats.get('vt_std'),
                vt_stats.get('vt_min'), vt_stats.get('vt_max'), vt_stats.get('vt_clip_neg'), vt_stats.get('vt_clip_pos')
            ])
            # 直接追記 (頻度1や小間隔で確実に行が出るようにする) + バッファは補助的に使用
            try:
                with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                self._train_rows_written = getattr(self, '_train_rows_written', 0) + 1
            except OSError as e:
                self._mark_disk_full(e, 'train_csv')
                return
            except Exception:
                pass
            # 直接追記を採用するため、ここでは _train_buf へは追加しない（重複flush防止）
        else:
            # デバッグ: 書き込みスキップ理由 (任意フラグ train_csv_debug が True のとき出力)
            if bool(self.cfg.get('train_csv_debug', False)):
                try:
                    self.log_text(f"[train-csv-skip] step={self.update_step} every={self.csv_train_every} rows={getattr(self,'_train_rows_written',None)} summary={self.csv_summary_only}")
                except Exception:
                    pass
        # TensorBoard 間引き
        if self.tb and not self._disk_full and (self.update_step % self.tb_train_every) == 0:
            try:
                for k,v in metrics.items():
                    if isinstance(v,(int,float)):
                        self.tb.add_scalar(f"train/{k}", v, self.update_step)
                # flush ポリシー: 一定秒数経過 or 強制
                now = time.time()
                if self.tb_flush_seconds <= 0 or (now - self._last_tb_flush_time) >= self.tb_flush_seconds:
                    try:
                        self.tb.flush()
                    except Exception:
                        pass
                    self._last_tb_flush_time = now
            except OSError as e:  # Disk full 等
                self._disable_tensorboard(f"OSError:{e}")
            except Exception as e:  # その他は一度警告し続行
                print(f"[TrainingLogger] TensorBoard書き込み失敗 (train): {e}")

    # ---------------- Validation metrics ----------------
    def log_validation(self, metrics: Dict[str, Any]):
        """検証ロスを train_updates.csv へ記録。

        - 対応する update_step をキーに保持し、次回の学習行に同梱して出力する。
        - これにより、val行の単独追記（重複行）を廃止し、学習行とまとめて一行で記録する。
        - TensorBoard への 'val/*' 出力は継続。
        """
        if self._disk_full:
            return
        # 注意: log_trainは最初にself.update_step += 1を実行するため、
        # log_trainが呼ばれる直前にlog_validationが呼ばれる場合、
        # log_trainでインクリメント後のupdate_stepで取得できるようにする
        # つまり、現在のupdate_step + 1で保存する
        # ただし、log_trainが呼ばれる直前にlog_validationが呼ばれることを前提とする
        step = self.update_step + 1  # 次の学習ステップ番号で整列（log_trainでインクリメント後の値）
        # 検証ロスを保持（lossのみ）
        try:
            vp = metrics.get("policy_loss")
            vv = metrics.get("value_loss")
            vh = metrics.get("hand_pred_loss")
            vhr = metrics.get("hand_recall")
            self._val_losses_by_step[step] = (vp, vv)
            # 直近値を更新（フォールバック用）
            self._last_val_pair = (vp, vv)
            self._last_val_hand = vh
            self._last_val_hand_recall = vhr
        except Exception:
            pass
        # 分離した検証専用行は廃止し、次回以降の train 行に同梱するのみ
        # TensorBoard
        if self.tb and not self._disk_full:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.tb.add_scalar(f"val/{k}", v, step)
                now = time.time()
                if self.tb_flush_seconds <= 0 or (now - self._last_tb_flush_time) >= self.tb_flush_seconds:
                    try:
                        self.tb.flush()
                    except Exception:
                        pass
                    self._last_tb_flush_time = now
            except OSError as e:
                self._disable_tensorboard(f"OSError:{e}")
            except Exception as e:
                print(f"[TrainingLogger] TensorBoard書き込み失敗 (validation): {e}")

        # Fallback: If no train rows exist yet (e.g. resume_skip_prevalidation=True was used and
        # pre-train validation was skipped), emit a placeholder train CSV row now so that the
        # validation columns are present in the CSV file. This preserves the previous behaviour
        # where one validation result is always visible in `train_updates.csv` even before any
        # training rows are written.
        try:
            if (not self._disk_full) and (not self.disable_csv) and (not self.csv_summary_only) and getattr(self, '_train_rows_written', 0) == 0:
                # Build a row with train-related columns set to None and val columns filled.
                # Header layout: update_step,policy_loss,value_loss,hand_pred_loss,entropy,train_count,
                # value_acc,value_brier,policy_kl,policy_top1_match,pos_rate,cum_pos_rate,hand_label_pos_rate,samples,
                # val_policy_loss,val_value_loss,val_hand_pred_loss,val_hand_recall
                vpl = metrics.get('policy_loss')
                vvl = metrics.get('value_loss')
                vhl = metrics.get('hand_pred_loss')
                vhr = metrics.get('hand_recall')
                row = [self.update_step] + [None] * 13 + [vpl, vvl, vhl, vhr]
                # append placeholder for vt columns
                row.extend([None, None, None, None, None, None, None])
                try:
                    with open(self.train_csv, 'a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow(row)
                    self._train_rows_written = getattr(self, '_train_rows_written', 0) + 1
                except OSError as e:
                    self._mark_disk_full(e, 'train_csv')
                except Exception:
                    pass
        except Exception:
            pass

    # ---------------- Episode metrics ----------------
    def log_episode(self, metrics: Dict[str, Any]):
        if self._disk_full:
            return
        self.episode_idx += 1
        # 旧ヘッダ互換処理 (一度だけ再生成)
        try:
            if os.path.exists(self.episode_csv):
                with open(self.episode_csv, 'r', encoding='utf-8') as rf:
                    header_line = rf.readline().strip()
                if 'phase_win_rate' not in header_line:
                    bak = self.episode_csv + '.bak'
                    if not os.path.exists(bak):
                        os.replace(self.episode_csv, bak)
                        with open(self.episode_csv, 'w', newline='', encoding='utf-8') as wf:
                            writer = csv.writer(wf)
                            writer.writerow([
                                "episode","avg_rank","first_rate","episode_len","phase_acc",
                                "phase_win_rate","phase_wins","phase_attempts","cum_phase_win_rate"
                            ])
        except Exception:
            pass
        if self.csv_summary_only:
            self._last_episode_metrics = dict(metrics)
        if (not self.disable_csv) and (not self.csv_summary_only) and (self.episode_idx % self.csv_episode_every) == 0:
            row = [
                self.episode_idx,
                metrics.get("avg_rank"),
                metrics.get("first_rate"),
                metrics.get("episode_len"),
                metrics.get("phase_acc"),
                metrics.get("phase_win_rate"),
                metrics.get("phase_wins"),
                metrics.get("phase_attempts"),
                metrics.get("cum_phase_win_rate"),
            ]
            if self._buffer_enabled:
                self._episode_buf.append(row)
                self._maybe_flush_episode()
            else:
                try:
                    with open(self.episode_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                except OSError as e:
                    self._mark_disk_full(e, 'episode_csv')
                    return
        if self.tb and not self._disk_full and (self.episode_idx % self.tb_ep_every) == 0:
            try:
                for k,v in metrics.items():
                    if isinstance(v,(int,float)):
                        self.tb.add_scalar(f"episode/{k}", v, self.episode_idx)
                now = time.time()
                if self.tb_flush_seconds <= 0 or (now - self._last_tb_flush_time) >= self.tb_flush_seconds:
                    try:
                        self.tb.flush()
                    except Exception:
                        pass
                    self._last_tb_flush_time = now
            except OSError as e:
                self._disable_tensorboard(f"OSError:{e}")
            except Exception as e:
                print(f"[TrainingLogger] TensorBoard書き込み失敗 (episode): {e}")

    # ---------------- MCTS root sample ----------------
    def log_mcts_sample(self, data: Dict[str, Any]):
        if self._disk_full or not self.log_mcts_samples:
            return
        if self._mcts_size_capped:
            return
        try:
            # サイズ上限 (バイト) チェック
            if self.mcts_jsonl_max_bytes and os.path.exists(self.mcts_jsonl):
                if os.path.getsize(self.mcts_jsonl) >= self.mcts_jsonl_max_bytes:
                    self._mcts_size_capped = True
                    self.log_text(f"[logger] mcts_samples.jsonl size cap reached -> stop appending ({self.mcts_jsonl_max_bytes} bytes)")
                    return
            with open(self.mcts_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except OSError as e:
            self._mark_disk_full(e, 'mcts')

    # ---------------- Free-form text logging ----------------
    def log_text(self, text: str, filename: str = "events.log", also_print: bool = False):
        """任意のテキストをログに追記し、可能ならTensorBoardにも出力する。

        - logs/events.log にタイムスタンプ付きで追記
        - TensorBoard が有効な場合は add_text で記録（step は update_step または episode_idx）
        - also_print=True の場合は標準出力にも表示
        """
        if not self._disk_full:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            line = f"[{ts}] {text}\n"
            # [perf-ep] を優先的に非同期追記
            if self._async_text_enabled and (not self._async_text_perf_only or (isinstance(text, str) and text.startswith('[perf-ep]'))):
                put_ok = False
                try:
                    if self._async_text_q is not None:
                        self._async_text_q.put_nowait(line)
                        put_ok = True
                except Exception:
                    put_ok = False
                if put_ok:
                    if also_print:
                        print(line.strip())
                    # 非同期キューへ載せたらここで終了（通常バッファには載せない）
                    pass
                else:
                    # キューに載せられない場合のみ従来経路（バッファ or 直書き）でフォールバック
                    if self._buffer_enabled:
                        self._text_buf.append(line)
                        if also_print:
                            print(line.strip())
                        self._maybe_flush_text()
                    else:
                        try:
                            path = os.path.join(self.log_dir, filename)
                            os.makedirs(self.log_dir, exist_ok=True)
                            with open(path, 'a', encoding='utf-8') as f:
                                f.write(line)
                            if also_print:
                                print(line.strip())
                        except OSError as e:
                            self._mark_disk_full(e, 'text')
                        except Exception:
                            pass
            else:
                # 従来経路
                if self._buffer_enabled:
                    self._text_buf.append(line)
                    if also_print:
                        print(line.strip())
                    self._maybe_flush_text()
                else:
                    try:
                        path = os.path.join(self.log_dir, filename)
                        os.makedirs(self.log_dir, exist_ok=True)
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(line)
                        if also_print:
                            print(line.strip())
                    except OSError as e:
                        self._mark_disk_full(e, 'text')
                    except Exception:
                        pass

        if self.tb and not self._disk_full:
            try:
                step = self.update_step or self.episode_idx or 0
                self.tb.add_text('misc/text', text, step)
                try:
                    self.tb.flush()
                except Exception:
                    pass
            except OSError as e:
                self._disable_tensorboard(f"OSError:{e}")
            except Exception:
                # TensorBoard へのテキスト出力失敗は無視
                pass

    # ---------------- Summary write at end ----------------
    def write_csv_summaries(self):
        """csv_summary_only=True の場合に最後の1行だけを書き出す。既存ファイルが無ければヘッダを新規作成。
        disable_csv=True の場合は何もしない。"""
        if self.disable_csv or not self.csv_summary_only:
            return
        try:
            if self._last_train_metrics:
                # バッファ内に未flush列があれば先に出す
                if self._buffer_enabled and self._train_buf:
                    self._flush_train(force=True)
                if not os.path.exists(self.train_csv):
                    with open(self.train_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "update_step","policy_loss","value_loss","hand_pred_loss","entropy","train_count",
                            "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","hand_label_pos_rate","samples",
                            "val_policy_loss","val_value_loss","val_hand_pred_loss","val_hand_recall",
                            "vt_count","vt_mean","vt_std","vt_min","vt_max","vt_clip_neg","vt_clip_pos"
                        ])
                with open(self.train_csv, 'a', newline='', encoding='utf-8') as f:
                    m = self._last_train_metrics
                    writer = csv.writer(f)
                    # summary モードでも current → prev の順で検証ロスを拾う
                    vpl, vvl, vhl, vhr = None, None, None, None
                    try:
                        vp, vv = self._val_losses_by_step.get(self.update_step, (None, None))
                        # hand は pair とは別に保持しているので最新属性から取得
                        vh = self._last_val_hand
                        vhr = self._last_val_hand_recall
                        if (vp is None and vv is None) and self.update_step > 0:
                            vp_prev, vv_prev = self._val_losses_by_step.get(self.update_step - 1, (None, None))
                            if vp_prev is not None or vv_prev is not None:
                                vp, vv = vp_prev, vv_prev
                                try:
                                    self._val_losses_by_step.pop(self.update_step - 1, None)
                                except Exception:
                                    pass
                        else:
                            try:
                                self._val_losses_by_step.pop(self.update_step, None)
                            except Exception:
                                pass
                        vpl, vvl, vhl, vhr = vp, vv, vh, vhr
                    except Exception:
                        pass
                    try:
                        vt = self._compute_value_target_stats()
                    except Exception:
                        vt = {}
                    vt_vals = [
                        vt.get('vt_count'), vt.get('vt_mean'), vt.get('vt_std'), vt.get('vt_min'), vt.get('vt_max'), vt.get('vt_clip_neg'), vt.get('vt_clip_pos')
                    ]
                    writer.writerow([
                        self.update_step,
                        m.get("policy_loss"), m.get("value_loss"), m.get("hand_pred_loss"), m.get("entropy"), m.get("train_count", self.update_step),
                        m.get("value_acc"), m.get("value_brier"), m.get("policy_kl"), m.get("policy_top1_match"),
                        m.get("pos_rate"), m.get("cum_pos_rate"), m.get("hand_label_pos_rate"), m.get("samples"),
                        vpl, vvl, vhl, vhr
                    ] + vt_vals)
            if self._last_episode_metrics:
                if self._buffer_enabled and self._episode_buf:
                    self._flush_episode(force=True)
                if not os.path.exists(self.episode_csv):
                    with open(self.episode_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "episode","avg_rank","first_rate","episode_len","phase_acc",
                            "phase_win_rate","phase_wins","phase_attempts","cum_phase_win_rate"
                        ])
                with open(self.episode_csv, 'a', newline='', encoding='utf-8') as f:
                    m = self._last_episode_metrics
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_idx,
                        m.get("avg_rank"), m.get("first_rate"), m.get("episode_len"), m.get("phase_acc"),
                        m.get("phase_win_rate"), m.get("phase_wins"), m.get("phase_attempts"), m.get("cum_phase_win_rate")
                    ])
        except Exception as e:
            print(f"[TrainingLogger] write_csv_summaries failed: {e}")

    # --------------- バッファ flush 支援メソッド ---------------
    def _maybe_flush_train(self):
        if self._disk_full:
            self._train_buf.clear()
            return
        now = time.time()
        if (len(self._train_buf) >= self._buf_max_records) or ((now - self._last_train_flush) >= self._buf_flush_interval):
            self._flush_train()

    def _maybe_flush_episode(self):
        if self._disk_full:
            self._episode_buf.clear()
            return
        now = time.time()
        if (len(self._episode_buf) >= self._buf_max_records) or ((now - self._last_episode_flush) >= self._buf_flush_interval):
            self._flush_episode()

    def _maybe_flush_text(self):
        if self._disk_full:
            self._text_buf.clear()
            return
        now = time.time()
        if (len(self._text_buf) >= self._text_buf_max_lines) or ((now - self._last_text_flush) >= self._text_buf_flush_interval):
            self._flush_text()

    def _compute_value_target_stats(self):
        """Latest selfplay joblib から value target の統計を計算して返す。

        戻り値: dict keys = vt_count, vt_mean, vt_std, vt_min, vt_max, vt_clip_neg, vt_clip_pos
        """
        try:
            # data ディレクトリはプロジェクトルート直下の data/
            proj_root = os.path.abspath(os.path.join(self.log_dir, '..'))
            data_dir = os.path.join(proj_root, 'data')
            pattern = os.path.join(data_dir, 'selfplay_ep*.joblib')
            files = glob.glob(pattern)
            if not files:
                return {}
            latest = max(files, key=lambda p: os.path.getmtime(p))
            obj = joblib.load(latest)
            samples = []
            if isinstance(obj, dict):
                for k in ('samples', 'replay', 'data', 'episodes'):
                    if k in obj and isinstance(obj[k], (list, tuple)):
                        samples = list(obj[k]); break
                if not samples and ('pi_q' in obj or 'value' in obj):
                    samples = [obj]
            elif isinstance(obj, (list, tuple)):
                samples = list(obj)

            vals = []
            for s in samples:
                try:
                    if not isinstance(s, dict):
                        continue
                    v = s.get('value')
                    if v is None:
                        continue
                    vals.append(float(v))
                except Exception:
                    continue
            if len(vals) == 0:
                return {}
            a = np.asarray(vals, dtype=np.float64)
            vt_count = int(a.size)
            vt_mean = float(a.mean())
            vt_std = float(a.std())
            vt_min = float(a.min())
            vt_max = float(a.max())
            vt_clip_neg = int((a <= -0.9999).sum())
            vt_clip_pos = int((a >= 0.9999).sum())
            return {
                'vt_count': vt_count,
                'vt_mean': vt_mean,
                'vt_std': vt_std,
                'vt_min': vt_min,
                'vt_max': vt_max,
                'vt_clip_neg': vt_clip_neg,
                'vt_clip_pos': vt_clip_pos,
            }
        except Exception:
            return {}


    def _bg_flush_loop(self):  # pragma: no cover (タイマースレッド)
        try:
            while not getattr(self, "_bg_stop", False):
                if self._disk_full:
                    # ディスクフル時は以降書き込みを停止
                    break
                now = time.time()
                # CSV (train)
                try:
                    if self._train_buf and ((now - self._last_train_flush) >= self._buf_flush_interval):
                        self._flush_train()
                except Exception:
                    pass
                # CSV (episode)
                try:
                    if self._episode_buf and ((now - self._last_episode_flush) >= self._buf_flush_interval):
                        self._flush_episode()
                except Exception:
                    pass
                # Text (events.log)
                try:
                    if self._text_buf and ((now - self._last_text_flush) >= self._text_buf_flush_interval):
                        self._flush_text()
                except Exception:
                    pass
                time.sleep(0.5)
        except Exception:
            # 背景スレッドは静かに終了
            pass

    def _flush_train(self, force: bool = False):
        if not self._train_buf:
            return
        try:
            with open(self.train_csv, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                for r in self._train_buf:
                    w.writerow(r)
        except OSError as e:
            self._mark_disk_full(e, 'train_csv')
            self._train_buf.clear()
            return
        except Exception:
            # 失敗時バッファを残す (再flush機会) force なら破棄
            if force:
                self._train_buf.clear()
            return
        self._train_buf.clear()
        self._last_train_flush = time.time()

    def _flush_episode(self, force: bool = False):
        if not self._episode_buf:
            return
        try:
            with open(self.episode_csv, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                for r in self._episode_buf:
                    w.writerow(r)
        except OSError as e:
            self._mark_disk_full(e, 'episode_csv')
            self._episode_buf.clear()
            return
        except Exception:
            if force:
                self._episode_buf.clear()
            return
        self._episode_buf.clear()
        self._last_episode_flush = time.time()

    def _flush_text(self, force: bool = False):
        if not self._text_buf:
            return
        path = os.path.join(self.log_dir, 'events.log')
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.writelines(self._text_buf)
        except OSError as e:
            self._mark_disk_full(e, 'text')
            self._text_buf.clear()
            return
        except Exception:
            if force:
                self._text_buf.clear()
            return
        self._text_buf.clear()
        self._last_text_flush = time.time()


    def flush_buffers(self, force: bool = False):
        """外部コール用: すべてのバッファを即時 flush."""
        if not self._buffer_enabled:
            return
        self._flush_train(force=force)
        self._flush_episode(force=force)
        self._flush_text(force=force)

    def _atexit_flush(self):  # pragma: no cover (プロセス終了パス)
        try:
            # 非同期テキストスレッドを停止し、残キューを吐き出す
            try:
                self._async_text_stop = True
                if self._async_text_q is not None:
                    # センチネル投入
                    try:
                        self._async_text_q.put_nowait(None)  # type: ignore[arg-type]
                    except Exception:
                        pass
                if self._async_text_thread is not None:
                    self._async_text_thread.join(timeout=1.5)
            except Exception:
                pass
            self._bg_stop = True
            self.flush_buffers(force=True)
        except Exception:
            pass

    def __del__(self):  # pragma: no cover
        try:
            self.flush_buffers(force=True)
        except Exception:
            pass

    # ------------ 非同期テキスト書き込みループ ([perf-ep]用) ------------
    def _async_text_writer_loop(self):  # pragma: no cover
        path = os.path.join(self.log_dir, 'events.log')
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass
        q = self._async_text_q
        if q is None:
            return
        try:
            while not self._async_text_stop:
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    continue
                if item is None:
                    break
                try:
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write(item)
                except OSError as e:
                    self._mark_disk_full(e, 'text')
                except Exception:
                    pass
        except Exception:
            # 静かに終了
            pass

__all__ = ["TrainingLogger"]
