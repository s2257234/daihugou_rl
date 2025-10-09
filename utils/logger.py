import os, json, csv, time, atexit
from typing import Dict, Any, Optional

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
        self._buf_flush_interval = float(self.cfg.get("log_buffer_flush_interval_sec", 5.0) or 5.0)
        self._buf_max_records = int(self.cfg.get("log_buffer_max_records", 64) or 64)
        # テキスト(events.log) 用: 行数閾値 / 時間間隔
        self._text_buf_max_lines = int(self.cfg.get("log_text_buffer_max_lines", 200) or 200)
        self._text_buf_flush_interval = float(self.cfg.get("log_text_flush_interval_sec", 5.0) or 5.0)

        # 内部バッファ
        self._train_buf = []  # list[list]
        self._episode_buf = []
        self._text_buf = []  # list[str]
        # 最終フラッシュ時刻
        now_ts = time.time()
        self._last_train_flush = now_ts
        self._last_episode_flush = now_ts
        self._last_text_flush = now_ts

        # headers (既存挙動維持: ただし即時ファイル生成はバッファ有効時も保持)
        if not self.disable_csv and not self.csv_summary_only:
            if not os.path.exists(self.train_csv):
                with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "update_step","policy_loss","value_loss","entropy","total_loss",
                        "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","samples"
                    ])
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

        # atexit で強制 flush (プロセス終了前に残バッファを吐き出す)
        try:
            atexit.register(self._atexit_flush)
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
        self.update_step += 1
        # 旧バージョンの train_updates.csv に追記しようとして列不足になるケースを検出し再生成 (一度だけ)
        try:
            if os.path.exists(self.train_csv):
                with open(self.train_csv, 'r', encoding='utf-8') as rf:
                    header_line = rf.readline().strip()
                if 'cum_pos_rate' not in header_line:
                    # バックアップして新ヘッダで再生成
                    bak = self.train_csv + '.bak'
                    if not os.path.exists(bak):
                        os.replace(self.train_csv, bak)
                        with open(self.train_csv, 'w', newline='', encoding='utf-8') as wf:
                            writer = csv.writer(wf)
                            writer.writerow([
                                "update_step","policy_loss","value_loss","entropy","total_loss",
                                "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","samples"
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
        if (not self.disable_csv) and (not self.csv_summary_only) and (self.update_step % self.csv_train_every) == 0:
            row = [
                self.update_step,
                metrics.get("policy_loss"),
                metrics.get("value_loss"),
                metrics.get("entropy"),
                metrics.get("loss"),
                metrics.get("value_acc"),
                metrics.get("value_brier"),
                metrics.get("policy_kl"),
                metrics.get("policy_top1_match"),
                metrics.get("pos_rate"),
                metrics.get("cum_pos_rate"),
                metrics.get("samples"),
            ]
            if self._buffer_enabled:
                self._train_buf.append(row)
                self._maybe_flush_train()
            else:
                try:
                    with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                except OSError as e:
                    self._mark_disk_full(e, 'train_csv')
                    return
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
                            "update_step","policy_loss","value_loss","entropy","total_loss",
                            "value_acc","value_brier","policy_kl","policy_top1_match","pos_rate","cum_pos_rate","samples"
                        ])
                with open(self.train_csv, 'a', newline='', encoding='utf-8') as f:
                    m = self._last_train_metrics
                    writer = csv.writer(f)
                    writer.writerow([
                        self.update_step,
                        m.get("policy_loss"), m.get("value_loss"), m.get("entropy"), m.get("loss"),
                        m.get("value_acc"), m.get("value_brier"), m.get("policy_kl"), m.get("policy_top1_match"),
                        m.get("pos_rate"), m.get("cum_pos_rate"), m.get("samples")
                    ])
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
            self.flush_buffers(force=True)
        except Exception:
            pass

    def __del__(self):  # pragma: no cover
        try:
            self.flush_buffers(force=True)
        except Exception:
            pass

__all__ = ["TrainingLogger"]
