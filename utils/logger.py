import os, json, csv, time
from typing import Dict, Any, Optional

class TrainingLogger:
    """Collects and writes training / episode / MCTS root statistics.

    Outputs:
      logs/train_updates.csv   : per train_step aggregate losses & accuracies
      logs/episodes.csv        : per episode aggregate performance
      logs/mcts_samples.jsonl  : sampled MCTS root statistics per move
      TensorBoard (logs/tb)    : scalar summaries (optional)
    """
    def __init__(self, log_dir: str = "logs", use_tensorboard: bool = True, clear_existing: bool = False):
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
                    print(f"[TrainingLogger] TensorBoard 有効: イベント出力先 = {tb_dir}")
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
        # files
        self.train_csv = os.path.join(log_dir, "train_updates.csv")
        self.episode_csv = os.path.join(log_dir, "episodes.csv")
        self.mcts_jsonl = os.path.join(log_dir, "mcts_samples.jsonl")
        # headers
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
        # config スナップショット (一度だけ) 既に存在しなければ
        self._config_dump_path = os.path.join(log_dir, "config_snapshot.json")
        self._config_written = False
        self.update_step = 0
        self.episode_idx = 0

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

    # ---------------- Train metrics ----------------
    def log_train(self, metrics: Dict[str, Any]):
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
        with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
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
            ])
        if self.tb:
            try:
                for k,v in metrics.items():
                    if isinstance(v,(int,float)):
                        self.tb.add_scalar(f"train/{k}", v, self.update_step)
                try:
                    self.tb.flush()
                except Exception:
                    pass
            except OSError as e:  # Disk full 等
                self._disable_tensorboard(f"OSError:{e}")
            except Exception as e:  # その他は一度警告し続行
                print(f"[TrainingLogger] TensorBoard書き込み失敗 (train): {e}")

    # ---------------- Episode metrics ----------------
    def log_episode(self, metrics: Dict[str, Any]):
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
        with open(self.episode_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_idx,
                metrics.get("avg_rank"),
                metrics.get("first_rate"),
                metrics.get("episode_len"),
                metrics.get("phase_acc"),
                metrics.get("phase_win_rate"),
                metrics.get("phase_wins"),
                metrics.get("phase_attempts"),
                metrics.get("cum_phase_win_rate"),
            ])
        if self.tb:
            try:
                for k,v in metrics.items():
                    if isinstance(v,(int,float)):
                        self.tb.add_scalar(f"episode/{k}", v, self.episode_idx)
                try:
                    self.tb.flush()
                except Exception:
                    pass
            except OSError as e:
                self._disable_tensorboard(f"OSError:{e}")
            except Exception as e:
                print(f"[TrainingLogger] TensorBoard書き込み失敗 (episode): {e}")

    # ---------------- MCTS root sample ----------------
    def log_mcts_sample(self, data: Dict[str, Any]):
        try:
            with open(self.mcts_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except OSError as e:
            # ディスク満杯なら静かに無視 (学習継続優先)
            if self._tb_disabled_reason is None:
                print(f"[TrainingLogger] MCTSサンプル書き込み失敗 (disk?) : {e}")

    # ---------------- Free-form text logging ----------------
    def log_text(self, text: str, filename: str = "events.log", also_print: bool = False):
        """任意のテキストをログに追記し、可能ならTensorBoardにも出力する。

        - logs/events.log にタイムスタンプ付きで追記
        - TensorBoard が有効な場合は add_text で記録（step は update_step または episode_idx）
        - also_print=True の場合は標準出力にも表示
        """
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            line = f"[{ts}] {text}\n"
            path = os.path.join(self.log_dir, filename)
            os.makedirs(self.log_dir, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
            if also_print:
                print(line.strip())
        except Exception:
            # 例外は握りつぶして学習を止めない
            pass

        if self.tb:
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

__all__ = ["TrainingLogger"]
