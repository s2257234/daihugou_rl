# daihugou_rl

大富豪（大貧民）AI 対戦 / AlphaZero 風強化学習 + Elo 評価環境。

学習 (self-play + MCTS) → チェックポイント保存 → Elo 対戦評価 → 自動グラフ生成 までを一通り回せる最小構成です。

## 主要機能

- 大富豪ルール: 革命 / 階段 / 8切り / ジョーカー流し / 順位に応じた次ゲームのカード交換
- 強化学習エージェント: AlphaZero 風 MCTS + Policy/Value ネット (MLP)
- リプレイバッファ（共有モード対応）
- ロギング: CSV / (任意) TensorBoard
- Elo レーティング: 対戦結果を多人数ペアワイズに分解して更新
- 評価自動可視化: 評価完了ごとに勝率・平均順位・レーティング推移 PNG を再生成

## ディレクトリ (抜粋)

```
agents/         # 学習/ベースラインエージェント (AlphaZero, Random, RuleBased, etc.)
game/           # ゲーム進行・環境・ルール
trainer/        # 学習ループ (self-play + train updates)
evaluation/     # Elo 評価, 自動プロット (evaluator / run_eval / plot_eval / rating)
checkpoints/    # 最新モデル保存 (policy_value_latest.pt)
logs/           # 学習 & 評価ログ
  └─ elo/      # Elo 関連ファイル (ratings.json, ratings.csv, eval_metrics.csv, figs/*.png)
requirements.txt
```

## 学習 (Self-Play + 更新)

少数エピソードで動作確認:

```powershell
python -m trainer.trainer --episodes 5 --updates 5 --device auto
```

出力:
- `checkpoints/policy_value_latest.pt` モデル
- `replay_buffer.joblib` リプレイバッファ
- `logs/episodes.csv`, `logs/train_updates.csv` など

## Elo 評価（自動グラフ付き）

学習済み (または初期) モデルをランダム/ルールベースと対戦評価:

```powershell
python -m evaluation.run_eval --episodes 20 --checkpoint checkpoints/policy_value_latest.pt
```

生成/更新される主ファイル:

| ファイル | 内容 |
|----------|------|
| `logs/elo/ratings.json` | 最新 Elo 状態 (player -> rating) |
| `logs/elo/ratings.csv`  | ゲームごとの全プレイヤー Elo 履歴 |
| `logs/elo/eval_metrics.csv` | episode 単位の `win_rate, avg_rank, rating_p0` |
| `logs/elo/figs/eval_progress.png` | WinRate & AvgRank 推移 |
| `logs/elo/figs/p0_rating.png` | P0 (学習エージェント) Elo 推移 |
| `logs/elo/figs/elo_history.png` | 全プレイヤー Elo 推移 |

オプション:

```text
--mix rule,random,random   # ベースライン3枠指定 (rule/random)
--no-tb                    # TensorBoard 無効
--no-auto-plot             # 評価終了後の自動PNG生成を無効化
--num-sim 32               # 評価時 MCTS シミュレーション数上書き
--filter-dominated         # 圧倒的無駄遣いの禁止フィルタを有効化（推論時のみ）
```

### 圧倒的無駄遣いの禁止フィルタ

`--filter-dominated` オプションを有効にすると、より弱いカードで勝てる場面で、より強いカードを使う手を候補から除外します。

**例**: 場に「3」が出ている時、手札に「4」と「2」がある場合
- **フィルタなし**: 「4」を出す、「2」を出す、パス
- **フィルタあり**: 「4」を出す、パス （「2」を出す手が除外される）

**理由**: 「4」で勝てる場面で「2」を使うメリットはほぼありません。

**効果**: MCTS探索木のサイズが削減され、より有望な手に探索リソースを集中できます。

詳細は [docs/DOMINATED_FILTER.md](docs/DOMINATED_FILTER.md) を参照してください。

## 手動でプロット再生成のみ行いたい場合

```powershell
python -m evaluation.plot_eval --elo-dir logs/elo
```

## 方針 / 実装メモ

- Elo: 4人最終順位を全ペア勝敗に変換し K=32 で Δ を均等適用。
- 勝率は「1位獲得率」、平均順位は 1(最良)～4(最悪)。
- 評価メトリクスは逐次 CSV 追記し、再実行で継続。リセットしたい場合は `logs/elo` フォルダを削除。

## 代表クラス / スクリプト

| パス | 役割 |
|------|------|
| `agents/drl_agent.py` | AlphaZero 風 MCTS エージェント |
| `agents/models.py` | PolicyValueNet (方策+価値) |
| `trainer/trainer.py` | Self-play & train スケジューラ |
| `evaluation/evaluator.py` | 単発評価ロジック (1ゲーム生成) |
| `evaluation/rating.py` | Elo 計算 & 永続化 |
| `evaluation/run_eval.py` | CLI, 自動プロット呼び出し |
| `evaluation/plot_eval.py` | CSV / Elo 履歴のPNG化 |
| `game/rules.py` | 革命 / 8切り / 階段 判定など |

## 今後の拡張候補

- K係数スケジューリング (初期高速収束 → 安定化)
- チェックポイント複数世代の総当たり評価
- 勝率の信頼区間 (Wilson) 表示
- 評価時にステップ数や革命頻度の併記



