# daihugou_rl

大富豪（大貧民）AI対戦・強化学習用シミュレーション環境

## 開発支援ツール

本リポジトリでは、AIコードレビュー・自動ドキュメント生成支援のために CodeRabbit（フリー） を導入しています。
CodeRabbitはVS Code拡張として利用でき、PRレビューやコード説明、リファクタ提案などをAIがサポートします。
（詳細: https://coderabbit.ai/ja ）


## 概要

このリポジトリは、トランプゲーム「大富豪（大貧民）」のルールに基づいたAI対戦・強化学習用のPython環境です。  
複数のエージェント（AI）が自動で対戦し、順位やカード交換、階段出し・革命などのルールも実装されています。

## ディレクトリ構成

```
daihugou_rl/
├── main.py                # メイン実行スクリプト
├── re_main.py             # 別バージョンのメイン（リファクタ用等）
├── agents/                # 標準エージェント（AI）群
│   ├── straight_agent.py
│   ├── random_agent.py
│   ├── rule_based_agent.py
│   └── ＿init＿.py
├── game/                  # ゲーム本体・ルール・環境
│   ├── card.py
│   ├── environment.py
│   ├── game.py
│   ├── player.py
│   ├── rules.py
│   └── ＿init＿.py
├── re_agents/             # 別バージョンのエージェント
│   ├── random_agent.py
│   └── ＿init＿.py
├── re_game/               # 別バージョンのゲーム本体
│   ├── card.py
│   ├── environment.py
│   ├── game.py
│   ├── player.py
│   ├── rules.py
│   └── ＿init＿.py
└── venv/                  # 仮想環境（無視してOK）
```

## 主なファイル・モジュール

- `main.py`  
  シミュレーションのエントリーポイント。複数エージェントで自動対戦し、順位集計も行う。

- `game/environment.py`  
  強化学習・AI対戦用の環境クラス（`DaifugoSimpleEnv`）。エージェントの行動選択や合法手生成、リセット処理など。

- `game/game.py`  
  ゲーム進行の本体クラス。カード配布、ターン管理、カード交換、順位決定など。

- `game/rules.py`  
  ルール判定、階段判定、革命、カード交換ロジック（順位に応じて強い/弱いカードを交換）など。

- `agents/`  
  さまざまなAIエージェント（例：ランダム、ルールベース、階段優先など）。

## 特徴的なルール実装

- 階段出し、ペア出し、ジョーカー、革命、8切りなど大富豪の主要ルールをサポート
- ゲーム終了後、前回の順位に応じて新しい手札からカード交換を自動実施
- エージェントは差し替え可能で、強化学習やAI対戦の実験が容易

## 実行方法

1. 必要なパッケージをインストール（例: numpy）
2. `main.py` を実行

```bash
python main.py
```

## カスタマイズ

- エージェントの追加・差し替えは `agents/` フォルダにクラスを追加し、`main.py` の `agent_classes` を編集してください。
- ルールやカード交換ロジックの調整は `game/rules.py` を参照。

