Language: [English](README.md) | 日本語

# Active Inference Visualizer

能動的推論（Active Inference）による身体的意思決定を可視化するインタラクティブな Streamlit ダッシュボード。[Priorelli et al. (2025)](https://doi.org/10.1016/j.neunet.2025.107249) の論文に基づき、オリジナルの PyTorch コードを純粋な NumPy で再実装しています。数学的基盤を透明かつアクセスしやすい形で提供します。

## 特徴

- リーチング課題を行う能動的推論エージェントの**インタラクティブシミュレーション**
- 信念ダイナミクス、自由エネルギー、腕の運動学の**リアルタイム可視化**
- **純粋な NumPy 実装** -- 全勾配を解析的に計算（PyTorch 不要）
- 能動的推論の数学を解説する**教育コンテンツ**
- さまざまなパラメータ設定を試せる**実験プリセット**

## 技術スタック

| カテゴリ | ツール |
|----------|--------|
| 言語 | Python 3.13+ |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) + pyproject.toml |
| Web フレームワーク | Streamlit |
| 数値計算 | NumPy, SciPy |
| 可視化 | Plotly（インタラクティブ）, Matplotlib（静的エクスポート） |
| リンター | Ruff, mypy（strict モード） |
| テスト | pytest + pytest-cov |

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/shimo4228/active-inference-viz.git
cd active-inference-viz

# 依存関係をインストール（uv が必要）
uv sync
```

## 使い方

```bash
# ダッシュボードを起動
uv run streamlit run src/active_inference_viz/app.py

# テストを実行
uv run pytest --cov=src --cov-report=term-missing

# リント & 型チェック
uv run ruff check src/ tests/
uv run mypy src/
```

## プロジェクト構成

```
src/active_inference_viz/
├── model/          # 数学モデルのコア
│   ├── config.py       # SimConfig（frozen dataclass）
│   ├── math_utils.py   # 線形代数ヘルパー
│   ├── discrete.py     # 離散状態推論
│   ├── continuous.py   # 連続状態推論
│   ├── brain.py        # エージェントの脳（信念更新）
│   └── simulation.py   # 試行ランナー
├── viz/            # 可視化コンポーネント
│   ├── theme.py        # カラースキーム & スタイリング
│   ├── arm_view.py     # 3関節アームの描画
│   └── belief_panel.py # 信念分布プロット
├── scenarios/      # 実験プリセット
├── tutorial/       # 教育コンテンツ
└── app.py          # Streamlit エントリーポイント

tests/              # pytest テストスイート
docs/
├── references/     # 論文表記法、オリジナルコードマッピング
└── MATH-REFERENCE.md
```

## ステータス

| フェーズ | 内容 | 状況 |
|----------|------|------|
| Phase 0 | プロジェクトセットアップ | 完了 |
| Phase 1 | 数学モデルのコア | 進行中 |
| Phase 2 | MVP 可視化 | 計画中 |

## ライセンス

MIT
