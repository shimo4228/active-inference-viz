Language: [English](README.md) | 日本語

# Active Inference Visualizer

能動的推論（Active Inference）による身体的意思決定を可視化する、インタラクティブな Streamlit ダッシュボードです。PLOS Computational Biology に掲載された [Priorelli et al. (2025)](https://doi.org/10.1371/journal.pcbi.1013180) の論文に基づき、オリジナルの PyTorch コードを純粋な NumPy で再実装しています。

アプリでは、3 関節の腕がノイズのある手がかりから 2 つの標的のどちらへ手を伸ばすかを決めます。ステップのスライダーを動かすと、信念と腕の動きが一緒に変わる様子を追えます。この能動的推論のモデルでは、標的を選ぶことと腕を動かすことが、同時に進む 2 つの結合した信念の更新です。ここでいう「身体的」な意思決定はこの意味です。2026-10 時点の同リポジトリの README によると、[元のコード](https://github.com/priorelli/embodied-decisions)は `main.py` から起動する PyTorch・Pymunk・Pyglet のスクリプトで、ブラウザで試せる画面がありません。このアプリは自分のマシンの Python で動き、ブラウザで開いて使います。作った経緯は[著者のほかの仕事](#著者のほかの仕事)にあります。

## 特徴

- **ステップごとの可視化**: 2 つの標的についての信念 P(T1) と P(T2)（標的 1 と標的 2 のそれぞれが正解である確率）と、腕・軌跡・エージェントが思っている手の位置をステップごとに表示します
- **純粋な NumPy 実装** -- 全勾配を解析的に計算（PyTorch 不要）
- **使い方パネル**: アプリ内で、結合した 2 つの過程を説明します。離散推論（どちらの標的か）と連続推論（腕をどう動かすか）です
- **手がかりパターンのプリセット**（論文の既定の系列、すべて T1、すべて T2、交互、ランダム 50/50）と、証拠ゲイン（信念がどれだけ速く強く決まるか）・手がかりの信頼度（1 つの手がかりがどれだけ情報を持つか）・到達の動きのスライダー

## 技術スタック

| カテゴリ | ツール |
|----------|--------|
| 言語 | Python 3.13+ |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) + pyproject.toml |
| Web フレームワーク | Streamlit |
| 数値計算 | NumPy（SciPy は宣言されているが、使うのはテストだけ） |
| 可視化 | Plotly（Matplotlib は宣言されているが、アプリではまだ使われていない） |
| リンター | Ruff, mypy（strict モード） |
| テスト | pytest + pytest-cov |

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/shimo4228/active-inference-viz.git
cd active-inference-viz

# 依存関係をインストール（uv が必要。Python 3.13+ が無ければ uv が取得する）
uv sync
```

## 使い方

リポジトリに含まれる `.streamlit/config.toml` で `headless = true` に設定しているので、起動してもブラウザは自動では開きません。起動したら http://localhost:8501 を開いてください。シミュレーションは手元で動きます。Streamlit 自身の匿名の利用統計は既定で有効で、同じファイルに `[browser]` 節を作って `gatherUsageStats = false` を書くと止められます。

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
├── scenarios/      # 実験プリセット用の置き場（空）
├── tutorial/       # 教育コンテンツ用の置き場（空）
└── app.py          # Streamlit エントリーポイント（画面、手がかりパターン、使い方パネル）

tests/              # pytest テストスイート
```

## ステータス

数学モデルのコアとダッシュボードは実装済みで、アプリは動きます。更新式が理論的に正しいかを著者は自分では判断できず、コードと論文の数式の対応は Claude Code が確かめました。テストが確かめているのは NumPy コード自身の性質（解析的なヤコビアンと数値微分の一致、信念の和が 1 になること、手がかりが信念を正しい向きに動かすこと）で、元の PyTorch コードの出力と比べるテストはありません。

## 著者のほかの仕事

- **[理論が分からない論文をブラウザで動かしてしまった ── 能動的推論 × Claude Code](https://zenn.dev/shimo4228/articles/active-inference-viz-dev-story)**（[English](https://dev.to/shimo4228/i-made-a-paper-i-dont-understand-run-in-the-browser-active-inference-x-claude-code-2jog)）: 著者が理論を理解していない論文から、Claude Code がこのアプリを作り直した記録です。最大の難所が 1 つの勾配の項の符号だったこと、結果が理論的に正しいかを著者がいまも判断できないことまで書いています。
- **[attention-not-self](https://github.com/shimo4228/attention-not-self)**: 瞑想の実践から生まれたエッセイと知識グラフです。アビダルマの心の地図を、予測処理・能動的推論と並べて読みます（日本語が正本で、英語の README もあります）。
- **[tiny-lm-lab](https://github.com/shimo4228/tiny-lm-lab)**: transformer の中身を書いて理解するために、Apple の MLX 上で小さな GPT を手で書き、著者の記事で学習させた学習用ラボです。
- **[shimo4228](https://github.com/shimo4228/shimo4228)**: 著者のハブです。5 つの長期プロジェクトとその DOI、Claude Code 用のツールがまとまっています。

## ライセンス

MIT です。

<details>
<summary>ツールと AI アシスタント向けの資料</summary>

Active Inference Visualizer（`active-inference-viz`）は、能動的推論による身体的意思決定をシミュレートして可視化する、インタラクティブな Streamlit アプリです。[Priorelli et al. (2025)](https://doi.org/10.1371/journal.pcbi.1013180) のモデルを、数式で読むだけでなく動くところを見たい人に向けています。元の実装（[priorelli/embodied-decisions](https://github.com/priorelli/embodied-decisions)）は PyTorch・Pymunk・Pyglet のスクリプトで、誰でもブラウザで試せる画面がありません。このリポジトリは数学のコアを純粋な NumPy で書き直し、勾配はすべて PyTorch の autograd ではなく解析的に（3 関節の腕のヤコビアンで）計算しています。著者はリンク先の記事で、更新式が理論的に正しいかを自分では判断できないと書いています。コードと論文の数式の対応は Claude Code が確かめましたが、元の PyTorch コードの出力と比べるテストはありません。

基本的な事実: ライセンスは MIT で、この README に書かれています（リポジトリに独立した LICENSE ファイルはありません）。Python 3.13+ を uv で管理し、画面は Streamlit、計算は NumPy、プロットは Plotly です（Matplotlib は依存に宣言されていますが、コードはまだ import していません。SciPy を import しているのはテストだけです）。開発には Ruff、strict モードの mypy、pytest を使います。状態: コード（`src/` と `tests/`）の最後の変更は 2026-02 です。数学のコアとダッシュボードは動き、`scenarios/` と `tutorial/` は空の置き場です。API キーは不要で、シミュレーションは手元で動きます。Streamlit の匿名の利用統計は既定で有効です。

モデルは 2 つの過程を結合します。離散推論は、ノイズのある手がかりの系列から証拠を積み上げ、2 つの標的のどちらが正しいかの信念 P(T1) と P(T2) を作ります。連続推論は、予測符号化によって腕を信じている標的へ動かすので、腕の位置が信念を映します。

例: `uv run streamlit run src/active_inference_viz/app.py` を実行して http://localhost:8501 を開きます。既定値（論文の 15 個の手がかりの系列、証拠ゲイン 5.0、手がかりの信頼度 alpha_c 0.6、動きのゲイン lambda 0.5）では、ステップのスライダーで手がかりの蓄積から到達までを追え、画面に P(Target 1)、P(Target 2)、現在の決定（T1、T2、Undecided）、フェーズ、腕の図、信念のプロットが出ます。

参照先: 数学のコアは `src/active_inference_viz/model/`（`discrete.py`、`continuous.py`、`brain.py`、`simulation.py`）、プロットは `src/active_inference_viz/viz/`、作った経緯は「著者のほかの仕事」の記事にあります。著者のほかの仕事の一覧は https://github.com/shimo4228/shimo4228 にあります。

</details>
