# Arxiv scraping tool

ArxivのAPIを利用して、

- 論文のメタデータを取得
- 論文のPDFをダウンロード
- RAGに使用するためのチャンキング

を実施するツールです。

## 前提条件

- uvがインストールされていること
- Pythonがインストールされていること

## 実行確認環境

- Windows 11
- Python: 3.13.2
- uv: 0.8.18

## 使い方

- 本ディレクトリに移動
- `uv sync --frozen`を実行してライブラリをインストール
- `uv run python .\arxiv_scraper_sliding_window.py --date-range-start "開始日時(YYYY-MM-DD)" --date-range-end "終了日時(YYYY-MM-DD)"`を実行する

    - 例: `uv run python .\arxiv_scraper_sliding_window.py --date-range-start "2023-01-01" --date-range-end "2023-12-31"`

## 出力

- `collected_papers_sliding/rag_data/rag_chunks.jsonl`: RAGに使用するためのチャンキングされたJSONLファイル
- `collected_papers_sliding/rag_data/rag_stats.json`: RAGに使用するためのチャンキングされたデータの統計情報
