# モデルの学習と評価に関する包括的なツールセット

### axolotl
Deep Seekモデルを動かすために使用したファイル群

#### ファイル詳細
- **`_sft.sh`**  
  SLURMジョブスクリプト。3つのGPUノードを使用し、各ノードにSSHで接続してsft_llama.shを実行する分散処理の制御ファイル

- **`deepseek.yml`**  
  Axolotlフレームワーク用のDeepSeekモデル（Qwen3-235B-A22B）の学習設定ファイル。QLoRA（4bit量子化）を使用し、数学データセット（OpenR1-Math-50-io）でファインチューニングを行う設定

- **`sft.sh`**  
  学習処理を実行するスクリプト。CUDA環境の設定、分散学習のパラメータ設定を行い、torchrunコマンドでAxolotlの学習を設定。DeepSpeed Zero3設定を使用した分散学習を設定

---

### dataset_scripts
データセット処理と合成データ作成のためのスクリプト群

#### 🔍 データセット評価
- **`Daime21.py`**  
  AIME数学問題の分野（Field）をLLMで特定する

- **`Daime31.py`**  
  AIME数学の問題と解答をLLMに与え、その解法と解答を出力させる

- **`Daime41.py`**  
  AIME数学の問題をLLMに与え、その解法と解答を出力させる

- **`Dds21.py`**  
  HuggingFaceからストリーミング形式で最初の100問の数学問題をダウンロード

- **`Dds91.py`**  
  データセットjsonファイルから項目（key）を抽出する

#### ⚡ 合成データセット作成
- **`Daime51.py`**  
  合成データ作成：ヒント（解法ステップ）をプロンプトにフル実装

- **`Daime52.py`**  
  合成データ作成：重要度の高いヒント半分程度をプロンプトに実装

- **`Daime53.py`**  
  合成データ作成：最重要のヒントのみをプロンプトに実装

- **`Daime01.py`**  
  csvからQwen3 32B向けparquet（message）形式ファイルを作成

- **`Daime01_extra.py`**  
  csvからQwen3 32B向けparquet（extra_info）形式ファイルを作成

---

### data_preprocessing
データ前処理のためのコード群

- **`extract_HLE500`**  
  HLE（Human-Level Evaluation）データセットから500問をランダムに抽出するためのコード

- **`extract_MixtureOfThoughts`**  
  データセットからMixtureOfThoughtsを抽出するためのコード