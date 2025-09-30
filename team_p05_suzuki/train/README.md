# Train

## モデル訓練ツール

* Step 0　conda環境構築
    * [README_install_conda.md](./README_install_conda.md)
* Step 1　シングルノードのファインチューニング＆強化学習
    * [README_single_node_SFT_PPO.md](./README_single_node_SFT_PPO.md)
* Step 2  マルチノードのファインチューニング＆強化学習
    * [README_multi_node_SFT_PPO.md](./README_multi_node_SFT_PPO.md)

## MLモデル管理 (ml_ops/)

モデルの設定と管理を行うディレクトリです。詳細は [ml_ops/models/README.md](./ml_ops/models/README.md) を参照してください。

### 主な機能
- モデルの自動ダウンロード
- 設定ファイルベースの管理
- Hugging Face Hubとの統合