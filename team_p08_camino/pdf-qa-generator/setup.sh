#!/bin/bash

# --- プロジェクト自動セットアップスクリプト (最終修正版) ---

set -e # エラーが発生したらスクリプトを停止

# --- サーバー環境の構築 ---
echo "✅ 1. vLLMサーバー用の仮想環境 (venv_server) を作成します..."
python -m venv venv_server
source venv_server/bin/activate

echo "✅ 2. vLLMサーバー用のライブラリをインストールします..."
pip install --upgrade pip
pip install -r requirements_server.txt
deactivate


# --- クライアント環境の構築 ---
echo "✅ 3. クライアント用の仮想環境 (venv_client) を作成します..."
python -m venv venv_client
source venv_client/bin/activate

echo "✅ 4. クライアント用のライブラリをインストールします..."
pip install --upgrade pip
# 必要なライブラリとバージョンが全て記載されたrequirements_client.txtからインストール
pip install -r requirements_client.txt
deactivate


echo "🎉 2つの仮想環境のセットアップが完了しました!"
echo "次に、Hugging Faceにログインしてください: huggingface-cli login"