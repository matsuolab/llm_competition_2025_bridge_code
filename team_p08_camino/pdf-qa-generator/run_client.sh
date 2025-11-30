#!/bin/bash

# クライアント用の仮想環境を有効化
echo "クライアント用の仮想環境を有効化しています..."
source venv_client/bin/activate

echo "クライアントを実行します..."
python run_batch.py