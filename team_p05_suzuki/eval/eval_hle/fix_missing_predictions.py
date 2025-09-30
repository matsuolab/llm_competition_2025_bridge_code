#!/usr/bin/env python3

import json
import asyncio
import os
from datasets import load_dataset
from openai import AsyncOpenAI

# 欠けている問題のID
MISSING_IDS = {
    '66ede973564cb32b38cc8a4a',  # インデックス 28
    '66ecb2eb54baa602e636a457',  # インデックス 31
    '66eaa401c7a3252f0f3fe535',  # インデックス 54
    '66ecd3c6e8b95a8f971fb485',  # インデックス 56
    '66e8cfa03add731d7fce4352'   # インデックス 60
}

async def predict_single_question(client, question, model_name):
    """単一の問題に対して予測を実行"""
    try:
        # プロンプトの構築
        prompt = f"Question: {question['question']}\n\nPlease provide a detailed answer."
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed and accurate answers to questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=35000,
            temperature=0.1
        )
        
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        print(f"Error processing question {question['id']}: {str(e)}")
        return None

async def process_missing_questions():
    """欠けている問題を処理"""
    # データセットを読み込み
    dataset = load_dataset('team-suzuki/hle-extract', split='test')
    
    # 欠けている問題を抽出
    missing_questions = []
    for item in dataset:
        if item['id'] in MISSING_IDS:
            missing_questions.append(item)
    
    print(f"処理対象の問題数: {len(missing_questions)}")
    
    # vLLMクライアントを作成
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )
    
    # 各問題を処理
    results = {}
    for question in missing_questions:
        print(f"処理中: {question['id']}")
        result = await predict_single_question(client, question, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        if result:
            results[question['id']] = result
            print(f"✅ 完了: {question['id']}")
        else:
            print(f"❌ 失敗: {question['id']}")
    
    return results

def main():
    """メイン処理"""
    print("欠けている5問の予測処理を開始します...")
    
    # 既存の予測結果を読み込み
    pred_file = "predictions/hle_DeepSeek-R1-0528-Qwen3-8B.json"
    with open(pred_file, 'r') as f:
        existing_predictions = json.load(f)
    
    print(f"既存の予測結果: {len(existing_predictions)} 問")
    
    # 欠けている問題を処理
    new_predictions = asyncio.run(process_missing_questions())
    
    print(f"新しく処理した問題: {len(new_predictions)} 問")
    
    # 結果をマージ
    existing_predictions.update(new_predictions)
    
    print(f"マージ後の総数: {len(existing_predictions)} 問")
    
    # バックアップを作成
    backup_file = pred_file + ".backup"
    with open(backup_file, 'w') as f:
        json.dump(existing_predictions, f, indent=2)
    
    print(f"バックアップを作成: {backup_file}")
    
    # 元のファイルを更新
    with open(pred_file, 'w') as f:
        json.dump(existing_predictions, f, indent=2)
    
    print(f"予測結果ファイルを更新: {pred_file}")
    
    # 検証
    if len(existing_predictions) == 120:
        print("✅ 成功: 120問すべての予測が完了しました！")
    else:
        print(f"⚠️ 警告: 予測結果は {len(existing_predictions)} 問です")

if __name__ == "__main__":
    main()
