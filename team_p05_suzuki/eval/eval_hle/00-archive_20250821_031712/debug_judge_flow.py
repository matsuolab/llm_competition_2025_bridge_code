#!/usr/bin/env python3
"""
judge処理のデバッグスクリプト
"""
import json
import os
from pathlib import Path
from datasets import load_dataset

def debug_judge_flow():
    """judge処理の各段階をデバッグ"""
    
    print("🔍 Judge処理フローのデバッグ")
    print("=" * 50)
    
    # 1. 予測結果ファイルの確認
    model_name = "Qwen3-235B-A22B"
    prediction_file = f"predictions/hle_{model_name}.json"
    
    print(f"1. 予測結果ファイル: {prediction_file}")
    if os.path.exists(prediction_file):
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        print(f"   ✅ ファイル存在: {len(predictions)} 件の予測結果")
        
        # サンプル表示
        sample_key = list(predictions.keys())[0]
        sample_data = predictions[sample_key]
        print(f"   📋 サンプル構造: {list(sample_data.keys())}")
    else:
        print(f"   ❌ ファイルが存在しません")
        return
    
    # 2. データセットの確認
    print(f"\n2. データセット確認")
    try:
        dataset = load_dataset("team-suzuki/hle-extract", split="test").to_dict()
        all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        print(f"   ✅ データセット読み込み: {len(all_questions)} 問")
        
        # IDの一致確認
        dataset_ids = set(q["id"] for q in all_questions)
        prediction_ids = set(predictions.keys())
        matched_ids = dataset_ids & prediction_ids
        print(f"   📊 ID一致数: {len(matched_ids)} / {len(dataset_ids)}")
        
    except Exception as e:
        print(f"   ❌ データセット読み込みエラー: {e}")
        return
    
    # 3. judge結果ファイルの確認
    judge_file = f"judged/judged_hle_{model_name}.json"
    print(f"\n3. Judge結果ファイル: {judge_file}")
    
    if os.path.exists(judge_file):
        with open(judge_file, 'r') as f:
            judged_predictions = json.load(f)
        print(f"   ✅ ファイル存在: {len(judged_predictions)} 件のjudge結果")
        
        # judge結果の構造確認
        if judged_predictions:
            sample_judge_key = list(judged_predictions.keys())[0]
            sample_judge_data = judged_predictions[sample_judge_key]
            print(f"   📋 Judge構造: {list(sample_judge_data.keys())}")
            
            # judge_responseの存在確認
            judge_with_response = sum(1 for v in judged_predictions.values() if "judge_response" in v)
            print(f"   📊 judge_response有り: {judge_with_response} / {len(judged_predictions)}")
        else:
            print(f"   ⚠️ Judge結果が空です")
    else:
        print(f"   ❌ Judge結果ファイルが存在しません")
    
    # 4. 処理対象問題の確認
    print(f"\n4. 処理対象問題の確認")
    questions_to_judge = [q for q in all_questions if q["id"] in predictions]
    print(f"   📊 Judge対象: {len(questions_to_judge)} 問")
    
    if os.path.exists(judge_file):
        with open(judge_file, 'r') as f:
            judged_predictions = json.load(f)
        unjudged_questions = [q for q in questions_to_judge if q["id"] not in judged_predictions]
        print(f"   📊 未Judge: {len(unjudged_questions)} 問")
    else:
        print(f"   📊 未Judge: {len(questions_to_judge)} 問 (judge結果ファイル未作成)")
    
    # 5. 環境変数確認
    print(f"\n5. 環境変数確認")
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key[:4] + '*' * (len(api_key) - 4)
        print(f"   ✅ OPENAI_API_KEY: {masked_key}")
    else:
        print(f"   ❌ OPENAI_API_KEY: 未設定")
    
    print("\n" + "=" * 50)
    print("🎯 デバッグ完了")

if __name__ == "__main__":
    debug_judge_flow()
