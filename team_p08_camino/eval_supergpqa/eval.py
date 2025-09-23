import argparse
import json
import os
import re
import string
from pathlib import Path
from datasets import load_dataset

def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s

def parse_prediction(response: str) -> str:
    """
    モデルの生成したテキストから最終回答を抽出します。
    <think>タグがある場合は、その後の部分を解析対象とします。
    その後、"Answer:" または "Exact Answer:" の後を探します。
    """
    if not isinstance(response, str):
        return ""

    response_to_parse = response
    # <think>タグがある場合は、その後の部分を解析対象の文字列とする
    if "</think>" in response:
        parts = response.split("</think>", 1)
        if len(parts) > 1:
            response_to_parse = parts[1].strip()
        else:
            response_to_parse = "" # </think>の後ろに何もない場合

    # 解析対象の文字列からキーワードを探す
    match = re.search(r"exact answer: (.*)", response_to_parse, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"answer: (.*)", response_to_parse, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # キーワードが見つからなければ、解析対象の文字列全体を返す
    # (モデルが指示に従わず、キーワードなしで回答した場合のフォールバック)
    return response_to_parse.strip()

def is_correct(prediction: str, label_data: dict, answer_type: str) -> bool:
    if not prediction:
        return False
    
    if answer_type == "multiple_choice":
        gold_answer_text = label_data.get("answer", "")
        gold_answer_letter = label_data.get("answer_letter", "")

        if prediction.lower() == gold_answer_text.lower():
            return True
        if gold_answer_letter and prediction.lower() == gold_answer_letter.lower():
            return True
        
        return False
        
    elif answer_type == "short_answer":
        return normalize_answer(prediction) == normalize_answer(label_data.get("answer", ""))
    else:
        return prediction == label_data.get("answer", "")

def main():
    parser = argparse.ArgumentParser(description="SuperGPQAタスクの評価スクリプト")
    parser.add_argument("--prediction_file", type=Path, required=True, help="モデルの予測結果が書かれたJSONファイルのパス")
    args = parser.parse_args()

    try:
        with open(args.prediction_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 予測ファイルが見つかりません: {args.prediction_file}")
        return

    print("正解データをローカルファイルからロードします...")
    label_file_path = "../SuperGPQA/data/SuperGPQA-all.jsonl"
    try:
        labels_dataset = load_dataset("json", data_files=label_file_path, split="train")
        labels = {item['uuid']: item for item in labels_dataset}
        print("ロード完了。")
    except Exception as e:
        print(f"エラー: データセット '{label_file_path}' のロードに失敗しました: {e}")
        return

    correct_count = 0
    total_count = 0
    for q_id, pred_data in predictions.items():
        if q_id not in labels:
            continue
        
        total_count += 1
        
        label_data = labels[q_id]
        model_response = pred_data.get("response", "")
        
        answer_type = "multiple_choice" if "options" in label_data and label_data["options"] else "exact_match"
        
        predicted_answer = parse_prediction(model_response)
        
        if is_correct(predicted_answer, label_data, answer_type):
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Total Questions Evaluated: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")

    output_dir = Path("eval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "prediction_file": str(args.prediction_file)
    }
    
    output_path = output_dir / "evaluation_result.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"評価結果を {output_path} に保存しました。")

if __name__ == "__main__":
    main()