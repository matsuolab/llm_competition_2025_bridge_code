#!/usr/bin/env python3
"""
評価結果をまとめてwandbにアップロードするスクリプト
"""

import json
import yaml
import os
import glob
import sys
from pathlib import Path
import wandb
from typing import Dict, List, Any
import re

import numpy as np
import pandas as pd


def find_latest_result_file(task_dir: str) -> str:
    """タスクディレクトリ内の最新の結果ファイルを見つける"""
    pattern = os.path.join(task_dir, "results", "**", "*.json")
    result_files = glob.glob(pattern, recursive=True)
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {task_dir}")
    
    # 最新のファイルを返す（ファイル名にタイムスタンプが含まれているため）
    return max(result_files, key=os.path.getctime)

def extract_metrics_from_results(results: Dict[str, Any]) -> Dict[str, float]:
    """結果からメトリクスを抽出し、@Kの場合はK=1の値を使用"""
    metrics = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            # ネストした辞書の場合は再帰的に処理
            nested_metrics = extract_metrics_from_results(value)
            for nested_key, nested_value in nested_metrics.items():
                metrics[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, (int, float)):
            metrics[key] = value
    
    return metrics

def load_task_results(task_name: str, eval_results_dir: str = "eval_results") -> Dict[str, Any]:
    """タスクの結果を読み込む"""
    task_dir = os.path.join(eval_results_dir, task_name)
    
    if not os.path.exists(task_dir):
        print(f"Warning: Task directory {task_dir} not found")
        return {}
    
    try:
        result_file = find_latest_result_file(task_dir)
        print(f"Loading results from: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 結果からメトリクスを抽出
        if "results" in data:
            task_results = {}
            for result_key, result_data in data["results"].items():
                if result_key != "all":  # "all"以外の結果を使用
                    metrics = extract_metrics_from_results(result_data)
                    task_results.update(metrics)
            
            return {
                "task_name": task_name,
                "metrics": task_results,
                "model_name": data.get("config_general", {}).get("model_name", "unknown"),
                "evaluation_time": data.get("config_general", {}).get("total_evaluation_time_secondes", "unknown")
            }
    
    except Exception as e:
        print(f"Error loading results for {task_name}: {e}")
        return {}

def calculate_average_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """全タスクの平均値を計算"""
    if not all_results:
        return {}

    task_metrics = {
        "gsm8k": "extractive_match",
        "aime24": "math_pass@1:1_samples",
        "gpqa-diamond": "gpqa_pass@1:1_samples",
        "truthfulqa-mc": "truthfulqa_mc1",
        "toxigen": "acc_norm"
    } 
    weights = [0.3, 0.3, 0.3, 0.05, 0.05]

    eval_metrics = {"average": 0.0}

    # 全メトリクス名を収集
    representative_metric = []
    for result in all_results:
        if "metrics" in result:
            for metric_name in result["metrics"].keys():
                if task_metrics[result["task_name"]] == metric_name:
                    representative_metric.append(result["metrics"][metric_name])
                    rep_name = result["task_name"] + "/" + metric_name
                    eval_metrics[rep_name] = result["metrics"][metric_name]
    eval_metrics["average"] = np.average(representative_metric, weights=weights)
    return eval_metrics

def make_evaluation_sample_tables(log_path, task):
    if task in ["toxigen", "truthfulqa-mc"]:
        df = pd.read_parquet(log_path, columns=["example", "choices", "predictions", "gold_index"])

        if task == "truthfulqa":
            # Exampleにある7個のQ-A pairと8個目Qから、最後のQのみを抜き出す
            df["question"] = df["example"].map(lambda x: "Question: " + re.findall("Q:\s*(.*)", x)[-1])
        else:
            df["question"] = df["example"].copy()
            
        df["formatted_choices"] = df["choices"].map(lambda x: "multiple-choice options are "+ "`" + "".join([f"{i}.{choice} " for i, choice in enumerate(x)]))[0] + "`"
        df["example"] = df["question"] + "\n" +df["formatted_choices"]
        df["gold"] = df["gold_index"].map(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df["predictions"] = df["predictions"].map(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
    else:
        df = pd.read_parquet(log_path, columns=["example", "predictions", "gold"])
        # wandbのUIで表示されるように、np.arrayからstringに変換
        df[["example", "predictions", "gold"]] = df[["example", "predictions", "gold"]].applymap(
            lambda x: "\n".join(x.tolist()) if isinstance(x, np.ndarray) else str(x)
        )

    df["dataset"] = task
    return df

def convert_train_params_to_flat(train_params: Dict[str, Any]) -> Dict[str, Any]:
    """train_paramsをフラットな形式に変換"""
    flat_params = {}
    for key, value in train_params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_params[f"{key}_{sub_key}"] = sub_value
        else:
            flat_params[key] = value
    return flat_params

def upload_to_wandb(run_name: str, all_results: List[Dict[str, Any]], average_metrics: Dict[str, float], log_training_config: Dict[str, Any]):
    """結果をwandbにアップロード"""
    # wandbの初期化
    wandb.init(
        project="minicomp-test",
        entity="LLMcompe-Team-Watanabe",
        name=run_name,
        config={
            "num_tasks": len(all_results),
            "tasks": [result.get("task_name", "unknown") for result in all_results]
        }
    )
    
    # 各タスクの結果をログ
    for result in all_results:
        if "metrics" in result:
            task_name = result.get("task_name", "unknown")
            metrics = result["metrics"]
            
            # タスク名をプレフィックスとして追加
            prefixed_metrics = {f"{task_name}_{k}": v for k, v in metrics.items()}
            wandb.log(prefixed_metrics)

    # input, prediction, answerが記録された最新のlogファイルのpathを取得
    tasks = set([result["task_name"] for result in all_results])
    detailed_log_paths = {task: list(Path(f"./eval_results/{task}/details").glob("**/*.parquet")) for task in tasks}
    detailed_log_paths = {
        k: max(v, key=lambda x: x.stat().st_mtime)
        for k, v in detailed_log_paths.items() if v
    }
    

    # サマリーテーブルを作成
    each_metrics_data = []
    for result in all_results:
        if "metrics" in result:
            row = {
                "run_name": run_name,
                "task": result.get("task_name", "unknown"),
                "model": result.get("model_name", "unknown"),
                "evaluation_time": result.get("evaluation_time", "unknown")
            }
            row.update(result["metrics"])
            each_metrics_data.append(row)
    
    # 平均値の行を追加
    average_metrics_data = []
    if average_metrics:
        avg_row = {"run_name": run_name} 
        avg_row.update(average_metrics)
        average_metrics_data.append(avg_row)
    
    wandb.save("config.yaml")
    wandb.save("train_config.json")

    # テーブルを作成してログ
    if log_training_config:
        table = wandb.Table(dataframe=pd.DataFrame([log_training_config]))
        wandb.log({"Training Config Table": table})
        print("Training Config Table logged")
    if each_metrics_data:
        table = wandb.Table(dataframe=pd.DataFrame(each_metrics_data))
        wandb.log({"Detail Metrics Table": table})
        print("Detail Metrics Table logged")
    if average_metrics_data:
        table = wandb.Table(data=pd.DataFrame(average_metrics_data))
        wandb.log({"Evaluation Score Table": table})
        print("Evaluation Score Table logged")
    if detailed_log_paths:
        df_list = []
        for task, path in detailed_log_paths.items():
            df = make_evaluation_sample_tables(path, task)
            df_list.append(df)
        table = wandb.Table(dataframe=pd.concat(objs=df_list, ignore_index=True)[["dataset", "example", "predictions", "gold"]])
        wandb.log({"Evaluation Samples Table": table})
        print("Evaluation Samples Table logged")
            
    wandb.finish()

def main():
    training_config_path = sys.argv[1]
    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    run_name = training_config["run_name"]
    with open("eval_config.yaml", "r") as f:
        generation_config = yaml.safe_load(f)["model_parameters"]["generation_parameters"]
    training_config["generation_parameters"] = generation_config
    log_training_config = convert_train_params_to_flat(training_config)

    """メイン関数"""
    # 評価対象のタスク
    tasks = ["gsm8k", "aime24", "gpqa-diamond", "truthfulqa-mc", "toxigen"]

    
    print("Loading evaluation results...")
    
    # 各タスクの結果を読み込み
    all_results = []
    for task in tasks:
        print(f"\nProcessing task: {task}")
        result = load_task_results(task)
        if result:
            all_results.append(result)
            print(f"  Loaded {len(result.get('metrics', {}))} metrics")
        else:
            print(f"  No results found for {task}")
    
    if not all_results:
        print("No results found for any task!")
        return
    
    # 平均値を計算
    print("\nCalculating averages...")
    average_metrics = calculate_average_metrics(all_results)
    print(f"Calculated {len(average_metrics)} average metrics")
    
    # 結果を表示
    print("\n=== Task Results ===")
    for result in all_results:
        task_name = result.get("task_name", "unknown")
        metrics = result.get("metrics", {})
        print(f"\n{task_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n=== Average Results ===")
    print(f"Average metrics: {average_metrics['average']:.4f}")
    
    # wandbにアップロード
    print("\nUploading to wandb...")
    try:
        upload_to_wandb(run_name, all_results, average_metrics, log_training_config)
        print("Successfully uploaded to wandb!")
    except Exception as e:
        print(f"Error uploading to wandb: {e}")

if __name__ == "__main__":
    main() 