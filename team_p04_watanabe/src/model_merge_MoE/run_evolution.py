#!/usr/bin/env python3
import os
import json
import yaml
from pathlib import Path
import subprocess
from datetime import datetime

def setup_environment():
    """環境変数とディレクトリの設定"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 作業ディレクトリの作成
    dirs = ["eval_tasks", "storage", "logs", "final_models"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_evolution(config_path, run_id):
    """進化的マージの実行"""
    cmd = [
        "mergekit-evolve",
        config_path,
        "--data-dir", f"storage/run_{run_id}",
        "--task-dir", "eval_tasks",
        "--max-fevals", "100",
        "--num-gpus", "8",
        "--vllm",
        "--trust-remote-code",
        "--save-generations",  # 生成結果を保存
        "--verbose",
    ]
    
    # Weights & Biasesを使う場合
    if os.getenv("WANDB_API_KEY"):
        cmd.extend(["--wandb", "--wandb-project", "openr1-evolution"])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def create_final_model(run_id):
    """最適なモデルの作成"""
    best_config = f"storage/run_{run_id}/best_config.yaml"
    output_path = f"final_models/openr1_evolved_{run_id}"
    
    cmd = [
        "mergekit-yaml",
        best_config,
        output_path,
        "--cuda",
        "--trust-remote-code",
        "--copy-tokenizer",
        "--lazy-unpickle"
    ]
    
    print(f"Creating final model: {output_path}")
    subprocess.run(cmd, check=True)
    
    return output_path

def main():
    run_id = setup_environment()
    
    print(f"Starting evolution run: {run_id}")
    print("=" * 50)
    
    # 進化的アルゴリズムの実行
    run_evolution("evol_config.yaml", run_id)
    
    # 最適モデルの作成
    model_path = create_final_model(run_id)
    
    print("=" * 50)
    print(f"Evolution completed! Model saved to: {model_path}")
    
    # 結果のサマリーを表示
    with open(f"storage/run_{run_id}/evolution_log.json", "r") as f:
        results = json.load(f)
        best_score = results.get("best_score", "N/A")
        print(f"Best score achieved: {best_score}")

if __name__ == "__main__":
    main()