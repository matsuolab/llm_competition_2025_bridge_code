import json
import time
from pathlib import Path

def monitor_progress(storage_dir):
    """進捗をリアルタイムで監視"""
    log_file = Path(storage_dir) / "evolution_log.json"
    
    while True:
        if log_file.exists():
            with open(log_file, "r") as f:
                data = json.load(f)
                
            print(f"\n{'='*50}")
            print(f"Generation: {data.get('current_generation', 0)}")
            print(f"Evaluations: {data.get('num_evaluations', 0)}")
            print(f"Best Score: {data.get('best_score', 0):.4f}")
            print(f"Current Score: {data.get('current_score', 0):.4f}")
            
            # 上位5モデルを表示
            if "top_models" in data:
                print("\nTop 5 Models:")
                for i, model in enumerate(data["top_models"][:5], 1):
                    print(f"  {i}. Score: {model['score']:.4f}")
        
        time.sleep(30)  # 30秒ごとに更新

# 使用例
# monitor_progress("storage/run_20240301_120000")