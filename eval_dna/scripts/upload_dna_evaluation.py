import wandb

# チーム名をentityで指定
wandb.init(
    entity="llm-2025-sahara",  # ここにチーム名を指定
    project="qwen3-32b", 
    name="eval_dna"
)

# evaluationアーティファクトを作成
artifact = wandb.Artifact(
    name="evaluation-results", 
    type="evaluation",
    description="Model inference results on test dataset"
)

# フォルダを追加
artifact.add_dir("eval_dna/evaluation_results/")

wandb.log_artifact(artifact)
wandb.finish()
