import wandb

# チーム名をentityで指定
wandb.init(
    entity="llm-2025-sahara",  # ここにチーム名を指定
    project="qwen3-32b", 
    name="eval_hle"
)

# predictionsアーティファクトを作成
predictions_artifact = wandb.Artifact(
    name="predictions", 
    type="evaluation",
    description="Model predictions on HLE dataset"
)

# judgedアーティファクトを作成
judged_artifact = wandb.Artifact(
    name="judged", 
    type="evaluation",
    description="Judge results on HLE dataset"
)

# predictionsフォルダを追加
predictions_artifact.add_dir("predictions/")

# judgedフォルダを追加
judged_artifact.add_dir("judged/")

wandb.log_artifact(predictions_artifact)
wandb.log_artifact(judged_artifact)
wandb.finish()