import argparse
import os
import wandb

def main():
    parser = argparse.ArgumentParser(description="Upload HLE evaluation results to wandb")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run", required=True, help="WandB run name", default="eval_hle")
    parser.add_argument("--prediction_dir", default="predictions/", help="Directory containing predictions")
    parser.add_argument("--judged_dir", default="judged/", help="Directory containing judged results")
    args = parser.parse_args()

    # WandB API keyを環境変数から取得してログイン
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        wandb.login()

    # チーム名をentityで指定
    wandb.init(
        entity="llm-2025-sahara",  # ここにチーム名を指定
        project=args.project, 
        name=args.run
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
    predictions_artifact.add_dir(os.path.expanduser(args.prediction_dir))

    # judgedフォルダを追加
    judged_artifact.add_dir(os.path.expanduser(args.judged_dir))

    wandb.log_artifact(predictions_artifact)
    wandb.log_artifact(judged_artifact)
    wandb.finish()

if __name__ == "__main__":
    main()