import argparse
import os
import wandb

def main():
    parser = argparse.ArgumentParser(description="Upload DNA evaluation results to wandb")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run", required=True, help="WandB run name", default="eval_dna")
    parser.add_argument("--evaluation_dir", default="eval_dna/evaluation_results/", help="Directory containing evaluation results")
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

    # evaluationアーティファクトを作成
    artifact = wandb.Artifact(
        name="evaluation-results", 
        type="evaluation",
        description="Model inference results on test dataset"
    )

    # フォルダを追加
    artifact.add_dir(os.path.expanduser(args.evaluation_dir))

    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()
