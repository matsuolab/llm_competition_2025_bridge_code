# light_r1_sftデータセット用 Qwen2.5-0.5B LoRA SFTスクリプト
# 動作確認済み: 8 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "  nproc_per_node: GPU数 (2 または 4 推奨)"
    echo "  save_path: チェックポイント保存先ディレクトリ"
    echo "  other_configs: 追加の設定オーバーライド (オプション)"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# 残りの引数を$@として扱うために最初の2つをシフト
shift 2

# FSDP (Fully Sharded Data Parallel) を使用した分散訓練実行
# --standalone: 単一ノードでの実行
# --nnodes=1: ノード数1
# --nproc_per_node: ノードあたりのプロセス数（GPU数）
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/light_r1_sft/train.parquet \
    data.val_files=$HOME/data/light_r1_sft/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=1 \
    +trainer.accumulate_grad_batches=2 \
    trainer.total_epochs=1 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    data.max_length=20960 \
    data.truncation=right \
    trainer.default_local_dir=$save_path \
    trainer.project_name=light-r1-sft-sft \
    trainer.experiment_name=light-r1-sft-sft-qwen-2.5-0.5b-instruct \
    trainer.seed=42 \
    trainer.logger=['console','wandb'] $@ 2>&1 | tee verl_demo.log
