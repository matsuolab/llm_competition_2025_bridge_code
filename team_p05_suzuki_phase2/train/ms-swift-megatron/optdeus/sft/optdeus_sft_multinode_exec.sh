# For more information on multi-node training launch methods, refer to:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node

set -eu

# Debug information
echo "=== Debug Info ==="
echo "MEGATRON_LM_PATH: $MEGATRON_LM_PATH"
echo "MODELSCOPE_CACHE: $MODELSCOPE_CACHE"
ls -la /workspace/Megatron-LM/ 2>/dev/null || echo "Megatron-LM directory not found"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
echo "=================="

# Set CUDA environment variables explicitly
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export NNODES=${NNODES}
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NPROC_PER_NODE=${NPROC_PER_NODE}
export NCCL_DEBUG=INFO

echo "=== Debug Info ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "=================="


ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

export WANDB_ENTITY=ken05-matuo-llm-88_llm_2025_suzuki

MODEL=${1}
MASTER_PORT=${2}


USE_HF=1  MASTER_PORT=${MASTER_PORT} megatron sft \
    --load ${MODEL} \
    --model_type qwen3_moe_thinking \
    --dataset /home/Competition2025/P05/shareP05/data/SFT_006_origin_2/train.parquet \
    --val_dataset /home/Competition2025/P05/shareP05/data/SFT_006_origin_2/validation.parquet \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --freeze_parameters_ratio 1.0 \
    --trainable_parameters_regex "decoder\.layers\.(47|56|65|74|83|92)\..*" \
    --lazy_tokenize true \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_expert_capacity_factor 1.0 \
    --moe_token_dispatcher_type alltoall \
    --bf16 true \
    --fp16 false \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/multinode/${MODEL} \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_exp_name optdeus_sft_multinode \
    --wandb_project optdeus_sft_megatron \
    --wandb_save_dir wandb_logs