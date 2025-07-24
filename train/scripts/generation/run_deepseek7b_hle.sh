set -x

data_path=$HOME/data/hle/test.parquet
save_path=$HOME/data/hle/deepseek7b_hle_gen_test.parquet
model_path=deepseek-ai/deepseek-llm-7b-chat

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    +rollout.seed=42 \
    rollout.temperature=0.0 \
    rollout.top_p=0.9 \
    rollout.prompt_length=2048 \
    rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8