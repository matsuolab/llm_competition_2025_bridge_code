# launch_training.py
import os
import sys
from datetime import datetime, timezone, timedelta

# Retrieve Slurm/HPC node information from environment variables; use default values if none are found
# In a real Ray Job environment, these values should be hard‑coded or provided through other mechanisms
NNODES = 3
GPUS_PER_NODE = 8
WANDB_ENTITY="llm-2025-sahara"
WANDB_PROJECT_NAME = "deepseek_r1_0528_gsm8k_ppo_peft"
WANDB_RUN_NAME = datetime.now(timezone(timedelta(hours=9))).isoformat()[:19].replace(":", "-")
WANDB_RUN_GROUP = "llama3.2_SFT_multinode_ppo"


# Build the argument list for verl.trainer.main_ppo
# https://github.com/volcengine/verl/blob/3a394c9b/docs/advance/ppo_lora.rst
args = [
    f"data.train_files={os.environ['HOME']}/data/gsm8k/train.parquet",
    f"data.val_files={os.environ['HOME']}/data/gsm8k/test.parquet",
    "data.train_batch_size=24",
    "data.max_prompt_length=512",
    "data.max_response_length=256",
    "data.dataloader_num_workers=0",
    f"actor_rollout_ref.model.path={os.environ['HOME']}/model/DeepSeek-R1-0528",
    "actor_rollout_ref.model.lora_rank=1",
    "actor_rollout_ref.model.lora_alpha=2",
    "actor_rollout_ref.model.target_modules=['q_proj','v_proj']",
    "actor_rollout_ref.model.use_shm=False",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.ppo_mini_batch_size=24",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.layered_summon=True",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "critic.optim.lr=1e-5",
    f"critic.model.path={os.environ['HOME']}/model/Llama-3.2-1B-Instruct",
    "critic.model.lora_rank=1",
    "critic.model.lora_alpha=2",
    "critic.model.target_modules=['q_proj','v_proj']",
    "critic.ppo_mini_batch_size=24",
    "critic.ppo_micro_batch_size_per_gpu=1",
    "algorithm.kl_ctrl.kl_coef=0.001",
    "trainer.logger=['console','wandb']",
    "trainer.val_before_train=False",
    f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
    f"trainer.nnodes={NNODES}",
    "trainer.save_freq=1",
    "trainer.test_freq=-1",
    f"trainer.default_local_dir={os.environ['HOME']}/training/multinode_ppo/gsm8k/checkpoints",
    f"trainer.project_name={WANDB_PROJECT_NAME}",
    f"trainer.experiment_name={WANDB_RUN_NAME}",
    "trainer.total_epochs=1",
]


from verl.trainer import main_ppo
sys.argv = ["verl.trainer.main_ppo"] + args
main_ppo.main()