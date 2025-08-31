# launch_training.py
import os
import sys

# Retrieve Slurm/HPC node information from environment variables; use default values if none are found
# In a real Ray Job environment, these values should be hard‑coded or provided through other mechanisms
NNODES = 2
GPUS_PER_NODE = 8
WANDB_ENTITY = "YOU_TEAM_ENTITY_NAME"
WANDB_PROJECT_NAME = "competition_verl_test"
WANDB_RUN_NAME = "llama3.2_SFT_multinode_ppo"
WANDB_RUN_GROUP = "llama3.2_SFT_multinode_ppo"

# Build the argument list for verl.trainer.main_ppo
args = [
    f"data.train_files={os.environ['HOME']}/data/gsm8k/train.parquet",
    f"data.val_files={os.environ['HOME']}/data/gsm8k/test.parquet",
    "data.train_batch_size=256",
    "data.max_prompt_length=512",
    "data.max_response_length=256",
    "data.dataloader_num_workers=0",
    f"actor_rollout_ref.model.path={os.environ['HOME']}/model/Llama-3.2-1B-Instruct",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.9",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "critic.optim.lr=1e-5",
    f"critic.model.path={os.environ['HOME']}/model/Llama-3.2-1B-Instruct",
    "critic.ppo_micro_batch_size_per_gpu=4",
    "algorithm.kl_ctrl.kl_coef=0.001",
    "trainer.logger=['console','wandb']",
    "trainer.val_before_train=False",
    f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
    f"trainer.nnodes={NNODES}",
    "trainer.save_freq=10",
    "trainer.test_freq=10",
    f"trainer.default_local_dir={os.environ['HOME']}/training/multinode/ppo/checkpoints",
    f"trainer.project_name={WANDB_PROJECT_NAME}",
    f"trainer.experiment_name={WANDB_RUN_NAME}",
    "trainer.total_epochs=15",
]


from verl.trainer import main_ppo
sys.argv = ["verl.trainer.main_ppo"] + args
main_ppo.main()