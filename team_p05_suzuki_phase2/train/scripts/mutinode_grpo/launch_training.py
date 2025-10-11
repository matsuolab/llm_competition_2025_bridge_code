# launch_training.py
import os
import sys
import logging
import subprocess
from pathlib import Path
from filelock import FileLock
import shutil


def cache_model_locally(source_path_str: str, local_path_str: str):
    """
    指定されたモデルをネットワークストレージからローカルにコピーする。
    同一ノード上の複数プロセスが同時にコピーしないようにロック機構を持つ。
    """
    source_path = Path(source_path_str)
    local_path = Path(local_path_str)
    lock_path = local_path.parent / f"{local_path.name}.lock"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not source_path.is_dir():
        logger.error(f"Source model path does not exist: {source_path}")
        raise FileNotFoundError(f"Source model path does not exist: {source_path}")

    # 同一ノードでの競合を避けるため、ロックを取得
    with FileLock(str(lock_path)):
        # ローカルにモデルが既に存在するか確認
        if local_path.is_dir() and any(local_path.iterdir()):
            logger.info(f"Model already cached at {local_path}. Skipping copy.")
            return

        logger.info(f"Caching model from {source_path} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # rsyncコマンドで高速にコピー
        try:
            # `check=True`でエラー時に例外を発生させる
            subprocess.run(
                ["rsync", "-avh", "--progress", str(source_path) + "/", str(local_path)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Model successfully cached to {local_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cache model. rsync error: {e.stderr}")
            raise

# Retrieve Slurm/HPC node information from environment variables; use default values if none are found
# In a real Ray Job environment, these values should be hard‑coded or provided through other mechanisms
WANDB_ENTITY = "ken05-matuo-llm-88_llm_2025_suzuki"
WANDB_PROJECT_NAME = "competition_verl_test"
# WANDB_RUN_NAME = "qwen3-235b_multinode_grpo"
WANDB_RUN_NAME = "qwen3-8b_multinode_grpo"

# Paths
RAY_DATA_HOME = os.environ.get('RAY_DATA_HOME', f"{os.environ['HOME']}/../shareP05/train/envs/train_env_sato_grpo/deps/verl")
CKPTS_DIR = f"{RAY_DATA_HOME}/ckpt/{WANDB_PROJECT_NAME}/{WANDB_RUN_NAME}"
TRAIN_FILE = f"{os.environ['HOME']}/data/gsm8k/train.parquet"
TEST_FILE = f"{os.environ['HOME']}/data/gsm8k/test.parquet"

# SOURCE_MODEL_PATH = "/home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B"
# job_id = os.getenv("RAY_JOB_ID", "default_job")
# os.makedirs(f"/nvme12/models/{job_id}/Qwen3-235B-A22B", exist_ok=True)
# LOCAL_MODEL_PATH = f"/nvme12/models/{job_id}/Qwen3-235B-A22B"
# cache_model_locally(SOURCE_MODEL_PATH, LOCAL_MODEL_PATH)

args = [
    f"data.train_files={TRAIN_FILE}",
    f"data.val_files={TEST_FILE}",
    "data.prompt_key=question",
    "+data.response_key=answer",
    "data.truncation=left",
    f"data.max_prompt_length=1024",
    f"data.max_response_length=4096",
    f"data.train_batch_size=128",
    "data.filter_overlong_prompts=True",
    f"actor_rollout_ref.rollout.n=4",
    f"algorithm.adv_estimator=grpo",
    f"algorithm.use_kl_in_reward=False",
    f"algorithm.kl_ctrl.kl_coef=0.0",
    f"actor_rollout_ref.actor.use_kl_loss=False",
    f"actor_rollout_ref.actor.kl_loss_coef=0.0",
    f"actor_rollout_ref.actor.clip_ratio_low=0.2",
    f"actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.actor.clip_ratio_c=10.0",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    # f"actor_rollout_ref.model.path={LOCAL_MODEL_PATH}",
    "actor_rollout_ref.model.path=Qwen/Qwen3-8B",
    "actor_rollout_ref.model.lora_rank=32",
    "actor_rollout_ref.model.lora_alpha=32",
    "actor_rollout_ref.model.target_modules=all-linear",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.layered_summon=True",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
    "actor_rollout_ref.actor.optim.weight_decay=0.1",
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=8",
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    f"actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.entropy_coeff=0",
    f"actor_rollout_ref.actor.loss_agg_mode=token-mean",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8",
    f"actor_rollout_ref.rollout.tensor_model_parallel_size=8",
    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
    f"actor_rollout_ref.rollout.max_num_batched_tokens=5120",
    f"actor_rollout_ref.rollout.temperature=1.0",
    f"actor_rollout_ref.rollout.top_p=1.0",
    f"actor_rollout_ref.rollout.top_k=-1",
    f"actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
    f"actor_rollout_ref.rollout.val_kwargs.top_p=0.7",
    f"actor_rollout_ref.rollout.val_kwargs.top_k=-1",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    "actor_rollout_ref.rollout.val_kwargs.n=1",
    "actor_rollout_ref.rollout.name=vllm",
    "+actor_rollout_ref.model.dtype=bf16",
    "reward_model.enable=False",
    "+reward_model.reward_kwargs.ground_truth_key=answer",
    # "reward_model.reward_manager=dapo",
    # f"+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True",
    # f"+reward_model.reward_kwargs.overlong_buffer_cfg.len=4096",
    # f"+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.1",
    # "+reward_model.reward_kwargs.overlong_buffer_cfg.log=False",
    # f"+reward_model.reward_kwargs.max_resp_len={MAX_RESPONSE_LENGTH}",
    'trainer.logger=["console","wandb"]',
    f"trainer.project_name={WANDB_PROJECT_NAME}",
    f"trainer.experiment_name={WANDB_RUN_NAME}",
    f"trainer.n_gpus_per_node=8",
    f"trainer.nnodes=2",
    "+trainer.cluster.dist_backend=nccl",
    "trainer.val_before_train=False",
    "trainer.test_freq=10",
    "trainer.save_freq=20",
    "trainer.total_epochs=1",
    "trainer.total_training_steps=100",
    f"trainer.default_local_dir={CKPTS_DIR}",
    "trainer.resume_mode=auto",
    "trainer.log_val_generations=10",
]

from verl.trainer import main_ppo
sys.argv = ["verl.trainer.main_ppo"] + args
main_ppo.main()

# ローカルモデルパスを削除する場合は、以下のコメントアウトを外してください
# if LOCAL_MODEL_PATH and Path(LOCAL_MODEL_PATH).exists():
#     shutil.rmtree(LOCAL_MODEL_PATH)
