import os

# ---- GPU隔離設定を一番最初に（torch等のimportより前）----
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)   # このプロセスから1枚だけ見える
os.environ["VLLM_DEVICE"] = "cuda:0"                   # プロセス視点の0番を使う
os.environ["VLLM_MAX_NUM_SEQS"] = "192"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# W&B を rank0 だけに（重複run回避＆main判定の目印にもなる）
os.environ["WANDB_REQUIRE_RANK_ZERO"] = "true"
if os.environ.get("LOCAL_RANK", "0") != "0":
    os.environ["WANDB_MODE"] = "disabled"

# Fix Unsloth cache permission issue on HPC systems
# Set custom cache directory with proper permissions
user_cache_dir = os.path.expanduser('~/.unsloth_cache')
user_tmp_dir = os.path.expanduser('~/tmp')

# Set multiple environment variables to override cache locations
os.environ['UNSLOTH_CACHE_DIR'] = user_cache_dir
os.environ['TMPDIR'] = user_tmp_dir
os.environ['TEMP'] = user_tmp_dir
os.environ['TMP'] = user_tmp_dir
os.environ['PYTORCH_CACHE_DIR'] = user_cache_dir
os.environ['HF_HOME'] = user_cache_dir
os.environ['TRANSFORMERS_CACHE'] = user_cache_dir

# Unsloth のコンパイル出力先を明示し、壊れキャッシュを掃除
os.environ["UNSLOTH_COMPILE_LOCATION"] = user_cache_dir
# -----------------------------------------------

import torch
import torch.distributed as dist
import tempfile
import shutil

# Create directories if they don't exist
os.makedirs(user_cache_dir, exist_ok=True)
os.makedirs(user_tmp_dir, exist_ok=True)

# Try to clean any problematic cache files before importing
problematic_cache = '/tmp/unsloth_compiled_cache'
if os.path.exists(problematic_cache) and os.access(problematic_cache, os.W_OK):
    try:
        shutil.rmtree(problematic_cache)
        print(f"Cleaned problematic cache directory: {problematic_cache}")
    except Exception as e:
        print(f"Could not clean cache directory: {e}")

# Set tempfile to use our custom directory
tempfile.tempdir = user_tmp_dir

print(f"Setting up custom cache directories:")
print(f"  Cache Dir: {user_cache_dir}")
print(f"  Temp Dir: {user_tmp_dir}")
print(f"  Temp file dir: {tempfile.gettempdir()}")

from unsloth import FastLanguageModel
from datasets import load_dataset
import re
import langid
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import numpy as np
import argparse

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# Add argument parser for dataset selection
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with Dataset Selection")
    parser.add_argument(
        "--dataset-mode", 
        type=str, 
        choices=["original", "huggingface"], 
        default="original",
        help="Choose dataset mode: 'original' for DAPO-Math-17k-Processed, 'huggingface' for team-suzuki/GRPO_000_origin"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token for private datasets"
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="Weights & Biases API key for logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="grpo-training",
        help="Weights & Biases project name (default: grpo-training)"
    )
    parser.add_argument(
        "--upload-model",
        type=str,
        default=None,
        help="Model name to upload to HuggingFace after training (e.g., 'team-suzuki/grpo-model-v1')"
    )
    parser.add_argument(
        "--model-private",
        action="store_true",
        help="Make the uploaded model private on HuggingFace Hub"
    )
    return parser.parse_args()

# Parse arguments
args = parse_args()

print(f"Process LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Print usage information
print("="*50)
print("GRPO Training Script - Dataset Mode Selection")
print("="*50)
print(f"Dataset Mode: {args.dataset_mode}")
if args.dataset_mode == "huggingface":
    print(f"HF Token: {'Provided' if args.hf_token else 'Not provided'}")
print(f"WandB Project: {args.wandb_project}")
print(f"WandB API Key: {'Provided' if args.wandb_api_key else 'Not provided'}")
print(f"Upload Model: {args.upload_model if args.upload_model else 'No upload'}")
if args.upload_model:
    print(f"Model Privacy: {'Private' if args.model_private else 'Public'}")
print(f"Unsloth Cache Directory: {os.environ.get('UNSLOTH_CACHE_DIR', 'Default')}")
print(f"Temp Directory: {os.environ.get('TMPDIR', 'Default')}")
print("="*50)

# Setup WandB if API key is provided
wandb_report = "none"
if args.wandb_api_key:
    try:
        import wandb
        wandb.login(key=args.wandb_api_key)
        wandb_report = "wandb"
        print(f"WandB login successful. Project: {args.wandb_project}")
    except ImportError:
        print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")
        wandb_report = "none"
    except Exception as e:
        print(f"Warning: WandB login failed: {e}")
        wandb_report = "none"

# マルチGPU対応: distributed training の初期化
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(0)          # プロセス視点の0番（=物理 local_rank）
    dist.init_process_group(backend="nccl")
    device_map = "auto"               # 1枚しか見えてないのでOK
    print(f"Initialized distributed training on GPU {local_rank} (physical GPU {local_rank})")
else:
    # ここから GPU を初期化
    device_map = "auto"
    print("Single GPU training mode")

# 以降で dtype 判定や FastLanguageModel.from_pretrained を呼ぶ
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-0528-Qwen3-8B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.50, # 0.7 → 0.50 に下げてOOM回避
    dtype = dtype,
    device_map = "cuda:0",   # 明示
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

reasoning_start = None
reasoning_end = None
user_token = None
assistant_token = None

for token in tokenizer.get_added_vocab().keys():
    if "think" in token and "/" in token:
        reasoning_end = token
    elif "think" in token:
        reasoning_start = token
    elif "user" in token:
        user_token = token
    elif "assistant" in token:
        assistant_token = token

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
You must think in Bahasa Indonesia."""
system_prompt

print(tokenizer.apply_chat_template([
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : f"<think>I think it's 2.2</think>2"},
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : f"<think>I think it's 2.2</think>2"},
], tokenize = False, add_generation_prompt = True))

# Dataset loading with mode selection
def load_dataset_by_mode(mode, hf_token=None):
    """Load dataset based on selected mode"""
    if mode == "original":
        print("Loading original DAPO-Math-17k-Processed dataset...")
        dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        return dataset, "prompt", "solution"
    elif mode == "huggingface":
        print("Loading HuggingFace team-suzuki/GRPO_000_origin dataset...")
        try:
            # Try to load the dataset with authentication if token provided
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
            
            dataset = load_dataset("team-suzuki/GRPO_000_origin", split="train")
            
            # Check dataset structure and adapt column names
            print(f"Dataset columns: {dataset.column_names}")
            print(f"First sample keys: {list(dataset[0].keys())}")
            
            # Common column name mappings
            prompt_col = None
            solution_col = None
            
            for col in dataset.column_names:
                if col.lower() in ["prompt", "question", "input", "text"]:
                    prompt_col = col
                elif col.lower() in ["solution", "answer", "output", "target", "label"]:
                    solution_col = col
            
            if prompt_col is None or solution_col is None:
                print(f"Warning: Could not automatically detect column names.")
                print(f"Available columns: {dataset.column_names}")
                print(f"Using first column as prompt, second as solution")
                prompt_col = dataset.column_names[0]
                solution_col = dataset.column_names[1] if len(dataset.column_names) > 1 else dataset.column_names[0]
            
            print(f"Using columns - Prompt: '{prompt_col}', Solution: '{solution_col}'")
            return dataset, prompt_col, solution_col
            
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            print("Falling back to original dataset...")
            dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
            return dataset, "prompt", "solution"
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

# Load dataset based on selected mode
dataset, prompt_column, solution_column = load_dataset_by_mode(args.dataset_mode, args.hf_token)
print(f"Dataset loaded successfully! Size: {len(dataset)}")

# Display sample data
print(f"\nSample prompt ({prompt_column}):")
print(dataset[0][prompt_column])
print(f"\nSample solution ({solution_column}):")
print(dataset[0][solution_column])

def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text

# Test extraction function with sample data
print(f"\nTesting answer extraction:")
print(extract_hash_answer(dataset[0][solution_column]))

# Map dataset with dynamic column names
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x[prompt_column]},
    ],
    "answer": extract_hash_answer(x[solution_column]),
})
dataset[0]

# Add optional EOS token matching
solution_end_regex = rf"{reasoning_end}(.*)"

match_format = re.compile(solution_end_regex, re.DOTALL)
match_format

match_format.findall(
    "Let me think!</think>"\
    f"Hence, the solution is 2.",
)

match_format.findall(
    "<think>Let me think!</think>"\
    f"\n\nHence, the solution is 2",
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <think> since we always prepend it!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                else: score -= 2.5 # Penalize wrong answers
            except:
                score -= 4.5 # Penalize
        scores.append(score)
    return scores

match_numbers = re.compile(
    r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
print(match_numbers.findall("  0.34  "))
print(match_numbers.findall("  123,456  "))
print(match_numbers.findall("  -0.234  "))
print(match_numbers.findall("17"))

def get_lang(text: str) -> str:
    if not text:
        return "und"
    lang, _ = langid.classify(text)
    return lang


print(get_lang("Hello, How are you")) # This should return en
print(get_lang("Aku berpikir kalau aku adalah kamu")) # This should return id
print(get_lang("我在这里")) # This should return zh

def format_and_language_reward_func(completions, **kwargs):
    scores = []

    for completion_item in completions:
        if not completion_item or not isinstance(completion_item[0], dict) or "content" not in completion_item[0]:
            scores.append(-5.0)
            print(f"Warning: Malformed completion item, assigning default low score: {completion_item}")
            continue

        content = completion_item[0]["content"]

        lang = get_lang(content)

        if lang == 'id':
            score = 5.0
        elif lang == 'en':
            score = -3.0
        elif lang == 'zh':
            score = -3.0
        else:
            score = -5.0

        scores.append(score)

    return scores

prompts = [
    [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
    [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
]
completions = [
    [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
    [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
]
format_and_language_reward_func(prompts=prompts, completions=completions)

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}"
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    return scores

tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# Filter only samples smaller than 90% max length
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1 # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer

# DDP状態の確定チェック
import torch.distributed as dist
print("DDP init:", dist.is_initialized())
if dist.is_initialized():
    print("WORLD_SIZE:", dist.get_world_size(), "RANK:", dist.get_rank())

training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = wandb_report, # Use WandB if available, otherwise none
    output_dir = "outputs",
    # マルチGPU対応の設定
    ddp_find_unused_parameters = False,
    # ddp_backend = "nccl",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# Initialize WandB run if enabled
if wandb_report == "wandb" and (not "LOCAL_RANK" in os.environ or os.environ.get("LOCAL_RANK") == "0"):
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "dataset_mode": args.dataset_mode,
                "max_seq_length": max_seq_length,
                "lora_rank": lora_rank,
                "temperature": training_args.temperature,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
                "warmup_ratio": training_args.warmup_ratio,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "optimizer": training_args.optim,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "num_generations": training_args.num_generations,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
                "max_steps": training_args.max_steps,
                "save_steps": training_args.save_steps,
            }
        )
        print(f"WandB run initialized for project: {args.wandb_project}")
    except Exception as e:
        print(f"Warning: Failed to initialize WandB run: {e}")
        training_args.report_to = "none"

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
        format_and_language_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# Trainer 生成後にAccelerateの状態を確認
from accelerate.state import AcceleratorState
st = AcceleratorState()
print("Accelerate:", st.distributed_type, "num_processes:", st.num_processes, "is_main:", st.is_main_process)

trainer.train()

# Upload model to HuggingFace Hub if requested (only on main process)
if args.upload_model and (not "LOCAL_RANK" in os.environ or os.environ.get("LOCAL_RANK") == "0"):
    print(f"\n{'='*50}")
    print("Uploading trained model to HuggingFace Hub...")
    print(f"Model name: {args.upload_model}")
    print(f"Privacy: {'Private' if args.model_private else 'Public'}")
    print(f"{'='*50}")
    
    try:
        # Ensure HuggingFace authentication
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("Authenticated with HuggingFace Hub using provided token")
        
        # Save the model and tokenizer
        model.save_pretrained_merged(
            "final_model",
            tokenizer,
            save_method="merged_16bit",
        )
        
        # Upload to HuggingFace Hub
        model.push_to_hub_merged(
            args.upload_model,
            tokenizer,
            save_method="merged_16bit",
            private=args.model_private,
            token=args.hf_token,
        )
        
        print(f"✓ Model successfully uploaded to: https://huggingface.co/{args.upload_model}")
        
        # Also save locally for backup
        print("Saving model locally as backup...")
        model.save_pretrained("outputs/final_model")
        tokenizer.save_pretrained("outputs/final_model")
        print("✓ Model saved locally in outputs/final_model")
        
    except Exception as e:
        print(f"✗ Failed to upload model: {e}")
        print("Saving model locally instead...")
        try:
            model.save_pretrained("outputs/final_model")
            tokenizer.save_pretrained("outputs/final_model")
            print("✓ Model saved locally in outputs/final_model")
        except Exception as save_error:
            print(f"✗ Failed to save model locally: {save_error}")

# マルチGPU環境でのクリーンアップ
if "LOCAL_RANK" in os.environ and dist.is_initialized():
    dist.destroy_process_group()

print(f"\n{'='*50}")
print("Training completed!")
if args.upload_model and (not "LOCAL_RANK" in os.environ or os.environ.get("LOCAL_RANK") == "0"):
    print(f"Check your uploaded model at: https://huggingface.co/{args.upload_model}")
print(f"{'='*50}")

# Usage Examples:
# 
# Single GPU Training:
# 1. Run with original dataset (default):
#    python grpo_main.py
#    python grpo_main.py --dataset-mode original
#
# 2. Run with HuggingFace dataset (public):
#    python grpo_main.py --dataset-mode huggingface
#
# 3. Run with HuggingFace dataset (private, requires token):
#    python grpo_main.py --dataset-mode huggingface --hf-token YOUR_HF_TOKEN
#
# 4. Run with WandB logging:
#    python grpo_main.py --wandb-api-key YOUR_WANDB_KEY --wandb-project my-project
#
# Multi-GPU Training:
# 5. Using torchrun (recommended):
#    torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --hf-token YOUR_HF_TOKEN
#    torchrun --nproc_per_node=4 grpo_main.py --dataset-mode original --wandb-api-key YOUR_WANDB_KEY
#
# 6. Using python -m torch.distributed.launch (legacy):
#    python -m torch.distributed.launch --nproc_per_node=2 grpo_main.py --dataset-mode huggingface
#    python -m torch.distributed.launch --nproc_per_node=4 grpo_main.py --dataset-mode original
#
# 7. Upload trained model to HuggingFace Hub:
#    torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --upload-model team-suzuki/grpo-model-v1 --hf-token YOUR_HF_TOKEN
#
# 8. Upload private model to HuggingFace Hub:
#    torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --upload-model team-suzuki/grpo-model-v1 --hf-token YOUR_HF_TOKEN --model-private
#
# 9. Complete multi-GPU example with all options:
#    torchrun --nproc_per_node=4 grpo_main.py --dataset-mode huggingface --hf-token YOUR_HF_TOKEN \
#                        --wandb-api-key YOUR_WANDB_KEY --wandb-project grpo-experiment \
#                        --upload-model team-suzuki/grpo-model-v1 --model-private
#
# 10. Set tokens via environment variables:
#     export HF_TOKEN=your_hf_token_here
#     export WANDB_API_KEY=your_wandb_key_here
#     torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --hf-token $HF_TOKEN \
#                         --wandb-api-key $WANDB_API_KEY --wandb-project my-project \
#                         --upload-model team-suzuki/grpo-model-v1
#
# 11. For HPC systems with SLURM:
#     srun --gpus=4 --nodes=1 --ntasks-per-node=4 python -m torch.distributed.launch \
#          --nproc_per_node=4 --nnodes=1 --node_rank=0 grpo_main.py \
#          --dataset-mode huggingface --hf-token YOUR_HF_TOKEN
#
# 12. For HPC systems with permission issues:
#     export UNSLOTH_CACHE_DIR=~/.unsloth_cache
#     export TMPDIR=~/tmp
#     mkdir -p ~/.unsloth_cache ~/tmp
#     torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --hf-token YOUR_HF_TOKEN \
#                         --wandb-api-key YOUR_WANDB_KEY --wandb-project hpc-experiment \
#                         --upload-model team-suzuki/grpo-hpc-model
#
# 13. If you still get permission errors, try cleaning the cache:
#     rm -rf /tmp/unsloth_compiled_cache/ (if you have permission)
#     or contact your HPC administrator
#
# Multi-GPU Notes:
# - Use torchrun for better process management and automatic environment setup
# - Only the main process (LOCAL_RANK=0) will upload models and initialize wandb
# - All GPUs will participate in training with data parallelism
# - Memory usage is distributed across GPUs
# - Training time should decrease roughly linearly with GPU count
#
# Model Upload Notes:
# - The --upload-model option requires a valid HuggingFace token (--hf-token)
# - Model name should follow the format: username/model-name or organization/model-name
# - Use --model-private to make the model private on HuggingFace Hub
# - The model will be saved both locally (outputs/final_model) and uploaded to HuggingFace
# - If upload fails, the model will still be saved locally
# - Only the main process (LOCAL_RANK=0) will handle model upload in multi-GPU training
#
# Note: Make sure to install required packages before using:
#    pip install wandb huggingface_hub
