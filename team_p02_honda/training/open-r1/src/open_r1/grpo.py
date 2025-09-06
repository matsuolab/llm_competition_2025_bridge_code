# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
# ModelConfigは不要になったため削除
from trl import GRPOTrainer, TrlParser

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)


# =================================================================================
# QLoRAとモデル実装に関する引数を保持する、専用のカスタムクラスを定義します。
# =================================================================================
@dataclass
class ModelArguments:
    """
    Contains arguments pertaining to model loading, quantization, and implementation.
    """
    model_name_or_path: str = field(metadata={"help": "The model checkpoint for weights initialization."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether or not to allow for custom models defined on the Hub."})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "The torch dtype to use for the model (e.g. 'bfloat16', 'float16', 'auto')."})

    # QLoRA/bitsandbytes parameters
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4-bit precision"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"})
    bnb_4bit_use_double_quant: bool = field(default=False, metadata={"help": "Use double quantization"})
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"help": "Compute dtype for 4-bit base models"})
    
    # Flash Attention
    attn_implementation: Optional[str] = field(default=None, metadata={"help": "Attention implementation to use (e.g., 'flash_attention_2')"})


def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelArguments):
    """
    Main training function for GRPO with QLoRA, vLLM (colocate mode), and a single LoRA adapter.
    """
    # 1. Setup
    # ----------------------------------------------------------------
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # 2. Load model, tokenizer and dataset
    # ----------------------------------------------------------------
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16 if model_args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
    )

    model_torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        torch_dtype=model_torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = get_dataset(script_args)
    reward_funcs = get_reward_funcs(script_args)

    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    # ★★★ エラー修正: 'solution'列を保持するように変更 ★★★
    # .map()で'prompt'列を作成した後、不要になった元のプロンプト列('problem')のみを削除します。
    # これにより、accuracy_reward関数が必要とする'solution'列がデータセットに保持されます。
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns([script_args.dataset_prompt_column])


    # 3. Initialize the standard GRPO trainer
    # ----------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.do_eval else None),
        processing_class=tokenizer,
        callbacks=get_callbacks(training_args, model_args),
    )

    # 4. Training
    # ----------------------------------------------------------------
    logger.info("*** Train ***")
    checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 5. Save model and push to hub
    # ----------------------------------------------------------------
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.create_model_card(
            dataset_name=script_args.dataset_name,
            tags=["open-r1"],
        )

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
