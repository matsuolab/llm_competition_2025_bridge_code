import logging
import os
import sys
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from sft_config import Config
from data_processor import DataProcessor
import wandb

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def load_and_process_data(config: Config, tokenizer):
    """データセットの読み込みと処理"""
    logger.info(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(config.dataset_name)
    train_dataset = dataset["train"]
    
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    
    # DataProcessorを使用してデータセットを処理
    processor = DataProcessor(tokenizer)
    train_dataset = processor.process_dataset(train_dataset)
    train_dataset = processor.validate_dataset(train_dataset)
    
    return train_dataset

def load_model_and_tokenizer(config: Config):
    """Load tokenizer & model - DeepSpeed ZeRO-3 compatible initialization"""
    logger.info(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DeepSpeed ZeRO-3対応のモデル初期化
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Model & tokenizer ready (DeepSpeed will handle device placement)")
    return model, tokenizer


def create_trainer(model, tokenizer, train_dataset, config: Config):
    """Create SFT trainer"""
    
    num_train_samples = len(train_dataset)
    
    # 環境変数から取得
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    logger.info(f"Distributed info: world_size={world_size}, local_rank={local_rank}")
    
    # LoRA configuration
    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")
        
    # Create training arguments
    training_args = SFTConfig(
        # Output
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimizer
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim="adamw_torch_fused",
        use_liger_kernel=config.use_liger_kernel,
        
        # Mixed precision
        bf16=config.bf16,
        tf32=config.tf32,
        
        # Checkpointing
        gradient_checkpointing=config.gradient_checkpointing,
        
        save_strategy="steps",
        save_steps=config.save_steps,
        
        # Logging
        logging_steps=config.logging_steps,
        report_to="wandb",
        run_name=config.wandb_run_name,
        
        # SFT specific
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={
            "num_of_sequences": num_train_samples,
        },

        dataset_num_proc=128,  
        ddp_timeout=10800,
        dataloader_num_workers=64,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        
        # Seed
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
    )
    
    return trainer


def main():
    """Main training function"""
    # Load config
    config = Config()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(
            entity="LLMcompe-Team-Watanabe",
            project=config.wandb_project,
            name=config.wandb_run_name
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load and process data
    train_dataset = load_and_process_data(config, tokenizer)
        
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, config)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()