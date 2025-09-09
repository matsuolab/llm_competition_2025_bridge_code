import os


class Config:
    
    def __init__(self):
        self.model_name = "Qwen/Qwen3-32B"
        self.dataset_name = "LLMcompe-Team-Watanabe/deepscaler-openr1-harvard-math-reasoning"
        self.max_seq_length = 8192
        
        # Training settings
        self.num_epochs = 1
        self.batch_size_per_gpu = 1
        self.gradient_accumulation_steps = 16
        self.learning_rate = 5e-6
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        # LoRA settings
        self.use_lora = False
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = "all-linear"
        
        # Performance settings
        self.bf16 = True  # Use bfloat16 mixed precision
        self.tf32 = True  # Enable TensorFloat-32 for training on Ampere GPUs
        self.use_flash_attention_2 = True  # Use Flash Attention 2 for faster training
        self.gradient_checkpointing = True  # Enable gradient checkpointing for memory efficiency
        self.use_liger_kernel = True  # Use Liger kernel for faster training
        
        # Evaluation and saving
        self.save_steps = 1000
        self.logging_steps = 1
        
        # Output
        self.output_dir = "YOUR/WORKING/DIRECTORY/PATH/output"  # Change this to your desired output directory
        
        # WandB
        self.wandb_project = "Qwen-32B-fft"
        self.wandb_run_name = "data-deepscaler-openr1-havard-40k-1ep-lr5e6-8192"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)