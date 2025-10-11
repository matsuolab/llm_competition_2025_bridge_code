"""
Common utilities for vLLM and Ray initialization used across data generation scripts.
"""

import os
import time
import multiprocessing
import gc

# Set memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '1'  # Reduce thread overhead
os.environ['MALLOC_ARENA_MAX'] = '1'  # Reduce memory fragmentation

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before importing torch or any CUDA-related libraries
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Lazy torch import to save memory
torch = None
CUDA_AVAILABLE = False
GPU_COUNT = 0

def _check_cuda():
    """Lazy check for CUDA availability."""
    global torch, CUDA_AVAILABLE, GPU_COUNT
    if torch is None:
        gc.collect()  # Clean memory before import
        try:
            import torch as _torch
            torch = _torch
            CUDA_AVAILABLE = torch.cuda.is_available()
            GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
        except MemoryError:
            print("Insufficient memory to import torch.")
            CUDA_AVAILABLE = False
            GPU_COUNT = 0
    return CUDA_AVAILABLE

# Lazy vLLM import to save memory
VLLM_AVAILABLE = False
LLM = None
SamplingParams = None

def _import_vllm():
    """Lazy import vLLM when needed."""
    global VLLM_AVAILABLE, LLM, SamplingParams
    if not VLLM_AVAILABLE:
        try:
            gc.collect()  # Clean memory before import
            from vllm import LLM as _LLM, SamplingParams as _SamplingParams
            LLM = _LLM
            SamplingParams = _SamplingParams
            VLLM_AVAILABLE = True
        except ImportError:
            VLLM_AVAILABLE = False
            print("vLLM not available. Will use API-based generation instead.")
        except MemoryError:
            VLLM_AVAILABLE = False
            print("Insufficient memory to import vLLM.")
    return VLLM_AVAILABLE

# Ray is completely disabled - use multiprocessing only
RAY_AVAILABLE = False


def wait_for_ray_cluster(tensor_parallel_size, timeout=600):
    """Ray cluster wait function - disabled (always returns False)."""
    print("Ray cluster waiting disabled - using multiprocessing backend")
    return False


def initialize_ray(ray_head_addr=None):
    """Ray initialization function - disabled (always returns False)."""
    print("Ray initialization disabled - using multiprocessing backend")
    return False


def initialize_vllm(config):
    """Initialize vLLM model and sampling parameters.
    
    Args:
        config: Dictionary containing vLLM configuration parameters:
            - model_path: Path to model or HuggingFace model ID
            - tensor_parallel_size: Number of GPUs to use
            - use_vllm: Whether to use vLLM
            - trust_remote_code: Whether to trust remote code
            - gpu_memory_utilization: GPU memory utilization fraction
            - max_model_len: Maximum model context length
            - quantization_method: Optional quantization method
            - dtype: Data type for model
            - use_lora: Whether to use LoRA
            - lora_adapter_path: Path to LoRA adapter
            - model_cache_dir: Directory for model cache
            - hf_token: HuggingFace token
            
    Returns:
        tuple: (llm_vllm, sampling_params_generator, sampling_params_solver) or (None, None, None) if failed
    """
    if not config.get("use_vllm", True):
        return None, None, None
    
    # Lazy import vLLM to save memory
    if not _import_vllm():
        return None, None, None
    
    # Check for GPU availability
    if not _check_cuda() or GPU_COUNT == 0:
        print("ERROR: No GPU detected. vLLM requires GPU for inference.")
        print("Switching to API mode...")
        return None, None, None
    
    print(f"GPU detected: {GPU_COUNT} device(s) available")
    
    # Skip Ray initialization for single-node execution
    use_ray = os.environ.get("VLLM_USE_RAY", "1") != "0"
    if use_ray:
        # Initialize Ray if needed
        if not initialize_ray():
            return None, None, None
        
        # Wait for Ray cluster to have enough resources
        print("Waiting for Ray cluster resources...")
        if not wait_for_ray_cluster(config["tensor_parallel_size"]):
            raise RuntimeError("Ray cluster not ready with sufficient resources")
        print("Ray cluster is ready")
    else:
        print("Ray disabled - using multiprocessing backend for vLLM")
        # Set environment variable to suppress distributed warnings
        os.environ["VLLM_SUPPRESS_DISTRIBUTED_WARNINGS"] = "1"
    
    # Prepare vLLM configuration
    vllm_config = {
        "model": config["model_path"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "trust_remote_code": config.get("trust_remote_code", True),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.95),
        "max_model_len": config.get("max_model_len", 8192),
        "disable_custom_all_reduce": True,  # Disable custom kernels for H100 compatibility
        "tokenizer_mode": "auto",  # Use the model's tokenizer
        "enforce_eager": config.get("enforce_eager", True),  # Allow override for AWQ models
        "distributed_executor_backend": "mp" if not use_ray else "ray",  # Use multiprocessing if Ray disabled
        "swap_space": config.get("swap_space", 4),  # Default 4GB, AWQ models may need more
        "max_num_seqs": config.get("max_num_seqs", 512),  # Maximum number of sequences
    }
    
    # Handle model download and cache directory
    if config.get("model_cache_dir"):
        vllm_config["download_dir"] = config["model_cache_dir"]
    if config.get("hf_token"):
        os.environ["HF_TOKEN"] = config["hf_token"]
    
    # Add quantization if specified
    quantization_method = config.get("quantization_method")
    if quantization_method and quantization_method not in ["", "none", "None"]:
        # Apply quantization if specified
        vllm_config["quantization"] = quantization_method
        
        # Add expert parallel for AWQ models with 8 GPUs (required for Qwen3 MoE models)
        if quantization_method in ["awq", "awq_marlin"] and config.get("tensor_parallel_size", 1) == 8:
            vllm_config["enable_expert_parallel"] = True
            print("Enabling expert parallel for AWQ model with 8 GPUs")
    
    # Set dtype
    dtype_str = config.get("dtype", "float16")
    if dtype_str == "float16":
        vllm_config["dtype"] = torch.float16
    elif dtype_str == "bfloat16":
        vllm_config["dtype"] = torch.bfloat16
    else:
        vllm_config["dtype"] = torch.float32
    
    # Add LoRA support if enabled
    if config.get("use_lora") and config.get("lora_adapter_path"):
        vllm_config["enable_lora"] = True
        vllm_config["max_lora_rank"] = 64  # Adjust as needed
        
    try:
        print(f"Creating vLLM instance with config: tensor_parallel_size={vllm_config['tensor_parallel_size']}, model={vllm_config['model']}")
        print("This may take a few minutes for large models...")
        
        # Ensure vLLM uses the existing Ray instance
        import ray
        if ray.is_initialized():
            print("Ray is already initialized, vLLM will use existing instance")
        
        llm_vllm = LLM(**vllm_config)
        print("vLLM initialized successfully")
        
        # Check GPU capability for FP8 (requires Compute Capability >= 9.0)
        if quantization_method == "fp8" and CUDA_AVAILABLE:
            device_cap = torch.cuda.get_device_capability(0)
            if device_cap[0] < 9:
                raise RuntimeError(f"FP8 quantization requires GPU with Compute Capability >= 9.0 (H100/Ada). "
                                 f"Current GPU has Compute Capability {device_cap[0]}.{device_cap[1]}")
        
        # Initialize sampling parameters for problem generation
        sampling_params_generator = SamplingParams(
            temperature=config.get("generation_temperature", 0.7),
            max_tokens=config.get("generation_max_tokens", 4096),
            top_p=config.get("generation_top_p", 0.95),
            presence_penalty=config.get("generation_presence_penalty", 0.0),
            frequency_penalty=config.get("generation_frequency_penalty", 0.0),
        )
        
        # Initialize sampling parameters for problem solving
        sampling_params_solver = SamplingParams(
            temperature=config.get("solver_temperature", 0.1),
            max_tokens=config.get("solver_max_tokens", 8192),
            top_p=config.get("solver_top_p", 0.95),
            presence_penalty=config.get("solver_presence_penalty", 0.0),
            frequency_penalty=config.get("solver_frequency_penalty", 0.0),
        )
        
        return llm_vllm, sampling_params_generator, sampling_params_solver
        
    except Exception as e:
        print(f"ERROR: Failed to initialize vLLM: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Could not initialize vLLM: {e}")


def cleanup_vllm(llm_vllm=None):
    """Clean up vLLM resources and shutdown properly.
    
    Args:
        llm_vllm: vLLM instance to cleanup (optional)
    """
    import gc
    
    # If using multiprocessing backend, properly terminate worker processes
    if llm_vllm is not None:
        try:
            # Clean up vLLM instance
            del llm_vllm
        except Exception as e:
            print(f"Warning: Error during vLLM cleanup: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Clean up torch distributed if initialized (with timeout)
    if torch is not None:
        try:
            if torch.distributed.is_initialized():
                # Set a short timeout for cleanup
                import signal
                import time

                def timeout_handler(signum, frame):
                    raise TimeoutError("Process group cleanup timed out")

                # Try to destroy with 5 second timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                try:
                    torch.distributed.destroy_process_group()
                    print("PyTorch distributed process group destroyed")
                except TimeoutError:
                    print("Warning: Process group cleanup timed out, skipping...")
                finally:
                    signal.alarm(0)  # Cancel alarm
        except Exception as e:
            print(f"Warning: Error destroying process group: {e}")
    
    # Clean up CUDA cache if available
    if torch is not None and CUDA_AVAILABLE:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Error cleaning CUDA cache: {e}")