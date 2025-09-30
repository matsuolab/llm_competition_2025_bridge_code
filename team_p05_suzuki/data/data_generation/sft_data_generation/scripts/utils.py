"""
Common utilities for vLLM and Ray initialization used across data generation scripts.
"""

import os
import time
import multiprocessing

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before importing torch or any CUDA-related libraries
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch

# Check for GPU availability
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

# Import vLLM if available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Will use API-based generation instead.")

# Import Ray if available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Some distributed features may be limited.")


def wait_for_ray_cluster(tensor_parallel_size, timeout=600):
    """Wait for Ray cluster to be ready with enough resources.
    
    Args:
        tensor_parallel_size: Number of GPUs required
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if cluster is ready, False otherwise
    """
    if not RAY_AVAILABLE:
        return False
        
    print(f"Waiting for Ray cluster to be ready (timeout: {timeout}s)...")
    print(f"Required GPUs: {tensor_parallel_size}")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Get Ray cluster resources
            resources = ray.available_resources()
            num_gpus = resources.get("GPU", 0)
            num_cpus = resources.get("CPU", 0)
            nodes = ray.nodes()
            num_nodes = len(nodes)
            
            # Get node details
            node_info = []
            for node in nodes:
                node_resources = node.get("Resources", {})
                node_gpus = node_resources.get("GPU", 0)
                node_state = node.get("Alive", False)
                node_addr = node.get("NodeManagerAddress", "unknown")
                node_info.append(f"{node_addr} (GPUs: {node_gpus}, Alive: {node_state})")
            
            print(f"Ray cluster status: {num_nodes} nodes, {num_gpus} GPUs, {num_cpus} CPUs available")
            if node_info:
                print("Nodes in cluster:")
                for info in node_info:
                    print(f"  - {info}")
            
            # Check if we have enough GPUs
            if num_gpus >= tensor_parallel_size:
                print(f"✓ Ray cluster ready with {num_gpus} GPUs (required: {tensor_parallel_size})")
                return True
            else:
                print(f"⏳ Waiting for more GPUs... (current: {num_gpus}, required: {tensor_parallel_size})")
                time.sleep(10)
                
        except Exception as e:
            print(f"Error checking Ray cluster status: {e}")
            time.sleep(10)
    
    print(f"ERROR: Timeout waiting for Ray cluster. Final status: {num_gpus} GPUs available, {tensor_parallel_size} required")
    return False


def initialize_ray(ray_head_addr=None):
    """Initialize Ray connection.
    
    Args:
        ray_head_addr: Optional Ray head address for connecting to existing cluster
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not RAY_AVAILABLE:
        return False
        
    if ray.is_initialized():
        print("Ray is already initialized")
        return True
        
    # Check for Ray head address
    if not ray_head_addr:
        ray_head_addr = os.getenv("RAY_HEAD_ADDR")
        
    if ray_head_addr:
        print(f"Connecting to existing Ray cluster at {ray_head_addr}")
        try:
            # Add 'ray://' scheme to address
            if not ray_head_addr.startswith('ray://'):
                ray_head_addr = f'ray://{ray_head_addr}'
            ray.init(
                address=ray_head_addr,
                ignore_reinit_error=True,
                object_store_memory=8 * 1024**3  # 8 GiB limit for /dev/shm
            )
            print("Successfully connected to Ray cluster")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to Ray cluster at {ray_head_addr}: {e}")
            raise RuntimeError(f"Could not connect to Ray cluster at {ray_head_addr}")
    else:
        # Single node execution - initialize Ray locally
        print("Initializing Ray locally for single-node execution")
        try:
            ray.init(
                ignore_reinit_error=True,
                object_store_memory=8 * 1024**3  # 8 GiB limit for /dev/shm  
            )
            print("Successfully initialized Ray locally")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize Ray locally: {e}")
            raise RuntimeError(f"Could not initialize Ray locally: {e}")


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
    if not config.get("use_vllm", True) or not VLLM_AVAILABLE:
        return None, None, None
    
    # Check for GPU availability
    if not CUDA_AVAILABLE or GPU_COUNT == 0:
        print("ERROR: No GPU detected. vLLM requires GPU for inference.")
        print("Switching to API mode...")
        return None, None, None
    
    print(f"GPU detected: {GPU_COUNT} device(s) available")
    
    # Initialize Ray if needed
    if not initialize_ray():
        return None, None, None
    
    # Wait for Ray cluster to have enough resources
    if not wait_for_ray_cluster(config["tensor_parallel_size"]):
        raise RuntimeError("Ray cluster not ready with sufficient resources")
    
    # Prepare vLLM configuration
    vllm_config = {
        "model": config["model_path"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "trust_remote_code": config.get("trust_remote_code", True),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.95),
        "max_model_len": config.get("max_model_len", 8192),
        "disable_custom_all_reduce": True,  # Disable custom kernels for H100 compatibility
        "tokenizer_mode": "auto",  # Use the model's tokenizer
        "enforce_eager": True,  # Force eager mode for H100+FP8 stability
    }
    
    # Handle model download and cache directory
    if config.get("model_cache_dir"):
        vllm_config["download_dir"] = config["model_cache_dir"]
    if config.get("hf_token"):
        os.environ["HF_TOKEN"] = config["hf_token"]
    
    # Add quantization if specified
    quantization_method = config.get("quantization_method")
    if quantization_method and quantization_method not in ["", "none", "None"]:
        # Check if model already has quantization config
        model_path = config["model_path"]
        model_needs_quantization = "Qwen3" in model_path or "qwen3" in model_path.lower()
        if model_needs_quantization:
            vllm_config["quantization"] = quantization_method
    
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