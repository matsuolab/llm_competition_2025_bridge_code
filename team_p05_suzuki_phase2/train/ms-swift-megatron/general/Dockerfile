FROM nvcr.io/nvidia/pytorch:25.05-py3

# Disable PyTorch compilation to avoid InductorError
ENV TORCH_COMPILE=0
ENV TORCHDYNAMO_DISABLE=1

RUN pip install --upgrade pip

# Install core dependencies first
RUN pip install pybind11

# Use the original PyTorch from the base image instead of reinstalling
# This ensures CUDA support is properly maintained
# Install ms-swift with proper dependencies
RUN pip install ms-swift==3.8.1

# Install additional training dependencies
RUN pip install deepspeed  # for multi-GPU training
RUN pip install liger-kernel  # to save GPU memory resources
# RUN pip install flash-attn==2.7.4.post1 --no-build-isolation  # required for packing
RUN pip install flash-attn --no-build-isolation  # required for packing
RUN pip install wandb

# Install complete Megatron-LM repository
# Clone the full repository to get megatron.training module
RUN git clone --branch core_r0.13.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM

# Install Megatron-LM in development mode and ensure Python path
RUN cd /workspace/Megatron-LM && pip install -e . && \
    echo "/workspace/Megatron-LM" >> /usr/local/lib/python3.12/dist-packages/megatron.pth

# If you are using multi-node training, please additionally set the `MODELSCOPE_CACHE` environment variable to a shared storage path.
# This will ensure that the dataset cache is shared, thereby speeding up preprocessing.
ENV MODELSCOPE_CACHE='/workspace/shared'

# Megatron-LM
# The training module in the dependent library Megatron-LM will be cloned and installed by swift via `git clone`. Alternatively, you can use the environment variable `MEGATRON_LM_PATH` to point to the path of an already downloaded repository (in offline environments, use the [core_r0.13.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.13.0)).
ENV MEGATRON_LM_PATH='/workspace/Megatron-LM'

# Fix CUDA library paths
RUN ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
RUN ln -s /usr/local/cuda/compat/lib.real /usr/local/cuda/compat/lib

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/compat/lib:${LD_LIBRARY_PATH}

# Verify CUDA installation and PyTorch CUDA support
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"

# Verify Megatron-LM installation
RUN python -c "import sys; print('Python path:'); [print(p) for p in sys.path]; print('\\nMegatron-LM directory contents:'); import os; print(os.listdir('/workspace/Megatron-LM') if os.path.exists('/workspace/Megatron-LM') else 'Directory not found'); print('\\nMegatron directory contents:'); print(os.listdir('/workspace/Megatron-LM/megatron') if os.path.exists('/workspace/Megatron-LM/megatron') else 'Megatron directory not found'); print('\\nTrying to import megatron.training...'); import megatron.training; print('megatron.training imported successfully')"