import os
import sys
import torch
import numpy

def test_python_version():
    """env_config.shのPYTHON_VERSIONと一致するかテスト"""
    expected_version = os.environ.get("PYTHON_VERSION", "N/A")
    actual_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert actual_version == expected_version, f"Pythonバージョン不一致: 期待値={expected_version}, 実際値={actual_version}"

def test_pytorch_cuda_version():
    """env_config.shのPYTORCH_CUDA_VERSIONと一致するかテスト"""
    expected_version = os.environ.get("PYTORCH_CUDA_VERSION", "N/A")
    actual_version = torch.version.cuda
    assert actual_version == expected_version, f"PyTorchのCUDAバージョン不一致: 期待値={expected_version}, 実際値={actual_version}"

def test_numpy_version_from_requirements():
    """requirements.txtで指定されたnumpyのバージョンと一致するかテスト"""
    expected_version = "1.26.4"
    actual_version = numpy.__version__
    assert actual_version == expected_version, f"numpyバージョン不一致: 期待値={expected_version}, 実際値={actual_version}"