import os
import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPUが利用できないためスキップ")
class TestGpuOperations:
    def test_cuda_is_available(self):
        """PyTorchからGPUが認識できるかテスト"""
        assert torch.cuda.is_available(), "torch.cuda.is_available()がFalseを返しました"

    def test_gpu_device_properties(self):
        """GPUデバイスのアーキテクチャが設定と一致するかテスト"""
        assert torch.cuda.device_count() >= 1, "利用可能なGPUがありません"
        
        expected_arch_str = os.environ.get("TORCH_CUDA_ARCH_LIST", "N/A")
        major, minor = torch.cuda.get_device_capability(0)
        actual_arch_str = f"{major}.{minor}"
        
        assert actual_arch_str == expected_arch_str, \
            f"GPUアーキテクチャ不一致: 期待値={expected_arch_str}, 実際値={actual_arch_str}"

    def test_tensor_operations_on_gpu(self):
        """GPU上での基本的なテンソル演算がエラーなく実行できるかテスト"""
        try:
            a = torch.randn(16, 16, device='cuda')
            b = torch.randn(16, 16, device='cuda')
            c = torch.matmul(a, b)
            # CPUに結果を戻すことで、CUDAカーネルの実行を同期・完了させる
            c_cpu = c.cpu()
            assert c_cpu.shape == (16, 16)
        except Exception as e:
            pytest.fail(f"GPU上でのテンソル演算で予期せぬエラーが発生しました: {e}")