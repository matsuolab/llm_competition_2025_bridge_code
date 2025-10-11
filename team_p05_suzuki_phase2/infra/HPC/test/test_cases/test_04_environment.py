import os
import pytest

def test_cuda_home_is_set_and_valid():
    """環境変数 CUDA_HOME が正しく設定されているかテスト"""
    cuda_home = os.environ.get('CUDA_HOME')
    assert cuda_home is not None, "環境変数 CUDA_HOME が設定されていません"
    assert os.path.isdir(cuda_home), f"CUDA_HOME に指定されたパスはディレクトリではありません: {cuda_home}"
    # bin/nvcc の存在も確認
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    assert os.path.isfile(nvcc_path), f"nvccが見つかりません: {nvcc_path}"

def test_ld_library_path_for_hpcx():
    """LD_LIBRARY_PATHにHPCXのパスが含まれているかテスト"""
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    hpcx_home = os.environ.get('HPCX_HOME')
    
    assert hpcx_home is not None, "環境変数 HPCX_HOMEが設定されていません (module load漏れの可能性)"
    
    expected_path_part = os.path.join(hpcx_home, 'ompi/lib')
    assert expected_path_part in ld_path, f"LD_LIBRARY_PATHにHPCXのパスが含まれていません"