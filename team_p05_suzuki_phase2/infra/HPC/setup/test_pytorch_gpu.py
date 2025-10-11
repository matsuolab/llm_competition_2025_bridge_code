import sys
import torch

def main():
    """
    PyTorchがGPUを正しく認識できるかテストし、結果を出力します。
    認識できない場合は、非ゼロの終了コードで終了します。
    """
    try:
        print(f"[INFO] PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"[SUCCESS] PyTorch can see the GPU.")
            print(f"  [INFO] GPU count: {torch.cuda.device_count()}")
            print(f"  [INFO] Current device index: {torch.cuda.current_device()}")
            print(f"  [INFO] Device name: {torch.cuda.get_device_name(0)}")
            # 成功時は終了コード 0
            sys.exit(0)
        else:
            print(f"[ERROR] PyTorch CANNOT see the GPU.")
            # 失敗時は終了コード 1
            sys.exit(1)

    except Exception as e:
        print(f"[CRITICAL] An unexpected error occurred during GPU test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()