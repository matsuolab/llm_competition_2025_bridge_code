import sys
import importlib

# 検証対象のライブラリリスト
LIBRARIES_TO_CHECK = [
    "apex",
    "transformer_engine",
    "flash_attn",
]

def main():
    """
    指定されたライブラリリストを順にインポートし、
    すべて成功したかどうかを返す。
    """
    print("--- カスタムライブラリ インポートテスト開始 ---", flush=True)
    all_successful = True
    
    for lib_name in LIBRARIES_TO_CHECK:
        try:
            importlib.import_module(lib_name)
            print(f"✓ {lib_name}: インポート成功", flush=True)
        except ImportError as e:
            print(f"✗ {lib_name}: インポート失敗 - {e}", flush=True)
            all_successful = False
            
    print("--- インポートテスト終了 ---", flush=True)
    
    if not all_successful:
        # 一つでも失敗したら、終了コード1でスクリプトを終了
        sys.exit(1)
        
    # すべて成功したら、終了コード0で正常終了
    sys.exit(0)

if __name__ == "__main__":
    main()