# run_batch.py
from src.batch_processor import BatchProcessor

if __name__ == "__main__":
    print("--- バッチ処理を開始 ---")
    batch_processor = BatchProcessor()
    batch_processor.run_all()
    print("\n--- バッチ処理の完了 ---")