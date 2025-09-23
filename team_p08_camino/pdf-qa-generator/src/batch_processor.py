# src/batch_processor.py
import json
import shutil
from pathlib import Path
from src.main import PDFQAGenerator

class BatchProcessor:
    """input_pdfsディレクトリ内の全PDFファイルを処理するクラス"""
    def __init__(self):
        self.gen = PDFQAGenerator()

    def run_all(self) -> None:
        input_dir = Path("input_pdfs")
        processed_dir = input_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        results = []
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print("処理対象のPDFファイルが input_pdfs ディレクトリに見つかりません。")
            return

        for pdf in pdf_files:
            print(f"🔍 PDFファイル '{pdf.name}' を処理中...")
            try:
                output_path = self.gen.process_pdf(pdf.name)
                shutil.move(str(pdf), processed_dir / pdf.name)
                print(f"✅ 完了: '{pdf.name}' -> 保存先: '{output_path}'")
                results.append({"pdf": pdf.name, "output": output_path})
            except Exception as e:
                print(f"❌ エラー: '{pdf.name}' の処理中に問題が発生しました: {e}")
                results.append({"pdf": pdf.name, "error": str(e)})

        log_path = Path("output_qa/batch_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("✅ バッチ完了")