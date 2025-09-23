# src/pdf_extractor.py
import subprocess, os
from typing import List, Dict

class PDFExtractor:
    def __init__(self, nougat_path="nougat"):
        self.nougat = nougat_path

    def extract(self, pdf_path: str) -> List[Dict]:
        """PDF ファイルパスを Nougat に直接渡す"""
        try:
            print(f"Nougat OCRを実行中 ... {os.path.basename(pdf_path)}")
            # stderr=subprocess.STDOUT で標準エラーもキャプチャ
            out = subprocess.check_output(
                [self.nougat, pdf_path, '--no-skipping'], # no-skippingを追加して品質向上
                text=True,
                timeout=300, # タイムアウトを延長
                stderr=subprocess.STDOUT
            )
            print("Nougat出力先頭 500 文字:\n", out[:500])
            if out.strip():
                return [{"type": "ocr", "page": 0, "content": out.strip()}]
            else:
                return []
        except subprocess.CalledProcessError as e:
            print("Nougatエラー出力:\n", e.output)
            return []
        except subprocess.TimeoutExpired as e:
            print(f"Nougat処理がタイムアウトしました: {pdf_path}")
            print("エラー出力:\n", e.output)
            return []