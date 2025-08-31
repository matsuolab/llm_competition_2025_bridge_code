# src/main.py
import json
from pathlib import Path
from src.pdf_extractor import PDFExtractor
from src.qa_generator import QAGenerator
from src.quality_checker import QualityChecker
from config.settings import Settings

class PDFQAGenerator:
    def __init__(self):
        self.settings = Settings()
        self.extractor = PDFExtractor()
        self.qa_gen = QAGenerator()
        self.checker = QualityChecker()

    def process_pdf(self, pdf_name: str)-> str:
        pdf_path = Path(self.settings.INPUT_DIR) / pdf_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")
        content = self.extractor.extract(str(pdf_path))
        if not content:
            raise ValueError(f"PDF '{pdf_name}' から抽出できるコンテンツがありませんでした。")
        print(f"[INFO] 抽出完了: {len(content)} blocks")

        qa_pairs = self.qa_gen.generate_qa_pairs(content)
        validated_qa_pairs = self.checker.validate_qa_pairs(qa_pairs)

        out_path = Path(self.settings.OUTPUT_DIR) / f"{Path(pdf_name).stem}_qa.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(validated_qa_pairs, f, ensure_ascii=False, indent=2)

        return str(out_path)