"""
テキスト正規化処理（英語専用）
NFKC正規化、空白・改行正規化、文分割など
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """テキスト正規化クラス（英語専用）"""
    
    def __init__(self):
        """初期化"""
        # 改行・空白正規化パターン
        self.whitespace_patterns = [
            (r'\r\n', '\n'),           # CRLF → LF
            (r'\r', '\n'),             # CR → LF
            (r'[ \t]+', ' '),          # 連続空白 → 単一空白
            (r'\n{3,}', '\n\n'),       # 3連続改行以上 → 2改行
            (r'^\s+|\s+$', ''),        # 前後の空白削除
        ]
        
        # 箇条書きパターン
        self.bullet_patterns = [
            r'^[-*•·◆▪▫◇○●□■]\s*',    # 各種箇条書き記号
            r'^\d+[.)]\s*',              # 番号付きリスト
            r'^[a-zA-Z][.)]\s*',         # アルファベットリスト
        ]
    
    def normalize(self, text: str) -> str:
        """
        テキストの総合正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        if not text:
            return ""
        
        # NFKC正規化（Unicode正規化）
        text = unicodedata.normalize('NFKC', text)
        
        # 共通正規化処理
        text = self.normalize_whitespace(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_bullets(text)
        text = self.normalize_quotes(text)
        
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        """
        空白・改行の正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        for pattern, replacement in self.whitespace_patterns:
            text = re.sub(pattern, replacement, text)
        
        # 各行の前後空白削除
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        句読点の正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        # 重複句読点の削除（省略記号は除く）
        text = re.sub(r'(?<!\.)([.!?]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        # 句読点前後の空白調整
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)  # 前の空白削除
        
        # 文末句読点後の空白を確保（改行は除く）
        text = re.sub(r'([.!?])(?=[A-Z])', r'\1 ', text)
        
        return text
    
    def normalize_quotes(self, text: str) -> str:
        """
        引用符の正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        # スマートクォートを通常の引用符に
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # バッククォートはそのまま保持（コード用）
        
        return text
    
    def normalize_bullets(self, text: str) -> str:
        """
        箇条書きの正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # 箇条書き記号を統一
            for pattern in self.bullet_patterns:
                if re.match(pattern, line):
                    # 記号を "- " に統一
                    line = re.sub(pattern, '- ', line)
                    break
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def split_sentences(self, text: str) -> List[str]:
        """
        文分割（英語用）
        
        Args:
            text: 入力テキスト
            
        Returns:
            文のリスト
        """
        # 改行で分割
        paragraphs = text.split('\n')
        
        sentences = []
        for para in paragraphs:
            if not para:
                continue
            
            # 略語を保護（Dr., Mr., etc.）
            para = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', para)
            para = re.sub(r'\b(Inc|Ltd|Co|Corp)\.\s*', r'\1<DOT> ', para)
            para = re.sub(r'\b(vs|etc|i\.e|e\.g)\.\s*', r'\1<DOT> ', para)
            
            # 文分割
            # 文末記号の後に空白と大文字が来る場合に分割
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            
            # 保護した略語を復元
            parts = [p.replace('<DOT>', '.') for p in parts]
            
            sentences.extend(parts)
        
        # 空文を除外
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """
        コードブロックの抽出と除外
        
        Args:
            text: 入力テキスト
            
        Returns:
            (コードブロック除外済みテキスト, コードブロックリスト)
        """
        code_blocks = []
        
        # ```で囲まれたコードブロック
        pattern = r'```[^`]*```'
        matches = re.findall(pattern, text, re.DOTALL)
        code_blocks.extend(matches)
        
        # コードブロックをプレースホルダーに置換
        text_without_code = re.sub(pattern, '[CODE_BLOCK]', text, flags=re.DOTALL)
        
        # インラインコード
        inline_pattern = r'`[^`]+`'
        inline_matches = re.findall(inline_pattern, text_without_code)
        code_blocks.extend(inline_matches)
        
        text_without_code = re.sub(inline_pattern, '[INLINE_CODE]', text_without_code)
        
        return text_without_code, code_blocks
    
    def normalize_numbers(self, text: str) -> str:
        """
        数値表現の正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        # カンマ区切りの数値（1,000 → 1000）
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        
        # パーセント表記の統一
        text = re.sub(r'(\d+)\s*percent', r'\1%', text, flags=re.IGNORECASE)
        
        # 分数表記の統一（スペース除去）
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)
        
        return text
    
    def check_truncation(self, text: str) -> bool:
        """
        文章が途中で切れているかチェック
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            途中切れの場合True
        """
        # 末尾の単語を取得
        text = text.strip()
        if not text:
            return False
        
        # 未閉じ括弧・引用符チェック
        open_parens = text.count('(') - text.count(')')
        open_brackets = text.count('[') - text.count(']')
        open_braces = text.count('{') - text.count('}')
        
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            logger.debug("未閉じ括弧を検出")
            return True
        
        # 引用符の数が奇数
        if text.count('"') % 2 != 0:
            logger.debug("未閉じ引用符を検出")
            return True
        
        # コードブロックの未閉じ
        if text.count('```') % 2 != 0:
            logger.debug("未閉じコードブロックを検出")
            return True
        
        # 不完全な文末パターン
        incomplete_patterns = [
            r',\s*$',                    # カンマで終わる
            r':\s*$',                    # コロンで終わる
            r'\b(and|or|but|so|thus|therefore|however)\s*$',  # 接続詞で終わる
            r'\b(the|a|an|this|that|these|those)\s*$',        # 冠詞/指示詞で終わる
            r'\b(is|are|was|were|be|been|being)\s*$',         # be動詞で終わる
            r'\b(to|for|with|from|at|in|on|by)\s*$',          # 前置詞で終わる
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug(f"不完全な文末パターンを検出: {pattern}")
                return True
        
        return False


def normalize_sample(sample: Dict[str, str]) -> Dict[str, str]:
    """
    データサンプルの正規化
    
    Args:
        sample: データサンプル（question, think, answer等を含む）
        
    Returns:
        正規化済みサンプル
    """
    normalizer = TextNormalizer()
    normalized = {}
    
    for key, value in sample.items():
        if isinstance(value, str):
            normalized[key] = normalizer.normalize(value)
        else:
            normalized[key] = value
    
    return normalized