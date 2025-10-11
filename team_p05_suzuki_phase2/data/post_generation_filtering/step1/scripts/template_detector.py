"""
テンプレートフレーズ検出器
think部分の定型的・冗長な表現を検出
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class TemplateDetector:
    """テンプレート的フレーズの検出クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 設定（template_detectionセクション）
        """
        if config is None:
            config = {}
        
        self.config = config
        
        # よく使われるテンプレートフレーズ
        self.template_phrases = config.get('template_phrases', [
            # 評価開始の定型文
            r"Now,?\s+(?:let'?s\s+)?evaluat(?:e|ing)\s+(?:the\s+|each\s+)?options?",
            r"Let'?s\s+(?:evaluate|analyze|examine|review)\s+(?:each\s+|the\s+)?options?",
            r"Reviewing\s+the\s+options",
            r"Looking\s+at\s+(?:the\s+|each\s+)?options?",
            r"Considering\s+(?:the\s+|each\s+)?options?",
            
            # 結論の定型文
            r"Based\s+on\s+(?:this|the\s+above)",
            r"Therefore,?\s+(?:the\s+)?option",
            r"Thus,?\s+(?:the\s+)?option",
            r"So,?\s+(?:the\s+)?option",
            r"Hence,?\s+(?:the\s+)?option",
            r"The\s+correct\s+answer\s+is",
            r"The\s+answer\s+is\s+(?:clearly\s+)?",
            
            # 選択肢の評価定型文
            r"Option\s+[A-F]\s+is\s+(?:in)?correct\s+because",
            r"Option\s+[A-F]:\s+(?:In)?correct",
            r"[A-F]\)\s+(?:In)?correct:",
            
            # 冗長な接続詞
            r"Additionally,?\s+",
            r"Furthermore,?\s+",
            r"Moreover,?\s+",
            r"In\s+addition,?\s+",
            r"On\s+the\s+other\s+hand,?\s+",
        ])
        
        # 冗長な説明パターン
        self.redundant_patterns = config.get('redundant_patterns', [
            r"As\s+(?:we\s+)?(?:mentioned|stated|discussed)\s+(?:earlier|above|before)",
            r"It'?s\s+(?:important|worth|crucial)\s+to\s+(?:note|mention|remember)\s+that",
            r"(?:First|Second|Third)ly,?\s+let'?s\s+(?:consider|examine|look\s+at)",
            r"This\s+is\s+because",
            r"The\s+reason\s+(?:is|being)\s+that",
        ])
        
        # 閾値設定
        self.max_template_count = config.get('max_template_count', 3)  # 3回以上でreview
        self.max_template_ratio = config.get('max_template_ratio', 0.15)  # 15%以上でreview
    
    def detect_templates(self, text: str) -> Dict[str, any]:
        """
        テンプレートフレーズを検出
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            検出結果の辞書
        """
        result = {
            'status': 'ok',
            'template_count': 0,
            'template_ratio': 0.0,
            'found_templates': [],
            'redundant_count': 0,
            'issues': []
        }
        
        # テキストの単語数を計算
        word_count = len(text.split())
        if word_count == 0:
            return result
        
        # テンプレートフレーズの検出
        template_matches = []
        for pattern in self.template_phrases:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                template_matches.append({
                    'pattern': pattern[:30] + '...' if len(pattern) > 30 else pattern,
                    'text': match.group(),
                    'position': match.start()
                })
        
        result['template_count'] = len(template_matches)
        result['found_templates'] = template_matches[:10]  # 最初の10個まで
        
        # テンプレートの割合を計算（マッチした文字数 / 全体の文字数）
        template_chars = sum(len(m['text']) for m in template_matches)
        result['template_ratio'] = template_chars / len(text) if len(text) > 0 else 0
        
        # 冗長パターンの検出
        redundant_matches = []
        for pattern in self.redundant_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            redundant_matches.extend(matches)
        
        result['redundant_count'] = len(redundant_matches)
        
        # 判定
        if result['template_count'] >= self.max_template_count:
            result['status'] = 'review'
            result['issues'].append(f'too_many_templates: {result["template_count"]} found')
        
        if result['template_ratio'] >= self.max_template_ratio:
            result['status'] = 'review'
            result['issues'].append(f'high_template_ratio: {result["template_ratio"]:.1%}')
        
        if result['redundant_count'] >= 3:
            if result['status'] == 'ok':
                result['status'] = 'review'
            result['issues'].append(f'redundant_phrases: {result["redundant_count"]} found')
        
        return result
    
    def detect_repetition(self, text: str) -> Dict[str, any]:
        """
        文章の繰り返しを検出
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            検出結果の辞書
        """
        result = {
            'has_repetition': False,
            'repeated_sentences': [],
            'similarity_scores': []
        }
        
        # 文に分割
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return result
        
        # 文同士の類似度チェック
        for i in range(len(sentences) - 1):
            for j in range(i + 1, len(sentences)):
                sent1 = sentences[i].lower()
                sent2 = sentences[j].lower()
                
                # 単語集合の類似度計算
                words1 = set(sent1.split())
                words2 = set(sent2.split())
                
                if len(words1) < 5 or len(words2) < 5:
                    continue
                
                # Jaccard類似度
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity > 0.7:  # 70%以上の類似度
                        result['has_repetition'] = True
                        result['repeated_sentences'].append({
                            'index1': i,
                            'index2': j,
                            'similarity': similarity,
                            'sent1': sentences[i][:100],
                            'sent2': sentences[j][:100]
                        })
                        result['similarity_scores'].append(similarity)
        
        return result
    
    def suggest_cleanup(self, text: str) -> str:
        """
        テンプレート文のクリーンアップ提案
        
        Args:
            text: 元のテキスト
            
        Returns:
            クリーンアップ後のテキスト
        """
        cleaned = text
        
        # 明らかに不要な定型文を削除
        remove_patterns = [
            r"Now,?\s+let'?s\s+evaluate\s+(?:the\s+|each\s+)?options?\.?\s*",
            r"Let'?s\s+(?:evaluate|analyze|examine)\s+each\s+option\.?\s*",
            r"Looking\s+at\s+the\s+options:?\s*",
            r"Based\s+on\s+(?:the\s+)?above,?\s*",
            r"It'?s\s+(?:important|worth)\s+to\s+note\s+that\s*",
        ]
        
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # 冗長な接続詞を簡潔に
        replacements = [
            (r"Additionally,?\s+", "Also, "),
            (r"Furthermore,?\s+", "Also, "),
            (r"Moreover,?\s+", "Also, "),
            (r"In\s+addition,?\s+", "Also, "),
            (r"On\s+the\s+other\s+hand,?\s+", "However, "),
        ]
        
        for old, new in replacements:
            cleaned = re.sub(old, new, cleaned, flags=re.IGNORECASE)
        
        # 連続した空白や改行を整理
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)
        
        return cleaned.strip()
