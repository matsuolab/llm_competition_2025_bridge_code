"""
ルールベースの矛盾検出器（軽量版）
LLMを使わず正規表現とパターンマッチングで高速処理
"""

import re
import logging
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContradictionDetectorLite:
    """ルールベースの矛盾検出クラス（軽量版）"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 設定（contradiction_detectionセクション）
        """
        if config is None:
            config = {}
        
        self.config = config
        
        # 基本的な矛盾パターン（高速処理のため簡略化）
        self.contradiction_patterns = [
            # 比較の矛盾: X > Y と Y > X
            {
                'pattern1': r'(\w+)\s+(?:is\s+)?(?:greater|larger|bigger|more)\s+than\s+(\w+)',
                'pattern2': r'{1}\s+(?:is\s+)?(?:greater|larger|bigger|more)\s+than\s+{0}',
                'type': 'comparison'
            },
            # 等価性の矛盾: X = Y と X != Y
            {
                'pattern1': r'(\w+)\s*=\s*(\w+)',
                'pattern2': r'{0}\s*!=\s*{1}',
                'type': 'equality'
            },
            # 真偽の矛盾: X is true と X is false
            {
                'pattern1': r'(\w+)\s+is\s+true',
                'pattern2': r'{0}\s+is\s+false',
                'type': 'boolean'
            },
        ]
        
        # 数値の矛盾チェック用パターン
        self.numeric_patterns = [
            r'(\w+)\s*=\s*(\d+(?:\.\d+)?)',  # x = 5
            r'(\w+)\s+is\s+(\d+(?:\.\d+)?)',  # x is 5
        ]
    
    def detect_contradictions(self, text: str) -> List[Dict]:
        """
        テキスト内の矛盾を高速検出
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            検出された矛盾のリスト
        """
        contradictions = []
        text_lower = text.lower()
        
        # パターンベースの矛盾検出（最大5つまで）
        for pattern_def in self.contradiction_patterns:
            if len(contradictions) >= 5:
                break
                
            matches1 = list(re.finditer(pattern_def['pattern1'], text_lower))
            
            for match1 in matches1[:10]:  # 各パターンで最大10マッチまで
                if len(contradictions) >= 5:
                    break
                    
                # パターン2を動的に生成
                if '{' in pattern_def['pattern2']:
                    groups = match1.groups()
                    pattern2 = pattern_def['pattern2']
                    for i, group in enumerate(groups):
                        pattern2 = pattern2.replace(f'{{{i}}}', re.escape(group))
                else:
                    pattern2 = pattern_def['pattern2']
                
                # パターン2を検索
                matches2 = re.finditer(pattern2, text_lower)
                for match2 in matches2:
                    # 同じ位置でなければ矛盾として記録
                    if abs(match1.start() - match2.start()) > 10:  # 10文字以上離れている
                        contradictions.append({
                            'type': pattern_def['type'],
                            'statement1': match1.group()[:50],  # 最大50文字
                            'statement2': match2.group()[:50],
                            'description': f"{pattern_def['type']}_contradiction"
                        })
                        break
        
        # 数値の矛盾チェック（簡略版）
        numeric_contradictions = self._detect_numeric_contradictions_lite(text_lower)
        contradictions.extend(numeric_contradictions[:2])  # 最大2つまで
        
        # 選択肢の矛盾チェック（簡略版）
        choice_contradictions = self._detect_choice_contradictions_lite(text)
        contradictions.extend(choice_contradictions[:2])  # 最大2つまで
        
        return contradictions[:5]  # 最大5つの矛盾を返す
    
    def _detect_numeric_contradictions_lite(self, text: str) -> List[Dict]:
        """
        数値の矛盾を簡易検出
        
        Args:
            text: チェック対象テキスト（小文字）
            
        Returns:
            数値の矛盾リスト
        """
        contradictions = []
        variable_values = defaultdict(set)
        
        # 変数への値の割り当てを収集（最大20個）
        count = 0
        for pattern in self.numeric_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if count >= 20:
                    break
                var_name = match.group(1)
                value = float(match.group(2))
                variable_values[var_name].add(value)
                count += 1
        
        # 同じ変数に異なる値が割り当てられているかチェック
        for var_name, values in variable_values.items():
            if len(values) > 1:
                contradictions.append({
                    'type': 'numeric',
                    'variable': var_name,
                    'values': list(values)[:3],  # 最大3つの値
                    'description': f"variable_{var_name}_conflict"
                })
                if len(contradictions) >= 2:
                    break
        
        return contradictions
    
    def _detect_choice_contradictions_lite(self, text: str) -> List[Dict]:
        """
        選択肢に関する矛盾を簡易検出
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            選択肢の矛盾リスト
        """
        contradictions = []
        
        # 正しいとされる選択肢を検出
        correct_patterns = [
            r'(?:option\s+)?([A-F])\s+is\s+(?:the\s+)?correct',
            r'answer\s+is\s+([A-F])',
        ]
        
        correct_options = set()
        for pattern in correct_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                option = match.group(1).upper()
                correct_options.add(option)
        
        # 複数の選択肢が正しいとされている場合
        if len(correct_options) > 1:
            contradictions.append({
                'type': 'multiple_correct',
                'options': list(correct_options),
                'description': 'multiple_correct_answers'
            })
        
        return contradictions
    
    def calculate_contradiction_score(self, contradictions: List[Dict]) -> float:
        """
        矛盾の深刻度スコアを計算（軽量版）
        
        Args:
            contradictions: 検出された矛盾のリスト
            
        Returns:
            スコア（0.0-1.0、高いほど深刻）
        """
        if not contradictions:
            return 0.0
        
        # 矛盾タイプごとの重み（簡略版）
        weights = {
            'comparison': 0.8,
            'equality': 0.9,
            'boolean': 1.0,
            'numeric': 1.0,
            'multiple_correct': 0.8
        }
        
        total_score = 0.0
        for contradiction in contradictions:
            c_type = contradiction.get('type', 'unknown')
            weight = weights.get(c_type, 0.5)
            total_score += weight
        
        # 正規化（最大1.0）
        normalized_score = min(1.0, total_score / 3.0)  # 3つ以上の矛盾で最大スコア
        
        return normalized_score
