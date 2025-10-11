"""
数式検証の軽量版（LLMを使わずPythonエンジンで検証）
Step1用の高速処理版
"""

import re
import ast
import logging
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)


class MathValidatorLite:
    """数式検証クラス（軽量版）"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 設定（math_validationセクション）
        """
        if config is None:
            config = {}
        
        self.config = config
        
        # 安全な数学関数のホワイトリスト
        self.safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }
        
        # 数式パターン（高速化のため基本的なもののみ）
        self.equation_patterns = [
            # 基本的な等式: 2 + 3 = 5
            r'([\d\s\+\-\*\/\(\)]+)\s*=\s*([\d\.]+)',
            # 変数への代入: x = 5
            r'(\w+)\s*=\s*([\d\.]+)',
        ]
    
    def validate_calculations(self, text: str) -> Dict[str, any]:
        """
        テキスト内の計算を高速検証
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            検証結果の辞書
        """
        result = {
            'has_math': False,
            'errors': [],
            'warnings': [],
            'score': 1.0,
            'checked_equations': 0
        }
        
        # 数式を抽出（最大10個まで検証）
        equations = self._extract_equations(text, limit=10)
        
        if not equations:
            return result
        
        result['has_math'] = True
        result['checked_equations'] = len(equations)
        
        # 各数式を検証
        for eq in equations:
            validation = self._validate_equation(eq)
            
            if validation['status'] == 'error':
                result['errors'].append(validation['error'])
                result['score'] *= 0.5  # エラーごとにスコアを半減
            elif validation['status'] == 'warning':
                result['warnings'].append(validation.get('warning', ''))
                result['score'] *= 0.9  # 警告ごとにスコアを少し減らす
        
        return result
    
    def _extract_equations(self, text: str, limit: int = 10) -> List[Dict]:
        """
        テキストから数式を抽出（高速版）
        
        Args:
            text: 入力テキスト
            limit: 抽出する最大数
            
        Returns:
            抽出された数式のリスト
        """
        equations = []
        
        for pattern in self.equation_patterns:
            if len(equations) >= limit:
                break
                
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                if len(equations) >= limit:
                    break
                    
                eq_dict = {
                    'full_text': match.group(0),
                    'left': match.group(1).strip(),
                    'right': match.group(2).strip(),
                    'type': 'equation'
                }
                
                equations.append(eq_dict)
        
        return equations
    
    def _validate_equation(self, equation: Dict) -> Dict:
        """
        単一の数式を検証（軽量版）
        
        Args:
            equation: 数式情報
            
        Returns:
            検証結果
        """
        validation = {
            'equation': equation['full_text'],
            'status': 'ok',
            'error': None
        }
        
        left = equation.get('left', '')
        right = equation.get('right', '')
        
        # 単純な算術式の検証のみ
        if self._is_arithmetic_expression(left):
            try:
                # 安全な評価
                calculated = self._safe_eval(left)
                
                # 右辺が数値の場合
                try:
                    expected = float(right)
                    
                    # 誤差の許容範囲
                    tolerance = 1e-6
                    if abs(calculated - expected) > tolerance:
                        validation['status'] = 'error'
                        validation['error'] = f"Math error: {left} = {calculated:.6f}, not {expected}"
                    
                except ValueError:
                    # 右辺が数値でない場合はスキップ
                    validation['status'] = 'skipped'
                    
            except Exception:
                # 評価できない場合はスキップ
                validation['status'] = 'skipped'
        else:
            # 変数を含む式はスキップ
            validation['status'] = 'skipped'
        
        return validation
    
    def _is_arithmetic_expression(self, expr: str) -> bool:
        """
        算術式かどうかを判定（高速版）
        
        Args:
            expr: 式の文字列
            
        Returns:
            算術式ならTrue
        """
        # 許可する文字: 数字、算術演算子、括弧、小数点、空白
        allowed_chars = set('0123456789+-*/()., \t')
        
        # すべての文字が許可リストに含まれているかチェック
        return all(c in allowed_chars for c in expr)
    
    def _safe_eval(self, expr: str) -> float:
        """
        安全に数式を評価（軽量版）
        
        Args:
            expr: 評価する式
            
        Returns:
            計算結果
            
        Raises:
            ValueError: 評価できない場合
        """
        # 空白とカンマを除去
        expr = expr.replace(' ', '').replace('\t', '').replace(',', '')
        
        try:
            # astを使って安全にパース
            node = ast.parse(expr, mode='eval')
            
            # 基本的な演算ノードのみ許可
            for n in ast.walk(node):
                if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                     ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.Pow, ast.USub, ast.UAdd,
                                     ast.Constant, ast.Num)):
                    raise ValueError(f"Unsafe operation")
            
            # 評価
            return eval(compile(node, '<string>', 'eval'))
            
        except Exception as e:
            raise ValueError(f"Evaluation failed: {str(e)}")
    
    def detect_calculation_patterns(self, text: str) -> List[str]:
        """
        計算を含む問題のパターンを検出（軽量版）
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            検出されたパターンのリスト
        """
        patterns_found = []
        
        # 基本的なパターンのみチェック
        calculation_indicators = [
            (r'calculate', 'calculation'),
            (r'what\s+is\s+\d+\s*[+\-*/]\s*\d+', 'arithmetic'),
            (r'\d+%\s+of\s+\d+', 'percentage'),
        ]
        
        text_lower = text.lower()
        for pattern, pattern_type in calculation_indicators:
            if re.search(pattern, text_lower):
                patterns_found.append(pattern_type)
        
        return patterns_found
