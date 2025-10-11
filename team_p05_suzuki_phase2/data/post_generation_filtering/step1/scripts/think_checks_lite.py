"""
thinking部分のチェック（軽量版）
n-gram重複、エントロピー分析、途中切れ・長さチェック
LLMを使わず統計的手法で高速処理
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
from util_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


class ThinkingCheckerLite:
    """thinking部分のチェッククラス（軽量版）"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定（think_checks, ngram, lengthセクション）
        """
        self.config = config
        self.tokenizer = get_tokenizer()
        
        # デフォルト設定
        self.length_config = config.get('length', {
            't_min': 30,           # 最小トークン数（fail判定）
            't_min_review': 50,    # レビュー判定の最小トークン数
            't_min_strict': 25     # 絶対的な最小値
        })
        
        self.ngram_config = config.get('ngram', {
            'n': 5,                       # n-gramのサイズ
            'topk': 20,                  # 上位k個
            'fail_repeated_5gram_strong': 5,      # 5以上の5-gram反復で強いfail
            'fail_repeated_5gram_medium': 3,      # 3以上の5-gram反復で中程度fail
            'fail_distinct_2_threshold': 0.63,    # distinct-2のfail閾値
            'coverage_review': 0.35,      # カバレッジ閾値（review）
            'entropy_review': 2.5,       # エントロピー閾値（review）
            'distinct_2_review': 0.70,   # distinct-2のreview閾値
            'distinct_3_min': 0.7,       # distinct-3の最小値
            'self_bleu_max': 0.4         # self-BLEUの最大値
        })
        
        self.truncation_config = config.get('truncation', {
            'check_brackets': True,
            'check_quotes': True,
            'check_endings': True,
            'check_sentence_completion': True,
            'check_math_expressions': True,
            'incomplete_patterns': [
                r'\.\.\.$',      # ...で終わる
                r',$',           # カンマで終わる
                r'\s+and$',      # andで終わる
                r'\s+or$',       # orで終わる
                r'\s+but$',      # butで終わる
                r'\s+so$',       # soで終わる
                r':\s*$'         # コロンで終わる（内容なし）
            ]
        })
    
    def check_truncation(self, text: str, token_count: int = None) -> Tuple[bool, str]:
        """
        途中切れチェック（改善版 - 誤検知を削減）
        
        Args:
            text: チェック対象テキスト
            token_count: トークン数（既に計算済みの場合）
            
        Returns:
            (途中切れか, 理由)
        """
        if not text or len(text) < 10:
            return False, ""
        
        # トークン数を計算（渡されていない場合）
        if token_count is None:
            token_count = self.tokenizer.count_tokens(text)
        
        # トークン数が0の場合は、トークナイザー失敗の可能性が高い（reviewへ）
        if token_count == 0 and len(text) > 50:
            return False, "tokenizer_failed_review"  # failではなくreviewへ
        
        reasons = []
        warning_reasons = []  # 警告レベル（reviewレベル）の理由
        
        # 文末が適切に終わっているかチェック（新規追加）
        text_stripped = text.strip()
        proper_endings = r'[.!?。！？]\s*$|[.!?。！？]["\')]\s*$'  # 句点や引用符付き句点で終わる
        has_proper_ending = bool(re.search(proper_endings, text_stripped[-10:]))
        
        # 括弧の対応チェック（警告レベルに格下げ）
        if self.truncation_config.get('check_brackets', True):
            if text.count('(') != text.count(')'):
                warning_reasons.append("unclosed_parentheses")
            if text.count('[') != text.count(']'):
                warning_reasons.append("unclosed_brackets")
            if text.count('{') != text.count('}'):
                warning_reasons.append("unclosed_braces")
        
        # 引用符チェック（警告レベルに格下げ）
        if self.truncation_config.get('check_quotes', True):
            if text.count('"') % 2 != 0:
                warning_reasons.append("unclosed_quotes")
            if text.count("'") % 2 != 0:
                # シングルクォートは所有格もあるので、明らかな場合のみ
                if re.search(r"'\w{10,}", text):  # 長い単語の前のクォート
                    warning_reasons.append("unclosed_single_quotes")
        
        # コードブロック（これは重要なのでreasons に残す）
        if text.count('```') % 2 != 0:
            reasons.append("unclosed_code_block")
        
        # 文末チェック
        if self.truncation_config.get('check_endings', True):
            last_100 = text[-100:].strip()
            
            # デフォルトの不完全な文末パターン
            incomplete_endings = [
                r'(and|or|but|so|thus|therefore|however)\s*$',
                r'(the|a|an|this|that)\s*$',
                r'(to|for|with|from|at|in|on|by)\s*$',
                r',\s*$',  # カンマで終わる
                r':\s*$',  # コロンで終わる（リストが続くはず）
            ]
            
            # 設定から追加のパターンを取得
            custom_patterns = self.truncation_config.get('incomplete_patterns', [])
            
            for pattern in incomplete_endings + custom_patterns:
                if re.search(pattern, last_100, re.IGNORECASE):
                    # 適切な終端があれば警告レベルに格下げ
                    if has_proper_ending:
                        warning_reasons.append(f"incomplete_ending_but_has_period")
                    else:
                        reasons.append(f"incomplete_ending")
                    break
        
        # 文の完全性チェック
        if self.truncation_config.get('check_sentence_completion', True):
            sentences = re.split(r'[.!?]+', text)
            if sentences and len(sentences[-1].strip()) > 20:  # 最後の文が長いが句点がない
                # トークン数が十分あり、文末記号はないが改行で終わっている場合は警告レベル
                if token_count > 0 and text_stripped[-1:] in '\n':
                    warning_reasons.append("missing_sentence_ending_but_newline")
                else:
                    reasons.append("missing_sentence_ending")
        
        # 数式の完全性チェック
        if self.truncation_config.get('check_math_expressions', True):
            # 数式パターンの不完全性
            math_patterns = [
                r'=\s*$',                    # =で終わる
                r'[+\-*/]\s*$',             # 演算子で終わる
                r'\d+\s*[+\-*/]\s*$',       # 数値+演算子で終わる
                r'\\[a-zA-Z]+\s*$',         # LaTeXコマンドで終わる
            ]
            
            for pattern in math_patterns:
                if re.search(pattern, text[-50:]):
                    reasons.append("incomplete_math_expression")
                    break
        
        # 判定ロジックの改善
        # トークン数が0で警告のみの場合は、fail判定しない（reviewへ）
        if token_count == 0 and not reasons and warning_reasons:
            return False, f"token_count_zero_with_warnings: {', '.join(warning_reasons[:3])}"
        
        # トークン数が十分（>0）で、文末が適切で、重大な問題（reasons）がない場合はOK
        if token_count > 0 and has_proper_ending and not reasons:
            return False, ""  # 問題なし
        
        # 重大な問題がある場合のみfail判定
        if reasons:
            return True, ", ".join(reasons[:3])  # 最初の3つの理由
        
        # 警告レベルの問題のみの場合はfailにしない
        if warning_reasons:
            return False, f"warnings_only: {', '.join(warning_reasons[:3])}"
        
        return False, ""
    
    def check_length(self, text: str) -> Tuple[str, int]:
        """
        長さチェック（無効化版 - 常にOKを返す）
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            (ステータス, トークン数)
        """
        token_count = self.tokenizer.count_tokens(text)
        
        # 長さチェックは無効化され、常に'ok'を返す
        # トークン数は記録のために計算するが、判定には使用しない
        return 'ok', token_count
    
    def calculate_distinct_n(self, text: str, n: int) -> float:
        """
        distinct-nメトリクスの計算
        
        Args:
            text: テキスト
            n: n-gramのサイズ
            
        Returns:
            distinct-n値（0-1の範囲）
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return unique_ngrams / total_ngrams
    
    def calculate_self_bleu(self, text: str, n: int = 4) -> float:
        """
        簡易self-BLEUの計算（n-gramオーバーラップに基づく）
        
        Args:
            text: テキスト
            n: n-gramの最大サイズ
            
        Returns:
            self-BLEU近似値（0-1の範囲）
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # 各文をn-gram集合に変換
        sentence_ngrams = []
        for sent in sentences[:10]:  # 最初の10文のみ（効率化のため）
            words = sent.lower().split()
            if len(words) < n:
                continue
            ngrams = set()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngrams.add(ngram)
            sentence_ngrams.append(ngrams)
        
        if len(sentence_ngrams) < 2:
            return 0.0
        
        # ペアワイズ類似度の計算
        similarities = []
        for i in range(len(sentence_ngrams)):
            for j in range(i + 1, len(sentence_ngrams)):
                set1, set2 = sentence_ngrams[i], sentence_ngrams[j]
                if len(set1) == 0 or len(set2) == 0:
                    continue
                overlap = len(set1 & set2)
                similarity = overlap / max(len(set1), len(set2))
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        return np.mean(similarities)
    
    def check_ngram_redundancy(self, text: str) -> Dict[str, any]:
        """
        n-gram重複とエントロピー分析（誤検知を大幅に削減した改良版）
        
        Args:
            text: チェック対象テキスト
            
        Returns:
            チェック結果辞書
        """
        result = {
            'status': 'ok',
            'ngram_coverage': 0.0,
            'repeated_5grams': [],
            'repeated_5grams_count': 0,  # 追加: 繰り返しn-gramの個数
            'entropy': 0.0,
            'distinct_2': 0.0,
            'distinct_3': 0.0,
            'self_bleu': 0.0
        }
        
        # 単語分割
        words = text.lower().split()
        
        if len(words) < 5:
            return result
        
        # distinct-nメトリクスの計算
        result['distinct_2'] = self.calculate_distinct_n(text, 2)
        result['distinct_3'] = self.calculate_distinct_n(text, 3)
        
        # self-BLEUの計算
        result['self_bleu'] = self.calculate_self_bleu(text, 4)
        
        # 5-gram分析
        n = 5
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return result
        
        # 頻度カウント
        ngram_counts = Counter(ngrams)
        
        # 上位20個のカバレッジ計算
        topk = min(20, len(ngram_counts))
        top_ngrams = ngram_counts.most_common(topk)
        total_count = len(ngrams)
        top_count = sum(count for _, count in top_ngrams)
        result['ngram_coverage'] = top_count / total_count if total_count > 0 else 0
        
        # 繰り返し5-gramの収集（2回以上繰り返されているもの）
        repeated_ngrams = []
        for ngram, count in ngram_counts.items():
            if count >= 2:
                repeated_ngrams.append((ngram, count))
                result['repeated_5grams'].append(' '.join(ngram))
                if len(result['repeated_5grams']) >= 10:  # 最大10個まで記録（分析用）
                    break
        
        result['repeated_5grams_count'] = len(repeated_ngrams)
        
        # エントロピー計算（単語レベル）
        if words:
            word_counts = Counter(words)
            probs = np.array(list(word_counts.values())) / len(words)
            # scipy.stats.entropyの代わりにnumpyで計算
            result['entropy'] = -np.sum(probs * np.log(probs + 1e-10))
        
        # ========== 改善された閾値設定 ==========
        # Fail判定用の閾値（より厳格に、誤検知を減らす）
        fail_repeated_5gram_strong = self.ngram_config.get('fail_repeated_5gram_strong', 5)  # 5以上で強いfail
        fail_repeated_5gram_medium = self.ngram_config.get('fail_repeated_5gram_medium', 3)  # 3以上で中程度fail（条件付き）
        fail_distinct_2_threshold = self.ngram_config.get('fail_distinct_2_threshold', 0.63)  # distinct-2の厳格な閾値
        
        # Review判定用の閾値
        coverage_review = self.ngram_config.get('coverage_review', 0.35)
        entropy_review = self.ngram_config.get('entropy_review', 2.5)
        distinct_2_review = self.ngram_config.get('distinct_2_review', 0.70)  # distinct-2のreview閾値
        distinct_3_min = self.ngram_config.get('distinct_3_min', 0.7)
        self_bleu_max = self.ngram_config.get('self_bleu_max', 0.4)
        
        # ========== 改善された判定ロジック（提案に基づく） ==========
        # STEP 1: Hard Fail判定（明確な冗長性のみ）
        hard_fail_reasons = []
        
        # 条件1: 繰り返し5-gram ≥ 5 → 強いfail
        if result['repeated_5grams_count'] >= fail_repeated_5gram_strong:
            hard_fail_reasons.append(f"high_5gram_repetition({result['repeated_5grams_count']})")
            result['status'] = 'fail_redundant'
        
        # 条件2: distinct-2 < 0.63 かつ 繰り返し5-gram ≥ 3 → 中程度fail
        elif (result['distinct_2'] < fail_distinct_2_threshold and 
              result['repeated_5grams_count'] >= fail_repeated_5gram_medium):
            hard_fail_reasons.append(f"low_distinct_2({result['distinct_2']:.2f}) + repetition({result['repeated_5grams_count']})")
            result['status'] = 'fail_redundant'
        
        # Hard Failの理由を記録
        if hard_fail_reasons:
            result['hard_fail_reasons'] = hard_fail_reasons
            return result  # Hard Failの場合はここで終了
        
        # STEP 2: Review判定（軽度の冗長性、またはボーダーライン）
        review_reasons = []
        
        # distinct-2が高く（≥0.70）、かつ5-gram反復が少ない（≤2）場合は、他の指標も考慮
        if result['distinct_2'] >= distinct_2_review and result['repeated_5grams_count'] <= 2:
            # この場合は他の指標が悪くてもreviewに留める
            if result['ngram_coverage'] > coverage_review:
                review_reasons.append(f"moderate_coverage({result['ngram_coverage']:.2f}) but good_distinct")
            if result['entropy'] < entropy_review:
                review_reasons.append(f"moderate_entropy({result['entropy']:.2f}) but good_distinct")
            # この条件では、基本的にfailにしない
        else:
            # 通常のreview判定
            # n-gramカバレッジが高め
            if result['ngram_coverage'] > coverage_review:
                review_reasons.append(f"moderate_coverage({result['ngram_coverage']:.2f})")
            
            # distinct-2が低め（ただしfail基準より上）
            if result['distinct_2'] < distinct_2_review and result['distinct_2'] >= fail_distinct_2_threshold:
                review_reasons.append(f"moderate_distinct_2({result['distinct_2']:.2f})")
            
            # distinct-3が低め
            if result['distinct_3'] < distinct_3_min:
                review_reasons.append(f"moderate_distinct_3({result['distinct_3']:.2f})")
            
            # self-BLEUが高め
            if result['self_bleu'] > self_bleu_max:
                review_reasons.append(f"high_self_bleu({result['self_bleu']:.2f})")
            
            # エントロピーが低め
            if result['entropy'] < entropy_review:
                review_reasons.append(f"moderate_entropy({result['entropy']:.2f})")
            
            # 繰り返し5-gramが少しある（fail基準未満）
            if result['repeated_5grams_count'] >= 2 and result['repeated_5grams_count'] < fail_repeated_5gram_medium:
                review_reasons.append(f"some_repetition({result['repeated_5grams_count']})")
        
        if review_reasons:
            result['status'] = 'review_redundant'
            result['review_reasons'] = review_reasons
        
        return result
    
    def check_explicit_derivation(self, think_text: str, answer_text: str) -> Tuple[bool, str]:
        """
        明示的導出チェック（軽量版）
        
        Args:
            think_text: thinking部分のテキスト
            answer_text: answer部分のテキスト
            
        Returns:
            (明示的導出があるか, 理由)
        """
        if not think_text or not answer_text:
            return False, "Empty text"
        
        # 文末部分を取得（最後の200文字）
        last_part = think_text[-500:].lower()
        answer_normalized = answer_text.strip().lower()
        
        # 答えが明示的に含まれているか
        if answer_normalized in last_part:
            return True, "Answer found in final part"
        
        # 単一選択肢の場合
        if len(answer_text.strip()) == 1 and answer_text.strip().upper() in 'ABCDEF':
            option = answer_text.strip().upper()
            
            # 選択肢への言及パターン
            patterns = [
                f'answer is {option}',
                f'{option} is correct',
                f'{option} is the answer',
                f'choose {option}',
                f'option {option}',
                f'\\({option}\\)',
            ]
            
            for pattern in patterns:
                if re.search(pattern, last_part, re.IGNORECASE):
                    return True, f"Option {option} explicitly mentioned"
        
        # 結論を示すパターン
        conclusion_patterns = [
            r'therefore.*answer',
            r'thus.*answer',
            r'so.*answer',
            r'the answer is',
            r'correct answer',
            r'final answer',
        ]
        
        for pattern in conclusion_patterns:
            if re.search(pattern, last_part):
                return True, "Conclusion pattern found"
        
        return False, "No explicit derivation"
    
    def run_all_checks(self, think_text: str, answer_text: str = "") -> Dict[str, any]:
        """
        すべてのチェックを実行（改善版 - 誤検知を削減）
        
        Args:
            think_text: thinking部分のテキスト
            answer_text: answer部分のテキスト（オプション）
            
        Returns:
            チェック結果の辞書
        """
        results = {
            'checks': {},
            'flags': [],
            'fatal_errors': [],
            'token_count': 0
        }
        
        # 1. 長さチェックを先に実行（トークン数を取得）
        length_status, token_count = self.check_length(think_text)
        results['token_count'] = token_count
        results['checks']['length'] = {
            'status': length_status,
            'tokens': token_count
        }
        
        # 2. 途中切れチェック（トークン数を渡す）
        is_truncated, truncation_reason = self.check_truncation(think_text, token_count)
        
        # トークン数が0で、警告のみの場合はreviewへ
        if token_count == 0 and truncation_reason.startswith(('token_count_zero', 'warnings_only')):
            results['flags'].append('review_truncation')
            results['checks']['truncation'] = truncation_reason
        elif is_truncated:
            # トークン数が十分（>0）で文末が句点で終わっている場合は、reviewに格下げ
            if token_count > 0 and re.search(r'[.!?。！？][\s"\')]*$', think_text.strip()[-20:]):
                results['flags'].append('review_truncation_with_period')
                results['checks']['truncation'] = f"has_period_but_{truncation_reason}"
            else:
                results['fatal_errors'].append('fail_truncated')
                results['checks']['truncation'] = truncation_reason
                return results  # Fail-fast
        elif truncation_reason:  # 警告レベルの問題がある
            results['checks']['truncation_warnings'] = truncation_reason
        
        # 3. 長さチェックの結果処理（無効化）
        # 長さチェックは無効化されているため、常に'ok'が返される
        # よって、エラーやフラグの追加は不要
        
        # 4. n-gram冗長性とエントロピーチェック
        ngram_result = self.check_ngram_redundancy(think_text)
        results['checks']['ngram'] = ngram_result
        
        if ngram_result['status'] == 'fail_redundant':
            results['fatal_errors'].append('fail_redundant')
            return results  # Fail-fast
        elif ngram_result['status'] != 'ok':
            results['flags'].append(ngram_result['status'])
        
        # 5. 明示的導出チェック（answerがある場合）
        if answer_text:
            has_derivation, derivation_reason = self.check_explicit_derivation(think_text, answer_text)
            results['checks']['explicit_derivation'] = {
                'has_derivation': has_derivation,
                'reason': derivation_reason
            }
            
            if not has_derivation:
                results['flags'].append('review_no_derivation')
        
        return results
