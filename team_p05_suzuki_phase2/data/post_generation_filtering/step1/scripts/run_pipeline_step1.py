"""
Step1: LLM非依存の基本フィルタリング処理
HuggingFaceデータセットの読み込みから中間出力まで
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import gc
import psutil
import traceback
import random

def get_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 物理メモリ使用量 (MB)
        'vms_mb': memory_info.vms / 1024 / 1024,  # 仮想メモリ使用量 (MB)
        'percent': process.memory_percent()        # システム全体に対する使用率
    }

def log_memory_usage(label: str):
    """メモリ使用量をログ出力"""
    try:
        memory = get_memory_usage()
        print(f"[MEMORY] {label}: RSS={memory['rss_mb']:.1f}MB, VMS={memory['vms_mb']:.1f}MB, Percent={memory['percent']:.1f}%")
        
        # システム全体のメモリ情報も取得
        sys_memory = psutil.virtual_memory()
        print(f"[SYSTEM] Available={sys_memory.available / 1024 / 1024:.1f}MB, Used={sys_memory.percent:.1f}%")
    except Exception as e:
        print(f"[MEMORY] Error getting memory info: {e}")

# 初期メモリ状態をログ
log_memory_usage("Script start")

# PyTorch関連のメモリエラー対策
print("[INFO] PyTorch動的コンパイル機能を無効化中...")
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

# GPU環境用メモリ効率化設定
print("[INFO] GPU環境用メモリ設定を適用中...")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # CUDA メモリ分割サイズ
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # トークナイザー並列化無効（メモリ節約）

try:
    # データセット関連のインポート前にメモリ状況を確認
    log_memory_usage("Before datasets import")
    from datasets import load_dataset, Dataset
    log_memory_usage("After datasets import")
    
    log_memory_usage("Before pandas import") 
    import pandas as pd
    log_memory_usage("After pandas import")
    
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    sys.exit(1)

# 内部モジュール
log_memory_usage("Before internal modules import")
sys.path.append(str(Path(__file__).parent))

try:
    from util_logging import setup_logging, AuditLogger, ProgressTracker
    log_memory_usage("After util_logging import")
    
    
    from normalize import normalize_sample, TextNormalizer
    log_memory_usage("After normalize import")
    
    from dedup import DuplicateDetector
    log_memory_usage("After dedup import")
    
    
    
    from template_detector import TemplateDetector
    log_memory_usage("After template_detector import")
    
    from think_checks_lite import ThinkingCheckerLite
    log_memory_usage("After think_checks_lite import")
    
    from math_validator_lite import MathValidatorLite
    log_memory_usage("After math_validator_lite import")
    
    from contradiction_detector_lite import ContradictionDetectorLite
    log_memory_usage("After contradiction_detector_lite import")
    
except Exception as e:
    print(f"[ERROR] Internal module import failed: {e}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class Step1Pipeline:
    """Step1: 基本フィルタリングパイプライン"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        log_memory_usage("Step1Pipeline init start")
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 分野別チューニング設定
        self.field_tuning_config = self.config.get('field_specific_tuning', {
            'enabled': False,
            'fields': {}
        })
        
        # 共有データディレクトリ
        self.shared_data_dir = Path(__file__).parent.parent.parent / 'shared_data'
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Step1ログディレクトリ
        self.step1_logs_dir = Path(__file__).parent.parent / 'logs'
        self.step1_logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_memory_usage("Before component initialization")
        
        # コンポーネント初期化
        logger.info("Step1コンポーネント初期化中...")
        
        
        self.normalizer = TextNormalizer()
        log_memory_usage("After normalizer init")
        
        self.dedup_detector = DuplicateDetector(self.config)
        log_memory_usage("After dedup_detector init")
        
        
        
        self.template_detector = TemplateDetector(self.config.get('template_detection', {}))
        log_memory_usage("After template_detector init")
        
        # 新しい軽量フィルタリングコンポーネント
        self.thinking_checker = ThinkingCheckerLite(self.config)
        log_memory_usage("After thinking_checker init")
        
        self.math_validator = MathValidatorLite(self.config.get('math_validation', {}))
        log_memory_usage("After math_validator init")
        
        self.contradiction_detector = ContradictionDetectorLite(self.config.get('contradiction_detection', {}))
        log_memory_usage("After contradiction_detector init")
        
        # 監査ログ（タイムスタンプ付きファイル名）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.audit_logger = AuditLogger(
            self.step1_logs_dir / f'audit_log_{timestamp}.jsonl',
            self.step1_logs_dir / f'summary_{timestamp}.json'
        )
        self.audit_logger.set_config_hash(self.config)
        
        log_memory_usage("Step1Pipeline init complete")
        logger.info("Step1パイプライン初期化完了")
    
    def get_field_config(self, sample: Dict) -> Dict:
        """
        サンプルの分野を判定し、対応する設定を取得
        
        Args:
            sample: データサンプル
            
        Returns:
            分野別の設定辞書
        """
        if not self.field_tuning_config.get('enabled', False):
            return {}
        
        # 分野の判定（questionやsource等から推定）
        question = sample.get('question', '').lower()
        source = sample.get('source', '').lower()
        
        # 分野判定ロジック（キーワードベース）
        field = 'default'
        if any(word in question for word in ['math', 'equation', 'calculate', 'solve', 'integral', 'derivative']):
            field = 'mathematics'
        elif any(word in question for word in ['physics', 'force', 'energy', 'momentum', 'velocity']):
            field = 'physics'
        elif any(word in question for word in ['biology', 'medical', 'disease', 'cell', 'dna', 'gene']):
            field = 'biology_medicine'
        elif any(word in question for word in ['code', 'program', 'algorithm', 'function', 'variable']):
            field = 'computer_science'
        
        # sourceベースの判定も追加
        if 'math' in source:
            field = 'mathematics'
        elif 'physics' in source:
            field = 'physics'
        elif 'bio' in source or 'med' in source:
            field = 'biology_medicine'
        elif 'code' in source or 'programming' in source:
            field = 'computer_science'
        
        return self.field_tuning_config.get('fields', {}).get(field, {})
    
    def apply_field_config(self, field_config: Dict):
        """
        分野別設定を各チェッカーに適用
        
        Args:
            field_config: 分野別設定
        """
        if not field_config:
            return
        
        # n-gram設定の調整（review用）
        if 'ngram_coverage_review' in field_config:
            self.thinking_checker.ngram_config['coverage_review'] = field_config['ngram_coverage_review']
        if 'distinct_2_min' in field_config:
            self.thinking_checker.ngram_config['distinct_2_min'] = field_config['distinct_2_min']
        if 'distinct_3_min' in field_config:
            self.thinking_checker.ngram_config['distinct_3_min'] = field_config['distinct_3_min']
        
        # 新しいhard fail基準の調整
        if 'hard_fail_entropy' in field_config:
            self.thinking_checker.ngram_config['hard_fail_entropy'] = field_config['hard_fail_entropy']
        if 'hard_fail_distinct_2' in field_config:
            self.thinking_checker.ngram_config['hard_fail_distinct_2'] = field_config['hard_fail_distinct_2']
        if 'hard_fail_ngram_coverage' in field_config:
            self.thinking_checker.ngram_config['hard_fail_ngram_coverage'] = field_config['hard_fail_ngram_coverage']
        if 'hard_fail_repeated_count' in field_config:
            self.thinking_checker.ngram_config['hard_fail_repeated_count'] = field_config['hard_fail_repeated_count']
        
        # 旧設定（互換性のため残す）
        if 'ngram_coverage_fail' in field_config:
            self.thinking_checker.ngram_config['coverage_fail'] = field_config['ngram_coverage_fail']
        if 'fivegram_repeat_fail' in field_config:
            self.thinking_checker.ngram_config['fivegram_repeat_fail'] = field_config['fivegram_repeat_fail']
        
        # 長さ設定の調整（無効化 - 長さチェックは無効になっているため）
        # if 'length_t_min' in field_config:
        #     self.thinking_checker.length_config['t_min'] = field_config['length_t_min']
        # if 'length_t_min_review' in field_config:
        #     self.thinking_checker.length_config['t_min_review'] = field_config['length_t_min_review']
        # if 'length_t_min_strict' in field_config:
        #     self.thinking_checker.length_config['t_min_strict'] = field_config['length_t_min_strict']
        
        # 途中切れ設定の調整
        if 'truncation_strict' in field_config and field_config['truncation_strict']:
            self.thinking_checker.truncation_config['check_sentence_completion'] = True
            self.thinking_checker.truncation_config['check_math_expressions'] = True
    
    def _categorize_fail_reason(self, issue: str) -> str:
        """
        Fail原因を分類してカテゴリ名を返す
        
        Args:
            issue: 問題の説明文字列
            
        Returns:
            分類されたカテゴリ名
        """
        issue_lower = issue.lower()
        
        # 必須フィールド不足
        if 'missing_fields' in issue_lower:
            return 'missing_fields'
        
        # 重複検出
        if 'duplicate' in issue_lower and 'near_duplicate' not in issue_lower:
            return 'duplicate'
        elif 'near_duplicate' in issue_lower:
            return 'near_duplicate'
        
        # 矛盾検出
        if 'contradiction_detected' in issue_lower or 'possible_contradiction' in issue_lower:
            return 'contradiction'
        
        # 数学エラー・警告
        if 'math_error' in issue_lower or 'math_warning' in issue_lower:
            return 'math_validation'
        
        # テンプレート・冗長フレーズ問題
        if any(keyword in issue_lower for keyword in [
            'too_many_templates', 'high_template_ratio', 'redundant_phrases', 'repetition_detected'
        ]):
            return 'template_redundancy'
        
        # thinking部分の長さ問題（無効化 - 長さチェックは無効になっているため）
        # if any(keyword in issue_lower for keyword in [
        #     'fail_length', 'review_length', 'length_too_short'
        # ]):
        #     return 'length_issues'
        
        # thinking部分の途中切れ問題
        if any(keyword in issue_lower for keyword in [
            'fail_truncated', 'review_truncation', 'truncation_with_period'
        ]):
            return 'truncation_issues'
        
        # n-gram冗長性・品質問題
        if any(keyword in issue_lower for keyword in [
            'fail_redundant', 'high_5gram_repetition', 'low_distinct_2', 
            'moderate_coverage', 'moderate_entropy', 'moderate_distinct',
            'high_self_bleu', 'some_repetition'
        ]):
            return 'ngram_quality'
        
        # 導出・論理問題
        if 'review_no_derivation' in issue_lower:
            return 'derivation_issues'
        
        # その他のthinking関連問題（上記に該当しない場合）
        if any(keyword in issue_lower for keyword in [
            'thinking', 'entropy', 'distinct', 'coverage', 'bleu'
        ]):
            return 'thinking_other'
        
        # 処理エラー
        if 'processing_error' in issue_lower:
            return 'processing_error'
        
        # その他
        return 'other'
    
    def process_sample(self, sample: Dict, sample_id: str) -> Dict:
        """
        単一サンプルの処理（Step1）
        
        Args:
            sample: データサンプル
            sample_id: サンプルID
            
        Returns:
            処理済みサンプル（Step1結果を含む）
        """
        # 元のサンプルデータのコピーを作成（元のデータを保持）
        processed_sample = dict(sample)
        
        step1_results = {
            'checks': {},
            'status': 'pass',
            'issues': []
        }
        
        try:
            # 分野別設定の取得と適用
            field_config = self.get_field_config(sample)
            if field_config:
                self.apply_field_config(field_config)
                processed_sample['detected_field'] = self.get_field_config(sample)
            
            # 必須フィールドチェック
            required_fields = ['question', 'think', 'answer']
            missing_fields = [f for f in required_fields if f not in sample or not sample[f]]
            
            if missing_fields:
                # 詳細なスキーマ情報をログ出力
                logger.warning(f"必須フィールドが不足 ({sample_id}): {missing_fields}")
                logger.warning(f"利用可能なフィールド ({sample_id}): {list(sample.keys())}")
                
                # 各必須フィールドの詳細チェック
                field_details = {}
                for field in required_fields:
                    if field not in sample:
                        field_details[field] = "フィールドが存在しません"
                    elif not sample[field]:
                        field_details[field] = f"フィールドが空です (型: {type(sample[field])}, 値: {repr(sample[field])})"
                    else:
                        field_details[field] = "OK"
                
                logger.warning(f"フィールド詳細 ({sample_id}): {json.dumps(field_details, ensure_ascii=False, indent=2)}")
                
                step1_results['checks']['schema'] = {
                    'status': 'fail',
                    'missing_fields': missing_fields,
                    'field_details': field_details,
                    'available_fields': list(sample.keys())
                }
                step1_results['status'] = 'fail'
                step1_results['issues'].append(f'missing_fields: {missing_fields}')
                
                # Step1結果を追加
                processed_sample['step1_status'] = step1_results['status']
                processed_sample['step1_issues'] = json.dumps(step1_results['issues'], ensure_ascii=False)
                processed_sample['step1_checks'] = json.dumps(step1_results['checks'], ensure_ascii=False)
                
                self.audit_logger.log_sample(sample_id, step1_results['status'], step1_results['issues'])
                return processed_sample
            
            # 正規化
            normalized = normalize_sample(sample)
            
            # ① question重複チェック
            dup_status, dup_sim, dup_id = self.dedup_detector.check_duplicate(
                normalized['question'], 
                sample_id
            )
            
            step1_results['checks']['dedup'] = {
                'status': dup_status,
                'similarity': dup_sim,
                'duplicate_of': dup_id
            }
            
            if dup_status == 'duplicate':
                step1_results['status'] = 'fail'
                step1_results['issues'].append(f'duplicate (sim={dup_sim:.3f}, id={dup_id})')
            elif dup_status == 'near_duplicate':
                if step1_results['status'] == 'pass':
                    step1_results['status'] = 'review'
                step1_results['issues'].append(f'near_duplicate (sim={dup_sim:.3f})')
            
            
            # ② テンプレートフレーズ検出（think部分）
            if 'think' in normalized and normalized['think']:
                template_results = self.template_detector.detect_templates(normalized['think'])
                step1_results['checks']['template'] = template_results
                
                if template_results['status'] == 'review':
                    if step1_results['status'] == 'pass':
                        step1_results['status'] = 'review'
                    step1_results['issues'].extend(template_results.get('issues', []))
                
                # 繰り返し検出
                repetition_results = self.template_detector.detect_repetition(normalized['think'])
                if repetition_results['has_repetition']:
                    if step1_results['status'] == 'pass':
                        step1_results['status'] = 'review'
                    step1_results['issues'].append(f"repetition_detected: {len(repetition_results['repeated_sentences'])} pairs")
                
                # ③ thinking部分の詳細チェック（新機能）
                thinking_checks = self.thinking_checker.run_all_checks(
                    normalized['think'], 
                    normalized.get('answer', '')
                )
                step1_results['checks']['thinking'] = thinking_checks
                
                # Fatal errorsがあればfail
                if thinking_checks.get('fatal_errors'):
                    step1_results['status'] = 'fail'
                    step1_results['issues'].extend(thinking_checks['fatal_errors'])
                # Flagsがあればreview
                elif thinking_checks.get('flags'):
                    if step1_results['status'] == 'pass':
                        step1_results['status'] = 'review'
                    step1_results['issues'].extend(thinking_checks['flags'])
                
                # ④ 矛盾検出（軽量版）
                contradictions = self.contradiction_detector.detect_contradictions(normalized['think'])
                if contradictions:
                    contradiction_score = self.contradiction_detector.calculate_contradiction_score(contradictions)
                    step1_results['checks']['contradictions'] = {
                        'score': contradiction_score,
                        'count': len(contradictions),
                        'types': [c.get('type') for c in contradictions]
                    }
                    
                    if contradiction_score >= 0.8:  # 明らかな矛盾
                        step1_results['status'] = 'fail'
                        step1_results['issues'].append(f'contradiction_detected (score={contradiction_score:.2f})')
                    elif contradiction_score >= 0.5:  # 疑わしい
                        if step1_results['status'] == 'pass':
                            step1_results['status'] = 'review'
                        step1_results['issues'].append(f'possible_contradiction (score={contradiction_score:.2f})')
                
                # ⑤ 数学的検証（問題文に数式が含まれる場合）
                if 'question' in normalized:
                    math_patterns = self.math_validator.detect_calculation_patterns(normalized['question'])
                    if math_patterns:
                        math_validation = self.math_validator.validate_calculations(normalized['think'])
                        step1_results['checks']['math'] = math_validation
                        
                        if math_validation.get('errors'):
                            step1_results['status'] = 'fail'
                            step1_results['issues'].append(f'math_error: {len(math_validation["errors"])} errors')
                        elif math_validation.get('warnings'):
                            if step1_results['status'] == 'pass':
                                step1_results['status'] = 'review'
                            step1_results['issues'].append(f'math_warning: {len(math_validation["warnings"])} warnings')
            
            # Step1結果を追加
            processed_sample['step1_status'] = step1_results['status']
            processed_sample['step1_issues'] = json.dumps(step1_results['issues'], ensure_ascii=False)
            processed_sample['step1_checks'] = json.dumps(step1_results['checks'], ensure_ascii=False)
            
            # 監査ログ記録
            self.audit_logger.log_sample(sample_id, step1_results['status'], step1_results['issues'], step1_results['checks'])
            
            # インデックスに追加（次のサンプルの重複チェック用）
            self.dedup_detector.add_to_index(normalized['question'], sample_id)
            
        except Exception as e:
            # 詳細なエラー情報を収集
            error_type = type(e).__name__
            error_msg = str(e)
            
            # KeyError特有の詳細情報を追加
            if isinstance(e, KeyError):
                missing_key = str(e).strip("'\"")
                logger.error(f"KeyError発生 ({sample_id}): キー '{missing_key}' が見つかりません")
                
                # データ構造の情報を出力
                available_keys = list(sample.keys()) if isinstance(sample, dict) else "データがdict型ではありません"
                logger.error(f"利用可能なキー ({sample_id}): {available_keys}")
                
                # サンプルデータの一部を安全に出力（機密情報を避けるため最初の100文字のみ）
                try:
                    sample_preview = {}
                    for key, value in sample.items():
                        if isinstance(value, str):
                            sample_preview[key] = value[:100] + "..." if len(value) > 100 else value
                        else:
                            sample_preview[key] = str(type(value))
                    logger.error(f"サンプルデータ構造 ({sample_id}): {json.dumps(sample_preview, ensure_ascii=False, indent=2)}")
                except Exception as preview_error:
                    logger.error(f"サンプルデータのプレビュー取得エラー ({sample_id}): {preview_error}")
                
                # 詳細なエラーメッセージを作成
                detailed_error_msg = f"KeyError: キー '{missing_key}' が見つかりません。利用可能なキー: {available_keys}"
            else:
                logger.error(f"サンプル処理エラー ({sample_id}): {error_type}: {error_msg}")
                detailed_error_msg = f"{error_type}: {error_msg}"
            
            # スタックトレースをログに出力
            logger.error(f"スタックトレース ({sample_id}):\n{traceback.format_exc()}")
            
            # 監査ログに詳細なエラー情報を記録
            extra_debug_info = {
                "original_error": error_msg,
                "traceback": traceback.format_exc()
            }
            
            # KeyErrorの場合は追加の診断情報
            if isinstance(e, KeyError):
                extra_debug_info.update({
                    "missing_key": str(e).strip("'\""),
                    "available_keys": list(sample.keys()) if isinstance(sample, dict) else [],
                    "sample_type": str(type(sample))
                })
            
            self.audit_logger.log_error(sample_id, detailed_error_msg, error_type, extra_debug_info)
            
            # エラー時の処理結果
            processed_sample['step1_status'] = 'error'
            processed_sample['step1_issues'] = json.dumps([f'processing_error: {detailed_error_msg}'], ensure_ascii=False)
            processed_sample['step1_error'] = detailed_error_msg
            processed_sample['step1_error_type'] = error_type
            
            # KeyErrorの場合は追加の診断情報も記録
            if isinstance(e, KeyError):
                processed_sample['step1_missing_key'] = str(e).strip("'\"")
                processed_sample['step1_available_keys'] = json.dumps(list(sample.keys()) if isinstance(sample, dict) else [], ensure_ascii=False)
        
        return processed_sample
    
    def process_dataset(self, dataset: Dataset, batch_size: int = 1000) -> Dataset:
        """
        データセット全体の処理
        
        Args:
            dataset: 入力データセット
            batch_size: バッチサイズ
            
        Returns:
            処理済みデータセット
        """
        total = len(dataset)
        processed_samples = []
        
        # 進捗トラッカー
        tracker = ProgressTracker(total, "Processing samples (Step1)")
        
        # メイン処理ループ
        for idx, sample in enumerate(dataset):
            # サンプルIDの安全な取得または生成
            try:
                if isinstance(sample, dict):
                    sample_id = sample.get('data_id', f"sample_{idx:06d}")
                else:
                    logger.warning(f"サンプル {idx} がdict型ではありません。型: {type(sample)}")
                    sample_id = f"sample_{idx:06d}"
                    # dict型でない場合は変換を試行
                    if hasattr(sample, '__dict__'):
                        sample = sample.__dict__
                    else:
                        logger.error(f"サンプル {idx} をdict型に変換できません")
                        continue
            except Exception as e:
                logger.error(f"サンプルID取得エラー (index {idx}): {e}")
                sample_id = f"sample_{idx:06d}_error"
            
            # サンプル処理
            processed_sample = self.process_sample(sample, sample_id)
            processed_samples.append(processed_sample)
            
            # 進捗更新
            tracker.update()
            
            # 中間保存
            if (idx + 1) % self.config['pipeline'].get('save_interval', 5000) == 0:
                logger.info(f"中間保存: {idx + 1}/{total}")
                self._save_intermediate(processed_samples, idx + 1)
        
        tracker.finish()
        
        # Datasetに変換
        return Dataset.from_pandas(pd.DataFrame(processed_samples))
    
    def _save_intermediate(self, samples: List[Dict], count: int):
        """中間結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.step1_logs_dir / f"intermediate_{count}_{timestamp}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"中間結果保存: {filename}")
    
    def run(self, input_dataset: str, output_name: Optional[str] = None, sample_size: Optional[int] = None, hf_dataset_name: Optional[str] = None):
        """
        パイプライン実行
        
        Args:
            input_dataset: 入力データセット名（必須）
            output_name: 出力名（デフォルト: step1_output）
            sample_size: サンプリングサイズ（Noneの場合は全データ）
            hf_dataset_name: HuggingFaceデータセット名（アップロード用）
        """
        # データセット名
        input_name = input_dataset
        output_name = output_name or 'step1_output'
        
        logger.info(f"Step1: データセット読み込み中: {input_name}")
        log_memory_usage("Before dataset loading")
        
        # データセット読み込み
        try:
            dataset = load_dataset(input_name, split='train')
            log_memory_usage("After dataset loading")
            original_size = len(dataset)
            logger.info(f"元のサンプル数: {original_size:,}")
            
            # サンプリング処理
            if sample_size is not None and sample_size < original_size:
                logger.info(f"ランダムサンプリング実行: {sample_size:,} サンプルを取得")
                
                # ランダムシードの設定
                random_seed = self.config.get('sampling', {}).get('random_seed', 42)
                random.seed(random_seed)
                
                # ランダムインデックスを生成
                indices = list(range(original_size))
                sampled_indices = sorted(random.sample(indices, sample_size))
                
                # サンプリング実行
                dataset = dataset.select(sampled_indices)
                log_memory_usage("After sampling")
                logger.info(f"サンプリング後のサンプル数: {len(dataset):,}")
            else:
                logger.info(f"全データを使用: {original_size:,} サンプル")
                
        except Exception as e:
            log_memory_usage("Dataset loading failed")
            logger.error(f"データセット読み込みエラー: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # バッチサイズ
        batch_size = self.config.get('pipeline', {}).get('batch_size', 1000)
        
        # 処理実行
        logger.info("Step1: フィルタリング処理開始...")
        filtered_dataset = self.process_dataset(dataset, batch_size)
        
        # サマリー出力
        samples_list = [dict(row) for row in filtered_dataset]
        
        # ステータス集計
        status_counts = {'pass': 0, 'review': 0, 'fail': 0, 'error': 0}
        fail_reasons = {}  # Fail原因別の集計
        
        for sample in samples_list:
            status = sample.get('step1_status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
            
            # Failの場合は原因を集計
            if status == 'fail':
                issues_json = sample.get('step1_issues', '[]')
                try:
                    issues = json.loads(issues_json)
                    for issue in issues:
                        # 原因を分類
                        reason_category = self._categorize_fail_reason(issue)
                        fail_reasons[reason_category] = fail_reasons.get(reason_category, 0) + 1
                except json.JSONDecodeError:
                    fail_reasons['parse_error'] = fail_reasons.get('parse_error', 0) + 1
        
        total = len(samples_list)
        logger.info("="*60)
        logger.info("Step1 処理結果サマリー")
        if sample_size is not None and sample_size < original_size:
            logger.info(f"サンプリング: {sample_size:,} / {original_size:,} ({sample_size/original_size:.1%})")
        logger.info(f"処理サンプル数: {total:,}")
        logger.info(f"  Pass:   {status_counts['pass']:,} ({status_counts['pass']/total:.1%})")
        logger.info(f"  Review: {status_counts['review']:,} ({status_counts['review']/total:.1%})")
        logger.info(f"  Fail:   {status_counts['fail']:,} ({status_counts['fail']/total:.1%})")
        logger.info(f"  Error:  {status_counts['error']:,} ({status_counts['error']/total:.1%})")
        
        # Fail内訳を表示
        if status_counts['fail'] > 0:
            logger.info("")
            logger.info("Fail内訳:")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True):
                percentage = count / status_counts['fail'] * 100
                logger.info(f"  {reason}: {count:,} ({percentage:.1f}%)")
        
        logger.info("="*60)
        
        # 監査ログ保存
        self.audit_logger.save_summary()
        
        # 共有データとして保存（Step2への入力データ）
        output_path = self.shared_data_dir / f"{output_name}.jsonl"
        logger.info(f"Step1結果を共有データとして保存（Step2への入力用）: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples_list:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # HuggingFaceへのアップロード（--hf-dataset引数が指定された場合）
        if hf_dataset_name:
            logger.info(f"HuggingFaceへのアップロード開始: {hf_dataset_name}")
            try:
                # プライベート設定の取得
                hf_config = self.config.get('huggingface', {})
                private = hf_config.get('private', True)
                
                # データセットをHuggingFaceにアップロード
                filtered_dataset.push_to_hub(hf_dataset_name, private=private)
                logger.info(f"HuggingFaceへのアップロード完了: {hf_dataset_name}")
            except Exception as e:
                logger.error(f"HuggingFaceアップロードエラー: {e}")
                
                # エラー時の追加バックアップ保存（タイムスタンプ付き）
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = self.shared_data_dir / f"step1_backup_{timestamp}.jsonl"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    for sample in samples_list:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                logger.info(f"エラー時バックアップ保存: {backup_path}")
        else:
            logger.info("HuggingFaceアップロードはスキップ（--hf-datasetが指定されていません）")
        
        return filtered_dataset


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description='Step1: 基本フィルタリングパイプライン')
    
    parser.add_argument(
        '--config',
        type=str,
        default='./config/step1_config.yaml',
        help='設定ファイルパス'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='入力データセット名（HuggingFace形式）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='step1_output',
        help='出力ファイル名'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='ログレベル'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='ランダムサンプリングサイズ（指定しない場合は全データを使用）'
    )
    
    parser.add_argument(
        '--hf-dataset',
        type=str,
        default=None,
        help='HuggingFaceデータセット名（アップロード用、例: username/dataset-name）'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    log_memory_usage("After logging setup")
    
    try:
        # パイプライン実行
        log_memory_usage("Before pipeline creation")
        pipeline = Step1Pipeline(config_path=args.config)
        log_memory_usage("After pipeline creation")
        
        pipeline.run(
            input_dataset=args.input, 
            output_name=args.output, 
            sample_size=args.sample_size,
            hf_dataset_name=args.hf_dataset
        )
        log_memory_usage("After pipeline execution")
        
    except Exception as e:
        log_memory_usage("Pipeline execution failed")
        print(f"[ERROR] Pipeline failed: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
