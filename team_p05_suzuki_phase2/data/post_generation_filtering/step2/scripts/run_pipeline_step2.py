"""
Step2: LLMベースの高度な検証処理
Step1の結果を受け取り、DeepSeek-R1による検証を実行
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

# データセット関連
from datasets import load_dataset, Dataset
import pandas as pd

# 内部モジュール
sys.path.append(str(Path(__file__).parent))
from util_logging import setup_logging, AuditLogger, ProgressTracker
from llm_validator import LLMValidator, cleanup_llm_resources
import atexit

logger = logging.getLogger(__name__)


class Step2Pipeline:
    """Step2: LLMベースの高度な検証パイプライン"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 共有データディレクトリ
        self.shared_data_dir = Path(__file__).parent.parent.parent / 'shared_data'
        
        # Step2ログディレクトリ
        self.step2_logs_dir = Path(__file__).parent.parent / 'logs'
        self.step2_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 検証機能の設定を読み込み（デフォルト値付き）
        self.validation_config = self.config.get('validation', {})
        self.enable_logical_consistency = self.validation_config.get('logical_consistency', True)
        self.enable_contradiction_detection = self.validation_config.get('contradiction_detection', True)
        self.enable_answer_quality = self.validation_config.get('answer_quality', True)
        
        # 有効な検証機能をログ出力
        logger.info("検証機能の設定:")
        logger.info(f"  - 論理的整合性チェック: {'有効' if self.enable_logical_consistency else '無効'}")
        logger.info(f"  - 意味的矛盾検出: {'有効' if self.enable_contradiction_detection else '無効'}")
        logger.info(f"  - 回答品質チェック: {'有効' if self.enable_answer_quality else '無効'}")
        
        # コンポーネント初期化
        logger.info("Step2コンポーネント初期化中...")
        self.llm_validator = LLMValidator(self.config)
        
        # 監査ログ（タイムスタンプ付きファイル名）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.audit_logger = AuditLogger(
            self.step2_logs_dir / f'audit_log_{timestamp}.jsonl',
            self.step2_logs_dir / f'summary_{timestamp}.json'
        )
        self.audit_logger.set_config_hash(self.config)
        
        logger.info("Step2パイプライン初期化完了")
        
        # 終了時のクリーンアップを登録
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """パイプラインのクリーンアップ処理"""
        try:
            # LLMバリデーターのクリーンアップ
            if hasattr(self, 'llm_validator') and self.llm_validator:
                self.llm_validator.cleanup()
            logger.info("Step2パイプラインのクリーンアップ完了")
        except Exception as e:
            logger.warning(f"クリーンアップ中の警告: {e}")
    
    def process_sample(self, sample: Dict, sample_id: str) -> Dict:
        """
        単一サンプルの処理（Step2）
        
        Args:
            sample: データサンプル（Step1結果を含む）
            sample_id: サンプルID
            
        Returns:
            処理済みサンプル（Step2結果を含む）
        """
        step2_results = {
            'checks': {},
            'status': 'pass',
            'issues': []
        }
        
        try:
            # Step1でfailしたサンプルはスキップ
            step1_status = sample.get('step1_status', 'unknown')
            if step1_status == 'fail':
                sample['step2_status'] = 'skipped'
                sample['step2_reason'] = 'step1_failed'
                return sample
            
            # 必須フィールドチェック
            required_fields = ['question', 'think', 'answer']
            missing_fields = [f for f in required_fields if f not in sample or not sample[f]]
            
            if missing_fields:
                sample['step2_status'] = 'error'
                sample['step2_issues'] = json.dumps([f'missing_fields: {missing_fields}'], ensure_ascii=False)
                return sample
            
            question = sample['question']
            think = sample['think']
            answer = sample['answer']
            
            # LLMによる検証（設定に基づいて実行）
            if self.llm_validator.llm_model:
                llm_validation_results = {}
                
                # 1. 論理的整合性チェック（常に有効）
                if self.enable_logical_consistency:
                    is_valid, issue, confidence = self.llm_validator.validate_logical_consistency(
                        question, think, answer
                    )
                    
                    if not is_valid:
                        step2_results['status'] = 'fail'
                        step2_results['issues'].append(f"Logical inconsistency: {issue}")
                    
                    llm_validation_results['logical_consistency'] = {
                        'valid': is_valid, 'issue': issue, 'confidence': confidence
                    }
                
                # 2. 意味的矛盾検出（設定により有効/無効）
                if self.enable_contradiction_detection:
                    contradictions = self.llm_validator.detect_contradictions(think)
                    if contradictions:
                        step2_results['status'] = 'fail'
                        step2_results['issues'].extend([f"Contradiction: {c}" for c in contradictions[:3]])
                    
                    llm_validation_results['contradictions'] = contradictions[:3] if contradictions else []
                else:
                    llm_validation_results['contradictions'] = 'skipped'
                
                # 3. 回答品質チェック（設定により有効/無効）
                if self.enable_answer_quality:
                    answer_type = sample.get('answer_type', 'freeform')
                    quality, quality_issue = self.llm_validator.check_answer_quality(
                        question, answer, answer_type
                    )
                    
                    if quality == "poor":
                        if step2_results['status'] == 'pass':
                            step2_results['status'] = 'review'
                        step2_results['issues'].append(f"Answer quality: {quality_issue}")
                    
                    llm_validation_results['answer_quality'] = {
                        'level': quality, 'issue': quality_issue
                    }
                else:
                    llm_validation_results['answer_quality'] = 'skipped'
                
                # 結果を記録
                step2_results['checks']['llm_validation'] = llm_validation_results
            
            # 最終ステータス決定
            # Step1がreviewでStep2がpassの場合でも、reviewを維持
            if step1_status == 'review' and step2_results['status'] == 'pass':
                step2_results['status'] = 'review'
                step2_results['issues'].append('step1_review')
            
            # Step2結果を追加
            sample['step2_status'] = step2_results['status']
            sample['step2_issues'] = json.dumps(step2_results['issues'], ensure_ascii=False)
            sample['step2_checks'] = json.dumps(step2_results['checks'], ensure_ascii=False)
            
            # 最終ステータス（Step1とStep2の組み合わせ）
            if step2_results['status'] == 'fail' or step1_status == 'fail':
                sample['final_status'] = 'fail'
            elif step2_results['status'] == 'review' or step1_status == 'review':
                sample['final_status'] = 'review'
            else:
                sample['final_status'] = 'pass'
            
            # 監査ログ記録
            self.audit_logger.log_sample(sample_id, step2_results['status'], step2_results['issues'], step2_results['checks'])
            
        except Exception as e:
            logger.error(f"サンプル処理エラー ({sample_id}): {e}")
            self.audit_logger.log_error(sample_id, str(e), type(e).__name__)
            
            # エラー時
            sample['step2_status'] = 'error'
            sample['step2_issues'] = json.dumps([f'processing_error: {str(e)}'], ensure_ascii=False)
            sample['step2_error'] = str(e)
            sample['final_status'] = 'error'
        
        return sample
    
    def process_dataset(self, samples: List[Dict], batch_size: int = 100) -> Dataset:
        """
        データセット全体の処理
        
        Args:
            samples: Step1処理済みサンプルのリスト
            batch_size: バッチサイズ（LLM処理のため小さめ）
            
        Returns:
            処理済みデータセット
        """
        total = len(samples)
        processed_samples = []
        
        # 進捗トラッカー（5件ごとに更新、LLM処理は遅いため頻繁に表示）
        tracker = ProgressTracker(total, "Processing samples (Step2)", update_interval=5)
        
        logger.info(f"Step2処理開始: 総サンプル数 {total:,}件")
        
        # 有効な検証機能の確認
        active_validations = []
        if self.enable_logical_consistency:
            active_validations.append("論理的整合性")
        if self.enable_contradiction_detection:
            active_validations.append("意味的矛盾検出")
        if self.enable_answer_quality:
            active_validations.append("回答品質")
        
        logger.info(f"実行する検証: {', '.join(active_validations) if active_validations else 'なし'}")
        
        # メイン処理ループ
        for idx, sample in enumerate(samples):
            # サンプルIDの取得または生成
            sample_id = sample.get('data_id', f"sample_{idx:06d}")
            
            # 処理開始のログ（LLM処理が遅い場合のため）
            if idx % 10 == 0 and idx > 0:
                logger.debug(f"サンプル {sample_id} の処理を開始中...")
            
            # サンプル処理
            processed_sample = self.process_sample(sample, sample_id)
            processed_samples.append(processed_sample)
            
            # 進捗更新
            tracker.update()
            
            # 中間保存（LLM処理は重いため頻度を上げる）
            save_interval = self.config['pipeline'].get('save_interval', 1000)
            if (idx + 1) % save_interval == 0:
                logger.info(f"中間保存: {idx + 1}/{total}")
                self._save_intermediate(processed_samples, idx + 1)
        
        tracker.finish()
        
        # Datasetに変換
        return Dataset.from_pandas(pd.DataFrame(processed_samples))
    
    def _save_intermediate(self, samples: List[Dict], count: int):
        """中間結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.step2_logs_dir / f"intermediate_{count}_{timestamp}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"中間結果保存: {filename}")
    
    def load_step1_results(self, input_file: str) -> List[Dict]:
        """
        Step1の結果を読み込み（shared_dataから）
        
        Args:
            input_file: Step1結果ファイル名
            
        Returns:
            サンプルのリスト
        """
        input_path = self.shared_data_dir / f"{input_file}.jsonl"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Step1結果ファイルが見つかりません: {input_path}")
        
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line)
                    
                    # 監査ログファイルの誤使用を検出
                    if 'timestamp' in sample and 'ng_list' in sample and 'question' not in sample:
                        logger.error(f"エラー: 監査ログファイルが誤って使用されています！")
                        logger.error(f"正しいStep1出力ファイル（shared_data/{input_file}.jsonl）を使用してください。")
                        logger.error(f"監査ログファイル（step1/logs/audit_log_*.jsonl）ではありません。")
                        raise ValueError(f"監査ログファイルが誤って使用されています。正しいStep1出力ファイルを使用してください。")
                    
                    # 必須フィールドの確認
                    required_fields = ['question', 'think', 'answer']
                    if not all(field in sample for field in required_fields):
                        logger.warning(f"行 {line_num}: 必須フィールドが不足しています。利用可能なフィールド: {list(sample.keys())}")
                    
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.error(f"行 {line_num} のJSONパースエラー: {e}")
                    continue
        
        logger.info(f"Step1結果を読み込み: {len(samples)} samples from {input_path}")
        
        # 最初のサンプルで診断情報を出力
        if samples:
            first_sample = samples[0]
            logger.info(f"最初のサンプルのフィールド: {list(first_sample.keys())[:10]}...")  # 最初の10フィールドのみ表示
            has_original = all(field in first_sample for field in ['question', 'think', 'answer'])
            has_step1 = all(field in first_sample for field in ['step1_status', 'step1_issues'])
            logger.info(f"元のデータフィールド（question, think, answer）: {'✓' if has_original else '✗'}")
            logger.info(f"Step1検証結果（step1_status, step1_issues）: {'✓' if has_step1 else '✗'}")
        
        return samples
    
    def run(self, input_file: Optional[str] = None, output_dataset: Optional[str] = None):
        """
        パイプライン実行
        
        Args:
            input_file: Step1結果ファイル名（デフォルト: step1_output）
            output_dataset: 出力データセット名（デフォルト: config値）
        """
        # 入力ファイル名
        input_file = input_file or 'step1_output'
        
        logger.info(f"Step2: Step1結果を読み込み中: {input_file}")
        
        # Step1結果読み込み
        samples = self.load_step1_results(input_file)
        logger.info(f"サンプル数: {len(samples)}")
        
        # バッチサイズ（LLM処理のため小さめ）
        batch_size = self.config['dataset'].get('batch_size', 100)
        
        # 処理実行
        logger.info("Step2: LLM検証処理開始...")
        filtered_dataset = self.process_dataset(samples, batch_size)
        
        # サマリー出力
        samples_list = [dict(row) for row in filtered_dataset]
        
        # ステータス集計
        final_counts = {'pass': 0, 'review': 0, 'fail': 0, 'error': 0}
        step2_counts = {'pass': 0, 'review': 0, 'fail': 0, 'error': 0, 'skipped': 0}
        
        for sample in samples_list:
            final_status = sample.get('final_status', 'unknown')
            step2_status = sample.get('step2_status', 'unknown')
            
            if final_status in final_counts:
                final_counts[final_status] += 1
            if step2_status in step2_counts:
                step2_counts[step2_status] += 1
        
        total = len(samples_list)
        
        logger.info("="*60)
        logger.info("Step2 処理結果サマリー")
        logger.info(f"総サンプル数: {total:,}")
        logger.info("")
        logger.info("Step2ステータス:")
        logger.info(f"  Pass:    {step2_counts['pass']:,} ({step2_counts['pass']/total:.1%})")
        logger.info(f"  Review:  {step2_counts['review']:,} ({step2_counts['review']/total:.1%})")
        logger.info(f"  Fail:    {step2_counts['fail']:,} ({step2_counts['fail']/total:.1%})")
        logger.info(f"  Error:   {step2_counts['error']:,} ({step2_counts['error']/total:.1%})")
        logger.info(f"  Skipped: {step2_counts['skipped']:,} ({step2_counts['skipped']/total:.1%})")
        logger.info("")
        logger.info("最終ステータス:")
        logger.info(f"  Pass:   {final_counts['pass']:,} ({final_counts['pass']/total:.1%})")
        logger.info(f"  Review: {final_counts['review']:,} ({final_counts['review']/total:.1%})")
        logger.info(f"  Fail:   {final_counts['fail']:,} ({final_counts['fail']/total:.1%})")
        logger.info(f"  Error:  {final_counts['error']:,} ({final_counts['error']/total:.1%})")
        logger.info("="*60)
        
        # 監査ログ保存
        self.audit_logger.save_summary()
        
        # 共有データとして保存（Step2結果の共有用）
        output_name_file = output_dataset or 'step2_output'
        # データセット名からファイル名を抽出（'/'が含まれる場合は最後の部分を使用）
        if '/' in output_name_file:
            output_name_file = output_name_file.split('/')[-1]
        
        shared_output_path = self.shared_data_dir / f"{output_name_file}.jsonl"
        logger.info(f"Step2結果を共有データとして保存: {shared_output_path}")
        
        with open(shared_output_path, 'w', encoding='utf-8') as f:
            for sample in samples_list:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # HuggingFaceへのアップロード
        output_name = output_dataset
        logger.info(f"最終データセットアップロード中: {output_name}")
        try:
            filtered_dataset.push_to_hub(output_name, private=True)
            logger.info("アップロード完了")
        except Exception as e:
            logger.error(f"アップロードエラー: {e}")
            
            # エラー時の追加バックアップ保存（タイムスタンプ付き）
            backup_path = self.shared_data_dir / f"final_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(backup_path, 'w', encoding='utf-8') as f:
                for sample in samples_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"エラー時バックアップ保存: {backup_path}")
        
        # 明示的なクリーンアップ
        self.cleanup()
        
        return filtered_dataset


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description='Step2: LLMベースの高度な検証パイプライン')
    
    parser.add_argument(
        '--config',
        type=str,
        default='./config/step2_config.yaml',
        help='設定ファイルパス'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='step1_output',
        help='Step1結果ファイル名'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='出力データセット名（HuggingFace形式）'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='ログレベル'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    
    # パイプライン実行
    pipeline = Step2Pipeline(config_path=args.config)
    try:
        pipeline.run(input_file=args.input, output_dataset=args.output)
    finally:
        # 明示的なクリーンアップ
        cleanup_llm_resources()
        logger.info("プログラム終了処理完了")


if __name__ == '__main__':
    main()
