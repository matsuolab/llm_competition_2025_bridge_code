#!/usr/bin/env python3
"""
Monitor task queue progress and merge results when complete
タスクキューの進捗を監視し、完了時に結果を統合
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskQueueMonitor:
    """タスクキューの進捗を監視"""
    
    def __init__(self, queue_dir: str, output_base_dir: str):
        self.queue_dir = Path(queue_dir)
        self.output_base_dir = Path(output_base_dir)
        self.task_status_file = self.queue_dir / "task_status.json"
        self.worker_registry_file = self.queue_dir / "worker_registry.json"
        
    def get_progress(self) -> Dict[str, Any]:
        """現在の進捗を取得"""
        if not self.task_status_file.exists():
            return None
        
        try:
            with open(self.task_status_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def get_active_workers(self) -> List[str]:
        """アクティブなワーカーのリストを取得"""
        if not self.worker_registry_file.exists():
            return []
        
        try:
            with open(self.worker_registry_file, 'r') as f:
                registry = json.load(f)
                
            # 2分以内にハートビートがあるワーカー
            active = []
            current_time = datetime.now()
            for worker_id, info in registry.get('workers', {}).items():
                last_heartbeat = datetime.fromisoformat(info['last_heartbeat'])
                if (current_time - last_heartbeat).total_seconds() < 120:
                    active.append(worker_id)
            
            return active
        except:
            return []
    
    def find_worker_outputs(self) -> List[Path]:
        """ワーカーの出力ディレクトリを検索"""
        pattern = f"{self.output_base_dir}_worker_*"
        return sorted(self.output_base_dir.parent.glob(pattern))
    
    def monitor_until_complete(self, check_interval: int = 30):
        """タスクが完了するまで監視"""
        
        logger.info(f"Monitoring task queue: {self.queue_dir}")
        logger.info(f"Output directory pattern: {self.output_base_dir}_worker_*")
        
        last_progress = None
        stale_count = 0
        max_stale_checks = 10  # 5分間進捗なしで終了判定
        
        while True:
            progress = self.get_progress()
            
            if progress is None:
                logger.warning("No progress file found. Waiting for initialization...")
                time.sleep(check_interval)
                continue
            
            # 進捗を表示
            total = progress.get('total', 0)
            completed = progress.get('completed', 0)
            failed = progress.get('failed', 0)
            processing = progress.get('processing', 0)
            pending = progress.get('pending', 0)
            percent = progress.get('progress_percent', 0)
            
            active_workers = self.get_active_workers()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {completed}/{total} ({percent:.1f}%)")
            logger.info(f"  Completed: {completed}")
            logger.info(f"  Processing: {processing}")
            logger.info(f"  Pending: {pending}")
            logger.info(f"  Failed: {failed}")
            logger.info(f"  Active workers: {len(active_workers)}")
            logger.info(f"{'='*60}")
            
            # 完了判定
            if total > 0 and completed + failed >= total:
                logger.info("✅ All tasks completed!")
                return True
            
            # 進捗なし判定
            if pending == 0 and processing == 0 and len(active_workers) == 0:
                stale_count += 1
                logger.warning(f"No active processing detected ({stale_count}/{max_stale_checks})")
                
                if stale_count >= max_stale_checks:
                    logger.info("No workers active and no pending tasks. Assuming completion.")
                    return True
            else:
                stale_count = 0
            
            # 進捗の変化を記録
            last_progress = progress
            
            time.sleep(check_interval)
    
    def merge_results(self):
        """結果を統合"""
        worker_dirs = self.find_worker_outputs()
        
        if not worker_dirs:
            logger.error(f"No worker output directories found matching: {self.output_base_dir}_worker_*")
            return False
        
        logger.info(f"Found {len(worker_dirs)} worker output directories")
        
        # マージ先ディレクトリ
        merged_dir = self.output_base_dir.parent / f"{self.output_base_dir.name}_merged"
        merged_dir.mkdir(exist_ok=True, parents=True)
        
        # 各ワーカーのJSONLファイルを統合
        total_problems = 0
        merged_file = merged_dir / "instruction_dataset.jsonl"
        
        with open(merged_file, 'w', encoding='utf-8') as outf:
            for worker_dir in worker_dirs:
                jsonl_file = worker_dir / "instruction_dataset.jsonl"
                if jsonl_file.exists():
                    logger.info(f"  Merging: {jsonl_file}")
                    with open(jsonl_file, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            outf.write(line)
                            total_problems += 1
        
        logger.info(f"✅ Merged {total_problems} problems into {merged_file}")
        
        # 統計情報を統合
        stats = {
            'total_problems': total_problems,
            'worker_count': len(worker_dirs),
            'merged_at': datetime.now().isoformat(),
            'queue_dir': str(self.queue_dir),
            'worker_dirs': [str(d) for d in worker_dirs]
        }
        
        stats_file = merged_dir / "merge_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Statistics saved to {stats_file}")
        
        # パーミッション設定
        try:
            merged_dir.chmod(0o777)
            merged_file.chmod(0o666)
            stats_file.chmod(0o666)
        except:
            pass
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Monitor task queue and merge results")
    parser.add_argument("--queue-dir", required=True, help="Task queue directory")
    parser.add_argument("--output-base", required=True, help="Base output directory")
    parser.add_argument("--check-interval", type=int, default=30, 
                       help="Check interval in seconds (default: 30)")
    parser.add_argument("--no-merge", action="store_true",
                       help="Only monitor, don't merge results")
    
    args = parser.parse_args()
    
    monitor = TaskQueueMonitor(args.queue_dir, args.output_base)
    
    # 完了まで監視
    completed = monitor.monitor_until_complete(args.check_interval)
    
    if completed and not args.no_merge:
        # 結果を統合
        logger.info("\n🔄 Starting result merge...")
        success = monitor.merge_results()
        
        if success:
            logger.info("\n✅ All done! Results have been merged.")
        else:
            logger.error("\n❌ Failed to merge results")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())