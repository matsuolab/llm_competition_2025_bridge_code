"""
Dynamic Task Queue for Fault-Tolerant Multi-Node Processing
動的タスクキューによる耐障害性のある複数ノード処理
"""

import os
import json
import time
import fcntl
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import random

logger = logging.getLogger(__name__)

class DynamicTaskQueue:
    """
    動的にタスクを管理し、ノードの追加・削除に対応するキュー
    ファイルベースで実装し、外部DBへの依存を避ける
    """
    
    def __init__(self, queue_dir: str = None):
        """
        Args:
            queue_dir: タスクキューを管理するディレクトリ
        """
        if queue_dir is None:
            # 環境変数から取得、なければデフォルト
            queue_dir = os.getenv('TASK_QUEUE_DIR', 
                                 '/home/Competition2025/P05/shareP05/data_generation/task_queue/default')
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルパス
        self.task_list_file = self.queue_dir / "task_list.json"
        self.task_status_file = self.queue_dir / "task_status.json"
        self.worker_registry_file = self.queue_dir / "worker_registry.json"
        self.checkpoint_dir = self.queue_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # ワーカー情報
        self.worker_id = self._generate_worker_id()
        self.heartbeat_interval = 30  # 30秒ごとにハートビート
        self.task_timeout = 600  # 10分でタスクタイムアウト
        
        logger.info(f"DynamicTaskQueue initialized: worker_id={self.worker_id}")
    
    def _generate_worker_id(self) -> str:
        """ワーカーIDを生成"""
        node_id = os.getenv('SLURM_NODEID', '0')
        job_id = os.getenv('SLURM_JOB_ID', str(os.getpid()))
        hostname = os.getenv('HOSTNAME', 'localhost')
        
        # ユニークなワーカーIDを生成
        worker_string = f"{hostname}_{job_id}_{node_id}_{os.getpid()}"
        worker_hash = hashlib.md5(worker_string.encode()).hexdigest()[:8]
        
        return f"worker_{node_id}_{worker_hash}"
    
    def initialize_task_list(self, items: List[Any], force: bool = False) -> bool:
        """
        タスクリストを初期化（最初のワーカーのみ実行）
        
        Args:
            items: 処理対象のアイテムリスト
            force: 強制的に再初期化
        
        Returns:
            初期化が成功したかどうか
        """
        lock_file = self.queue_dir / ".init.lock"
        
        try:
            # アトミックなロック取得
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # 既存のタスクリストをチェック
                if self.task_list_file.exists() and not force:
                    logger.info("Task list already exists, skipping initialization")
                    return False
                
                # タスクリストを作成
                tasks = []
                for idx, item in enumerate(items):
                    task_id = f"task_{idx:06d}"
                    tasks.append({
                        'task_id': task_id,
                        'item_index': idx,
                        'item_data': item,
                        'status': 'pending',
                        'worker_id': None,
                        'started_at': None,
                        'completed_at': None,
                        'attempts': 0,
                        'last_error': None
                    })
                
                # 保存
                with open(self.task_list_file, 'w') as f:
                    json.dump({
                        'total_tasks': len(tasks),
                        'created_at': datetime.now().isoformat(),
                        'tasks': tasks
                    }, f, indent=2)
                
                # 初期ステータスファイルを作成
                self._update_global_status()
                
                logger.info(f"Initialized task list with {len(tasks)} tasks")
                return True
                
        except (IOError, OSError):
            # ロック取得失敗（他のワーカーが初期化中）
            logger.debug("Another worker is initializing the task list")
            
            # 初期化完了を待つ
            max_wait = 30
            for _ in range(max_wait):
                if self.task_list_file.exists():
                    return False
                time.sleep(1)
            
            raise RuntimeError("Task list initialization timeout")
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        次の未処理タスクを取得してロック
        
        Returns:
            タスク情報、またはNone（全タスク完了）
        """
        with self._file_lock(self.task_list_file):
            tasks_data = self._load_json(self.task_list_file)
            if not tasks_data:
                return None
            
            tasks = tasks_data.get('tasks', [])
            current_time = datetime.now()
            
            # 未処理またはタイムアウトしたタスクを探す
            for task in tasks:
                if task['status'] == 'pending':
                    # タスクを割り当て
                    task['status'] = 'processing'
                    task['worker_id'] = self.worker_id
                    task['started_at'] = current_time.isoformat()
                    task['attempts'] += 1
                    
                    # 保存
                    self._save_json(self.task_list_file, tasks_data)
                    self._register_worker()
                    
                    logger.info(f"Acquired task: {task['task_id']} (attempt {task['attempts']})")
                    return task
                
                elif task['status'] == 'processing':
                    # タイムアウトチェック
                    if task['started_at']:
                        started = datetime.fromisoformat(task['started_at'])
                        if (current_time - started).total_seconds() > self.task_timeout:
                            # タイムアウトしたタスクを再割り当て
                            logger.warning(f"Task {task['task_id']} timed out, reassigning")
                            task['status'] = 'pending'
                            task['last_error'] = f"Timeout after {self.task_timeout}s"
                            
                            # 保存して再帰
                            self._save_json(self.task_list_file, tasks_data)
                            return self.get_next_task()
            
            # 全タスク処理済みまたは処理中
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        タスクを完了としてマーク
        
        Args:
            task_id: タスクID
            result: タスクの結果
        
        Returns:
            成功したかどうか
        """
        with self._file_lock(self.task_list_file):
            tasks_data = self._load_json(self.task_list_file)
            if not tasks_data:
                return False
            
            tasks = tasks_data.get('tasks', [])
            
            for task in tasks:
                if task['task_id'] == task_id:
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.now().isoformat()
                    
                    # 結果を保存
                    result_file = self.checkpoint_dir / f"{task_id}_result.json"
                    self._save_json(result_file, result)
                    
                    # タスクリストを更新
                    self._save_json(self.task_list_file, tasks_data)
                    self._update_global_status()
                    
                    logger.info(f"Completed task: {task_id}")
                    return True
            
            logger.error(f"Task not found: {task_id}")
            return False
    
    def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """
        タスクを失敗としてマーク
        
        Args:
            task_id: タスクID
            error: エラーメッセージ
            retry: リトライ可能にするか
        
        Returns:
            成功したかどうか
        """
        with self._file_lock(self.task_list_file):
            tasks_data = self._load_json(self.task_list_file)
            if not tasks_data:
                return False
            
            tasks = tasks_data.get('tasks', [])
            
            for task in tasks:
                if task['task_id'] == task_id:
                    task['last_error'] = error
                    
                    if retry and task['attempts'] < 3:
                        # リトライ可能
                        task['status'] = 'pending'
                        logger.warning(f"Task {task_id} failed (attempt {task['attempts']}), will retry")
                    else:
                        # 最終失敗
                        task['status'] = 'failed'
                        task['completed_at'] = datetime.now().isoformat()
                        logger.error(f"Task {task_id} permanently failed: {error}")
                    
                    # 保存
                    self._save_json(self.task_list_file, tasks_data)
                    self._update_global_status()
                    return True
            
            return False
    
    def get_progress(self) -> Dict[str, Any]:
        """現在の進捗を取得"""
        status = self._load_json(self.task_status_file)
        if not status:
            return {
                'total': 0,
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0,
                'progress_percent': 0
            }
        return status
    
    def _register_worker(self):
        """ワーカーを登録・ハートビート更新"""
        with self._file_lock(self.worker_registry_file):
            registry = self._load_json(self.worker_registry_file) or {'workers': {}}
            
            registry['workers'][self.worker_id] = {
                'last_heartbeat': datetime.now().isoformat(),
                'hostname': os.getenv('HOSTNAME', 'unknown'),
                'pid': os.getpid(),
                'slurm_node': os.getenv('SLURM_NODEID', 'unknown'),
                'slurm_job': os.getenv('SLURM_JOB_ID', 'unknown')
            }
            
            # 古いワーカーをクリーンアップ
            current_time = datetime.now()
            dead_workers = []
            for worker_id, info in registry['workers'].items():
                last_heartbeat = datetime.fromisoformat(info['last_heartbeat'])
                if (current_time - last_heartbeat).total_seconds() > 120:  # 2分
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                del registry['workers'][worker_id]
                logger.debug(f"Removed dead worker: {worker_id}")
            
            self._save_json(self.worker_registry_file, registry)
    
    def cleanup_stale_tasks(self):
        """古いタスクやデッドワーカーのタスクをクリーンアップ"""
        with self._file_lock(self.task_list_file):
            tasks_data = self._load_json(self.task_list_file)
            if not tasks_data:
                return
            
            registry = self._load_json(self.worker_registry_file) or {'workers': {}}
            active_workers = set(registry.get('workers', {}).keys())
            
            tasks = tasks_data.get('tasks', [])
            cleaned = 0
            
            for task in tasks:
                if task['status'] == 'processing' and task['worker_id']:
                    # ワーカーが死んでいる場合
                    if task['worker_id'] not in active_workers:
                        task['status'] = 'pending'
                        task['last_error'] = 'Worker died'
                        cleaned += 1
            
            if cleaned > 0:
                self._save_json(self.task_list_file, tasks_data)
                logger.info(f"Cleaned up {cleaned} stale tasks")
    
    def _update_global_status(self):
        """グローバルステータスを更新"""
        tasks_data = self._load_json(self.task_list_file)
        if not tasks_data:
            return
        
        tasks = tasks_data.get('tasks', [])
        
        status_counts = {
            'pending': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0
        }
        
        for task in tasks:
            status = task.get('status', 'pending')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(tasks)
        done = status_counts['completed'] + status_counts['failed']
        progress = (done / total * 100) if total > 0 else 0
        
        status = {
            'total': total,
            **status_counts,
            'progress_percent': round(progress, 2),
            'updated_at': datetime.now().isoformat()
        }
        
        self._save_json(self.task_status_file, status)
    
    def _file_lock(self, file_path: Path):
        """ファイルロックコンテキストマネージャー"""
        import contextlib
        lock_path = file_path.with_suffix('.lock')
        
        @contextlib.contextmanager
        def lock_context():
            lock_file = None
            try:
                lock_file = open(lock_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
        
        return lock_context()
    
    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """JSONファイルを読み込み"""
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _save_json(self, file_path: Path, data: Dict):
        """JSONファイルに保存"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        try:
            file_path.chmod(0o666)
        except:
            pass


def process_with_queue(queue: DynamicTaskQueue, 
                       process_func: callable,
                       dataset: List[Any]) -> List[Dict]:
    """
    動的キューを使用してデータを処理
    
    Args:
        queue: DynamicTaskQueueインスタンス
        process_func: 処理関数
        dataset: データセット
    
    Returns:
        処理結果のリスト
    """
    # 最初のワーカーがタスクリストを初期化
    queue.initialize_task_list(dataset)
    
    # クリーンアップ（デッドワーカーのタスクを解放）
    queue.cleanup_stale_tasks()
    
    results = []
    
    while True:
        # 次のタスクを取得
        task = queue.get_next_task()
        
        if task is None:
            # 全タスク完了または処理中
            progress = queue.get_progress()
            
            if progress['pending'] == 0 and progress['processing'] == 0:
                # 全タスク完了
                logger.info("All tasks completed")
                break
            else:
                # 他のワーカーが処理中、少し待つ
                logger.info(f"Waiting for other workers... (processing: {progress['processing']})")
                time.sleep(10)
                queue.cleanup_stale_tasks()
                continue
        
        # タスクを処理
        try:
            logger.info(f"Processing {task['task_id']} (item {task['item_index']})")
            
            result = process_func(task['item_data'], task_index=task['item_index'])
            
            # 成功を記録
            queue.complete_task(task['task_id'], result)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {task['task_id']}: {e}")
            queue.fail_task(task['task_id'], str(e))
        
        # 定期的に進捗を表示
        if len(results) % 10 == 0:
            progress = queue.get_progress()
            logger.info(f"Progress: {progress['completed']}/{progress['total']} "
                       f"({progress['progress_percent']}%)")
    
    return results


# 使用例
if __name__ == "__main__":
    import sys
    
    # テスト用データセット
    test_dataset = [{'id': i, 'data': f'item_{i}'} for i in range(100)]
    
    # キューを初期化
    queue = DynamicTaskQueue()
    
    # 引数でアクションを選択
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == "init":
            # タスクリストを初期化
            queue.initialize_task_list(test_dataset, force=True)
            print("Task list initialized")
        
        elif action == "status":
            # ステータスを表示
            progress = queue.get_progress()
            print(f"Progress: {json.dumps(progress, indent=2)}")
        
        elif action == "process":
            # 処理を実行
            def dummy_process(item, task_index=None):
                time.sleep(0.1)  # 処理をシミュレート
                return {'result': f"processed_{item['id']}"}
            
            results = process_with_queue(queue, dummy_process, test_dataset)
            print(f"Processed {len(results)} items")
        
        elif action == "cleanup":
            # クリーンアップ
            queue.cleanup_stale_tasks()
            print("Cleanup completed")
    
    else:
        print("Usage: python dynamic_task_queue.py [init|status|process|cleanup]")