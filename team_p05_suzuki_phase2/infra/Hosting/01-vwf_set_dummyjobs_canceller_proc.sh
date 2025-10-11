#!/bin/bash

# =============================================================================
# 01-vwf_set_dummyjobs_canceller_proc.sh
# バックグラウンドでダミージョブをキャンセルするプロセスを設定するスクリプト
# =============================================================================

# 設定
NODES=("osk-gpu63" "osk-gpu64" "osk-gpu65")
INTERVAL=20  # 実行間隔（秒）
DURATION=60  # 実行時間（秒）

# ログディレクトリの作成
mkdir -p ./logs
LOG_FILE="./logs/dummy_job_canceller.log"

# clr63相当機能の関数（auto_cancel.shの内容をベース）
clear_node_jobs() {
    local node=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Clearing jobs on node: $node" | tee -a "$LOG_FILE"
    
    # 設定（auto_cancel.shから）
    local USERNAME="kan.hatakeyama"
    local SCANCEL_SCRIPT="$HOME/../shareP05/scripts/scancel.sh"
    
    # scancel.shの存在確認
    if [ ! -x "$SCANCEL_SCRIPT" ]; then
        echo "[$timestamp] エラー: $SCANCEL_SCRIPT が存在しないか実行可能ではありません" | tee -a "$LOG_FILE"
        return 1
    fi
    
    echo "[$timestamp] ノード $node のジョブを確認中..." | tee -a "$LOG_FILE"
    
    # ジョブIDを取得（エラーハンドリング付き）
    local job_ids=$(squeue --nodelist="$node" --user="$USERNAME" --noheader --format="%i" 2>/dev/null || true)
    
    if [ -z "$job_ids" ]; then
        echo "[$timestamp]   $node にジョブが見つかりません" | tee -a "$LOG_FILE"
        return 0
    fi
    
    # ジョブIDごとにキャンセル実行
    local job_count=0
    while IFS= read -r job_id; do
        if [ -n "$job_id" ]; then
            echo "[$timestamp]   ジョブID $job_id をキャンセル中..." | tee -a "$LOG_FILE"
            bash "$SCANCEL_SCRIPT" "$job_id" || true
            ((job_count++))
        fi
    done <<< "$job_ids"
    
    echo "[$timestamp]   $node で ${job_count} 個のジョブをキャンセルしました" | tee -a "$LOG_FILE"
    echo "[$timestamp] Completed clearing jobs on node: $node" | tee -a "$LOG_FILE"
}

# メイン処理関数
run_job_cleaner() {
    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION))
    
    echo "🚀 Starting background dummy job canceller..." | tee -a "$LOG_FILE"
    echo "📝 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "⏰ Duration: ${DURATION} seconds, Interval: ${INTERVAL} seconds" | tee -a "$LOG_FILE"
    echo "🖥️  Target nodes: ${NODES[*]}" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    while [ $(date +%s) -lt $end_time ]; do
        for node in "${NODES[@]}"; do
            clear_node_jobs "$node" &
        done
        
        # 全ノードの処理完了を待つ
        wait
        
        # 次の実行まで待機（残り時間が間隔より短い場合は終了）
        local current_time=$(date +%s)
        local remaining_time=$((end_time - current_time))
        
        if [ $remaining_time -gt $INTERVAL ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting ${INTERVAL} seconds until next execution..." | tee -a "$LOG_FILE"
            sleep $INTERVAL
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining time (${remaining_time}s) is less than interval. Finishing..." | tee -a "$LOG_FILE"
            break
        fi
    done
    
    echo "✅ Dummy job canceller completed after ${DURATION} seconds" | tee -a "$LOG_FILE"
}

# バックグラウンド実行の制御
case "${1:-start}" in
    "start")
        echo "Starting dummy job canceller in background..."
        # 現在のスクリプト自体をバックグラウンドで実行
        nohup "$0" foreground > /dev/null 2>&1 &
        BG_PID=$!
        echo "Background dummy job canceller started with PID: $BG_PID"
        echo "Log file: $LOG_FILE"
        echo "To stop: kill $BG_PID"
        echo $BG_PID > ./logs/dummy_job_canceller.pid
        ;;
    "stop")
        if [ -f ./logs/dummy_job_canceller.pid ]; then
            PID=$(cat ./logs/dummy_job_canceller.pid)
            if kill -0 $PID 2>/dev/null; then
                kill $PID
                echo "Job cleaner (PID: $PID) stopped"
                rm -f ./logs/dummy_job_canceller.pid
            else
                echo "Job cleaner process not found"
                rm -f ./logs/dummy_job_canceller.pid
            fi
        else
            echo "No PID file found"
        fi
        ;;
    "status")
        if [ -f ./logs/dummy_job_canceller.pid ]; then
            PID=$(cat ./logs/dummy_job_canceller.pid)
            if kill -0 $PID 2>/dev/null; then
                echo "Job cleaner is running (PID: $PID)"
            else
                echo "Job cleaner is not running"
                rm -f ./logs/dummy_job_canceller.pid
            fi
        else
            echo "Job cleaner is not running"
        fi
        ;;
    "foreground")
        echo "Running dummy job canceller in foreground..."
        run_job_cleaner
        ;;
    *)
        echo "Usage: $0 {start|stop|status|foreground}"
        echo "  start      - Start dummy job canceller in background (default)"
        echo "  stop       - Stop background dummy job canceller"
        echo "  status     - Check if dummy job canceller is running"
        echo "  foreground - Run dummy job canceller in foreground"
        exit 1
        ;;
esac
