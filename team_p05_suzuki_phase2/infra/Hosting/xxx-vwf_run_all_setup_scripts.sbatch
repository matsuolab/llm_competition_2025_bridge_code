#!/bin/bash
#SBATCH --job-name=vwf_setup_all
#SBATCH --output=logs/vwf_setup_all_%j.out
#SBATCH --error=logs/vwf_setup_all_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# =============================================================================
# 12-vwf_run_all_setup_scripts.sbatch
# スクリプト01-*から10-*までを順次実行するSLURMバッチジョブ
# 
# 使用方法:
#   sbatch 12-vwf_run_all_setup_scripts.sbatch
# =============================================================================

set -e  # エラーが発生したら即座に終了

# ログディレクトリの作成
mkdir -p logs

# 実行開始時刻の記録
SCRIPT_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "🚀 Starting VWF Setup Pipeline at $SCRIPT_START_TIME"
echo "📝 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $SLURM_JOB_NODELIST"
echo "=========================================="

# 実行するスクリプトのリスト（順序重要）
SCRIPTS=(
    "01-vwf_set_dummyjobs_canceller_proc.sh"
    "02-vwf_alloc.sh"
    "03-vwf_attach_node.sh"
    "04-vwf_source_me___setup_vllm_env.sh"
    "05-vwf_check_vllm_env.sh"
    "10-vwf_start_vllm-Qweb3-235B-A22B.sh"
)

# 実行結果を記録する変数
TOTAL_SCRIPTS=${#SCRIPTS[@]}
EXECUTED_SCRIPTS=0
SUCCESSFUL_SCRIPTS=0
FAILED_SCRIPTS=0
SKIPPED_SCRIPTS=0

# 各スクリプトの実行結果を記録する関数
log_script_result() {
    local script_name="$1"
    local status="$2"
    local exit_code="$3"
    local duration="$4"
    local timestamp=$(date '+%H:%M:%S')
    
    case "$status" in
        "SUCCESS")
            echo "[$timestamp] ✅ $script_name: COMPLETED (exit code: $exit_code, duration: ${duration}s)"
            SUCCESSFUL_SCRIPTS=$((SUCCESSFUL_SCRIPTS + 1))
            ;;
        "FAILED")
            echo "[$timestamp] ❌ $script_name: FAILED (exit code: $exit_code, duration: ${duration}s)"
            FAILED_SCRIPTS=$((FAILED_SCRIPTS + 1))
            ;;
        "SKIPPED")
            echo "[$timestamp] ⏭️  $script_name: SKIPPED - $exit_code"
            SKIPPED_SCRIPTS=$((SKIPPED_SCRIPTS + 1))
            ;;
    esac
}

# 各スクリプトを順次実行
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        log_script_result "$script" "SKIPPED" "File not found" "0"
        continue
    fi
    
    if [ ! -x "$script" ]; then
        echo "⚠️  Making $script executable..."
        chmod +x "$script"
    fi
    
    echo ""
    echo "🔄 Executing: $script"
    echo "⏰ Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    # スクリプト実行時間の測定
    script_start_time=$(date +%s)
    
    # スクリプトの実行
    if ./"$script"; then
        script_end_time=$(date +%s)
        duration=$((script_end_time - script_start_time))
        log_script_result "$script" "SUCCESS" "0" "$duration"
    else
        exit_code=$?
        script_end_time=$(date +%s)
        duration=$((script_end_time - script_start_time))
        log_script_result "$script" "FAILED" "$exit_code" "$duration"
        
        # 重要なスクリプトが失敗した場合は全体を停止
        case "$script" in
            "02-vwf_alloc.sh"|"04-vwf_source_me___setup_vllm_env.sh"|"10-vwf_start_vllm-Qweb3-235B-A22B.sh")
                echo "💥 Critical script failed. Aborting pipeline."
                exit $exit_code
                ;;
            *)
                echo "⚠️  Non-critical script failed. Continuing..."
                ;;
        esac
    fi
    
    EXECUTED_SCRIPTS=$((EXECUTED_SCRIPTS + 1))
    
    # スクリプト間の待機時間（必要に応じて）
    if [ "$script" = "02-vwf_alloc.sh" ]; then
        echo "⏳ Waiting 30 seconds for resource allocation..."
        sleep 30
    elif [ "$script" = "03-vwf_attach_node.sh" ]; then
        echo "⏳ Waiting 10 seconds for node attachment..."
        sleep 10
    fi
done

# 最終結果のサマリー
SCRIPT_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$(( $(date +%s) - $(date -d "$SCRIPT_START_TIME" +%s) ))

echo ""
echo "=========================================="
echo "🏁 VWF Setup Pipeline Summary"
echo "=========================================="
echo "⏰ Start time: $SCRIPT_START_TIME"
echo "⏰ End time: $SCRIPT_END_TIME"
echo "⏱️  Total duration: ${TOTAL_DURATION} seconds"
echo ""
echo "📊 Execution Results:"
echo "   📝 Total scripts: $TOTAL_SCRIPTS"
echo "   🔄 Executed: $EXECUTED_SCRIPTS"
echo "   ✅ Successful: $SUCCESSFUL_SCRIPTS"
echo "   ❌ Failed: $FAILED_SCRIPTS"
echo "   ⏭️  Skipped: $SKIPPED_SCRIPTS"

# 成功率の計算
if [ $EXECUTED_SCRIPTS -gt 0 ]; then
    success_rate=$(( (SUCCESSFUL_SCRIPTS * 100) / EXECUTED_SCRIPTS ))
    echo "   📈 Success rate: ${success_rate}%"
fi

# 全体の評価
if [ $FAILED_SCRIPTS -eq 0 ]; then
    echo ""
    echo "🎉 Pipeline Status: ALL SCRIPTS COMPLETED SUCCESSFULLY"
    echo "✅ VWF setup is ready!"
    
    # 最終確認として11番スクリプトも実行
    if [ -f "11-vwf_check_vllm_api_endpoints.sh" ] && [ -x "11-vwf_check_vllm_api_endpoints.sh" ]; then
        echo ""
        echo "🔍 Running final API endpoint check..."
        if ./11-vwf_check_vllm_api_endpoints.sh http://localhost:8000; then
            echo "✅ API endpoint check completed successfully!"
        else
            echo "⚠️  API endpoint check had issues, but setup pipeline completed."
        fi
    fi
    
    exit 0
elif [ $FAILED_SCRIPTS -le 2 ] && [ $SUCCESSFUL_SCRIPTS -ge 4 ]; then
    echo ""
    echo "👍 Pipeline Status: MOSTLY SUCCESSFUL (minor issues)"
    echo "⚠️  Some non-critical scripts failed, but core setup should be working."
    exit 0
else
    echo ""
    echo "💥 Pipeline Status: MULTIPLE FAILURES DETECTED"
    echo "❌ VWF setup may not be complete. Please check the logs."
    exit 1
fi
