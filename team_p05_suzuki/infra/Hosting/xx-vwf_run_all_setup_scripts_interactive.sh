#!/bin/bash

# =============================================================================
# 12-vwf_run_all_setup_scripts_interactive.sh
# スクリプト01-*から10-*までを順次実行するインタラクティブスクリプト
# 
# 使用方法:
#   ./12-vwf_run_all_setup_scripts_interactive.sh
#   ./12-vwf_run_all_setup_scripts_interactive.sh --auto  # 自動実行モード
# =============================================================================

# 設定
AUTO_MODE=false
PAUSE_BETWEEN_SCRIPTS=true
STOP_ON_FAILURE=true

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=true
            PAUSE_BETWEEN_SCRIPTS=false
            shift
            ;;
        --no-pause)
            PAUSE_BETWEEN_SCRIPTS=false
            shift
            ;;
        --continue-on-failure)
            STOP_ON_FAILURE=false
            shift
            ;;
        -h|--help)
            echo "VWF Setup Scripts Interactive Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto                 Run all scripts automatically without prompts"
            echo "  --no-pause            Don't pause between scripts"
            echo "  --continue-on-failure Continue even if critical scripts fail"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Scripts to be executed:"
            echo "  01-vwf_set_dummyjobs_canceller_proc.sh"
            echo "  02-vwf_alloc.sh"
            echo "  03-vwf_attach_node.sh"
            echo "  04-vwf_source_me___setup_vllm_env.sh"
            echo "  05-vwf_check_vllm_env.sh"
            echo "  10-vwf_start_vllm-Qweb3-235B-A22B.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# ログディレクトリの作成
mkdir -p logs

# 実行開始時刻の記録
SCRIPT_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "🚀 Starting VWF Setup Pipeline (Interactive Mode)"
echo "📝 Start time: $SCRIPT_START_TIME"
echo "🔧 Auto mode: $AUTO_MODE"
echo "⏸️  Pause between scripts: $PAUSE_BETWEEN_SCRIPTS"
echo "🛑 Stop on failure: $STOP_ON_FAILURE"
echo "=========================================="

# 実行するスクリプトのリスト
SCRIPTS=(
    "01-vwf_set_dummyjobs_canceller_proc.sh"
    "02-vwf_alloc.sh"
    "03-vwf_attach_node.sh"
    "04-vwf_source_me___setup_vllm_env.sh"
    "05-vwf_check_vllm_env.sh"
    "10-vwf_start_vllm-Qweb3-235B-A22B.sh"
)

# 重要なスクリプト（失敗時に停止対象）
CRITICAL_SCRIPTS=(
    "02-vwf_alloc.sh"
    "04-vwf_source_me___setup_vllm_env.sh"
    "10-vwf_start_vllm-Qweb3-235B-A22B.sh"
)

# 実行結果を記録する変数
TOTAL_SCRIPTS=${#SCRIPTS[@]}
EXECUTED_SCRIPTS=0
SUCCESSFUL_SCRIPTS=0
FAILED_SCRIPTS=0
SKIPPED_SCRIPTS=0

# ユーザー入力を待つ関数
wait_for_user() {
    if [ "$AUTO_MODE" = false ]; then
        echo ""
        read -p "Press Enter to continue, or 'q' to quit: " user_input
        if [ "$user_input" = "q" ] || [ "$user_input" = "Q" ]; then
            echo "👋 User requested to quit. Exiting..."
            exit 0
        fi
    fi
}

# スクリプト実行結果を記録する関数
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

# 重要なスクリプトかどうかを判定
is_critical_script() {
    local script="$1"
    for critical in "${CRITICAL_SCRIPTS[@]}"; do
        if [ "$script" = "$critical" ]; then
            return 0
        fi
    done
    return 1
}

# 各スクリプトを順次実行
for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "📋 Next script: $script"
    
    if [ ! -f "$script" ]; then
        echo "❌ File not found: $script"
        log_script_result "$script" "SKIPPED" "File not found" "0"
        continue
    fi
    
    if [ ! -x "$script" ]; then
        echo "🔧 Making $script executable..."
        chmod +x "$script"
    fi
    
    # スクリプトの説明を表示
    case "$script" in
        "01-vwf_set_dummyjobs_canceller_proc.sh")
            echo "📝 Description: Set up dummy jobs canceller process"
            ;;
        "02-vwf_alloc.sh")
            echo "📝 Description: Allocate compute resources"
            echo "⚠️  CRITICAL: This script allocates GPU resources"
            ;;
        "03-vwf_attach_node.sh")
            echo "📝 Description: Attach to allocated compute node"
            ;;
        "04-vwf_source_me___setup_vllm_env.sh")
            echo "📝 Description: Set up vLLM environment"
            echo "⚠️  CRITICAL: This script sets up the Python environment"
            ;;
        "05-vwf_check_vllm_env.sh")
            echo "📝 Description: Check vLLM environment setup"
            ;;
        "10-vwf_start_vllm-Qweb3-235B-A22B.sh")
            echo "📝 Description: Start vLLM server with Qwen3-235B-A22B model"
            echo "⚠️  CRITICAL: This script starts the main vLLM service"
            ;;
    esac
    
    wait_for_user
    
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
        
        # 成功後の待機時間
        case "$script" in
            "02-vwf_alloc.sh")
                echo "⏳ Waiting 30 seconds for resource allocation to complete..."
                sleep 30
                ;;
            "03-vwf_attach_node.sh")
                echo "⏳ Waiting 10 seconds for node attachment..."
                sleep 10
                ;;
        esac
        
    else
        exit_code=$?
        script_end_time=$(date +%s)
        duration=$((script_end_time - script_start_time))
        log_script_result "$script" "FAILED" "$exit_code" "$duration"
        
        # 重要なスクリプトが失敗した場合の処理
        if is_critical_script "$script"; then
            echo "💥 CRITICAL SCRIPT FAILED: $script"
            if [ "$STOP_ON_FAILURE" = true ]; then
                echo "🛑 Stopping pipeline due to critical failure."
                if [ "$AUTO_MODE" = false ]; then
                    read -p "Do you want to continue anyway? (y/N): " continue_choice
                    if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
                        exit $exit_code
                    fi
                else
                    exit $exit_code
                fi
            fi
        else
            echo "⚠️  Non-critical script failed. Continuing..."
        fi
    fi
    
    EXECUTED_SCRIPTS=$((EXECUTED_SCRIPTS + 1))
    
    # スクリプト間の一時停止
    if [ "$PAUSE_BETWEEN_SCRIPTS" = true ] && [ "$AUTO_MODE" = false ]; then
        echo ""
        echo "✅ Script completed. Ready for next script..."
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
    
    # 最終確認の提案
    if [ "$AUTO_MODE" = false ]; then
        echo ""
        read -p "Would you like to run the API endpoint check? (Y/n): " api_check
        if [ "$api_check" != "n" ] && [ "$api_check" != "N" ]; then
            if [ -f "11-vwf_check_vllm_api_endpoints.sh" ] && [ -x "11-vwf_check_vllm_api_endpoints.sh" ]; then
                echo "🔍 Running API endpoint check..."
                ./11-vwf_check_vllm_api_endpoints.sh http://localhost:8000
            else
                echo "⚠️  API check script not found or not executable."
            fi
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
    echo "❌ VWF setup may not be complete. Please check the individual script outputs."
    exit 1
fi
