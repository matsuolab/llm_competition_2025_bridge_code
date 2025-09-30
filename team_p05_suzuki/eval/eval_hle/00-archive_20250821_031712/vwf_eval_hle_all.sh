#!/bin/bash
# HLE評価実行Wrapperスクリプト
# 作成日: 2025-08-15
# 目的: predict → judge の完全自動実行

set -e

# カラー出力設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
HLE評価実行Wrapperスクリプト

使用方法:
    $0 [オプション]

オプション:
    -c, --config CONFIG_NAME    使用する設定ファイル名 (デフォルト: config_final)
    -t, --test                  テストモード (5問のみ実行)
    -r, --resume                継続実行モード
    -s, --skip-predict          推論をスキップしてjudgeのみ実行
    -j, --skip-judge            judgeをスキップして推論のみ実行
    -b, --background            バックグラウンド実行
    -h, --help                  このヘルプを表示

例:
    $0                          # 120問フル実行
    $0 --test                   # 5問テスト実行
    $0 --resume                 # 継続実行
    $0 --config config_test20   # 20問実行
    $0 --skip-predict           # judgeのみ実行

推奨実行順序:
    1. テスト実行: $0 --test
    2. フル実行:   $0
    3. 継続実行:   $0 --resume (中断時)
EOF
}

# vLLMサーバー確認
check_vllm_server() {
    if curl -s --connect-timeout 5 http://localhost:8000/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 実行前チェック
pre_check() {
    log_info "実行前チェックを開始します..."
    
    # 必要ファイルの確認
    local required_files=(
        "predict_improved.py"
        "judge_improved.py"
        "conf/${CONFIG_NAME}.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必要ファイルが見つかりません: $file"
            return 1
        fi
    done
    
    # vLLMサーバー確認
    if ! check_vllm_server; then
        log_error "vLLMサーバーが応答しません (http://localhost:8000/health)"
        log_info "vLLMサーバーを起動してから再実行してください"
        return 1
    fi
    
    # ディレクトリ作成
    mkdir -p predictions judged
    
    log_success "事前チェック完了"
    return 0
}

# 推論実行
run_predict() {
    log_info "=== Step 1: 推論実行 ==="
    log_info "設定ファイル: conf/${CONFIG_NAME}.yaml"
    
    local start_time=$(date +%s)
    
    if [ "$BACKGROUND" = "true" ]; then
        local log_file="predict_$(date +%Y%m%d_%H%M%S).log"
        log_info "バックグラウンド実行開始: $log_file"
        nohup python predict_improved.py --config-name="$CONFIG_NAME" > "$log_file" 2>&1 &
        local pid=$!
        log_success "推論をバックグラウンドで開始しました (PID: $pid)"
        log_info "進捗確認: tail -f $log_file"
        return 0
    else
        if python predict_improved.py --config-name="$CONFIG_NAME"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "推論が完了しました (実行時間: ${duration}秒)"
            return 0
        else
            log_error "推論実行に失敗しました"
            return 1
        fi
    fi
}

# 評価実行
run_judge() {
    log_info "=== Step 2: 評価実行 ==="
    
    # 予測結果ファイルの確認
    local latest_link="predictions/predict-latest.json"
    if [ ! -L "$latest_link" ]; then
        log_error "予測結果ファイルが見つかりません: $latest_link"
        log_info "先にpredict_improved.pyを実行してください"
        return 1
    fi
    
    log_info "予測結果: $latest_link -> $(readlink $latest_link)"
    
    local start_time=$(date +%s)
    
    if python judge_improved.py --config-name="$CONFIG_NAME"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "評価が完了しました (実行時間: ${duration}秒)"
        return 0
    else
        log_error "評価実行に失敗しました"
        return 1
    fi
}

# 結果表示
show_results() {
    log_info "=== Step 3: 結果確認 ==="
    
    echo
    log_info "📁 予測結果ファイル:"
    ls -la predictions/ | tail -5
    
    echo
    log_info "📁 評価結果ファイル:"
    ls -la judged/ | tail -5
    
    echo
    log_info "📊 最新の評価サマリー:"
    local latest_leaderboard=$(ls -t /home/Competition2025/P05/shareP05/eval/output/eval_hle/leaderboard/ 2>/dev/null | head -1)
    if [ -n "$latest_leaderboard" ]; then
        local summary_file="/home/Competition2025/P05/shareP05/eval/output/eval_hle/leaderboard/$latest_leaderboard/summary.json"
        if [ -f "$summary_file" ]; then
            log_success "リーダーボード: $latest_leaderboard"
            if command -v jq > /dev/null; then
                cat "$summary_file" | jq .
            else
                cat "$summary_file"
            fi
        else
            log_warning "サマリーファイルが見つかりません"
        fi
    else
        log_warning "リーダーボード結果が見つかりません"
    fi
}

# メイン処理
main() {
    local CONFIG_NAME="config_final"
    local TEST_MODE=false
    local RESUME_MODE=false
    local SKIP_PREDICT=false
    local SKIP_JUDGE=false
    local BACKGROUND=false
    
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_NAME="$2"
                shift 2
                ;;
            -t|--test)
                CONFIG_NAME="config_test5"
                TEST_MODE=true
                shift
                ;;
            -r|--resume)
                CONFIG_NAME="config_test5_resume"
                RESUME_MODE=true
                shift
                ;;
            -s|--skip-predict)
                SKIP_PREDICT=true
                shift
                ;;
            -j|--skip-judge)
                SKIP_JUDGE=true
                shift
                ;;
            -b|--background)
                BACKGROUND=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "不明なオプション: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # ヘッダー表示
    echo "=== HLE評価実行Wrapperスクリプト ==="
    echo "日時: $(date)"
    echo "設定: $CONFIG_NAME"
    if [ "$TEST_MODE" = true ]; then
        echo "モード: テスト (5問)"
    elif [ "$RESUME_MODE" = true ]; then
        echo "モード: 継続実行"
    else
        echo "モード: フル実行 (120問)"
    fi
    echo
    
    # 実行前チェック
    if ! pre_check; then
        exit 1
    fi
    
    local total_start_time=$(date +%s)
    
    # 推論実行
    if [ "$SKIP_PREDICT" != true ]; then
        if ! run_predict; then
            exit 1
        fi
        
        if [ "$BACKGROUND" = true ]; then
            log_info "バックグラウンド実行中です。完了後に以下を実行してください:"
            log_info "$0 --skip-predict --config $CONFIG_NAME"
            exit 0
        fi
    fi
    
    # 評価実行
    if [ "$SKIP_JUDGE" != true ]; then
        if ! run_judge; then
            exit 1
        fi
    fi
    
    # 結果表示
    show_results
    
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    echo
    log_success "=== 全処理完了 ==="
    log_info "総実行時間: ${total_duration}秒 ($(($total_duration / 60))分)"
    
    if [ "$TEST_MODE" = true ]; then
        echo
        log_info "🚀 フル実行コマンド:"
        echo "    $0"
        echo
        log_info "🔄 継続実行コマンド (中断時):"
        echo "    $0 --resume"
    fi
}

# スクリプト実行
main "$@"
