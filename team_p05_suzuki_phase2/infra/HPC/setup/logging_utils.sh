#!/bin/bash

# --- 色設定 ---
readonly C_RESET='\033[0m'
readonly C_RED='\033[0;31m'
readonly C_GREEN='\033[0;32m'
readonly C_YELLOW='\033[0;33m'
readonly C_BLUE='\033[0;34m'
readonly C_CYAN='\033[0;36m'
readonly C_BOLD='\033[1m'

# --- ログ出力関数 ---
_log_with_timestamp() {
    local log_level=$1
    local color_code=$2
    local message=$3
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    # エラー以外はstdout、エラーはstderrに出力
    if [ "$log_level" == "ERROR" ]; then
        echo -e "${color_code}[${timestamp}] ${log_level}: ${message}${C_RESET}" >&2
    else
        echo -e "${color_code}[${timestamp}] ${log_level}: ${message}${C_RESET}"
    fi
}

log_info()    { _log_with_timestamp "INFO"    "${C_BLUE}"   "$1"; }
log_success() { _log_with_timestamp "SUCCESS" "${C_GREEN}"  "$1"; }
log_warn()    { _log_with_timestamp "WARNING" "${C_YELLOW}" "$1"; }
log_error()   { _log_with_timestamp "ERROR"   "${C_RED}"    "$1"; }

print_header() {
    echo -e "\n${C_CYAN}${C_BOLD}=======================================================================${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD} $1 ${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD}=======================================================================${C_RESET}"
}

# --- タイマー機能（ネスト対応） ---
export __TIMER_STACK__=()

start_timer() {
    __TIMER_STACK__+=($(date +%s))
    if [ -n "$1" ]; then
        # メッセージがある場合はINFOログとして出力
        log_info "$1"
    fi
}

end_timer() {
    local end_time
    end_time=$(date +%s)
    if [ ${#__TIMER_STACK__[@]} -gt 0 ]; then
        # スタックの最後の要素（最新の開始時間）を取得
        local start_time=${__TIMER_STACK__[-1]}

        # ★★★ 修正箇所 ★★★
        # スタックの最後の要素を削除する
        # BASH_ARGC is available in bash 4.3+
        if ((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
             unset '__TIMER_STACK__[-1]'
        else # Fallback for older bash versions
             __TIMER_STACK__=("${__TIMER_STACK__[@]:0:(${#__TIMER_STACK__[@]} - 1)}")
        fi

        local elapsed=$((end_time - start_time))
        local mins=$((elapsed / 60))
        local secs=$((elapsed % 60))
        log_success "処理が完了しました (経過時間: ${mins}分${secs}秒)"
    else
        log_warn "タイマースタックが空です。end_timerが過剰に呼び出された可能性があります。"
    fi
}

# --- エラーハンドリング機能 ---
handle_error() {
    local exit_code=$1
    local line_no=$2
    local failed_command=$3
    log_error "コマンドが失敗しました (終了コード: ${exit_code}, 行番号: ${line_no})"
    log_error "失敗したコマンド: ${failed_command}"
}