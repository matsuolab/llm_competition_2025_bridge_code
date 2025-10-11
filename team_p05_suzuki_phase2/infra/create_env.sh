#!/bin/bash -l
#
# Conda環境を生成し、環境変数を設定するスクリプト。
# ロギング等のユーティリティ関数を内包しています。
#
# [使い方]
# source ./create_env.sh <env_name> [python_version]  # 推奨
# bash ./create_env.sh <env_name> [python_version]    # 直接実行も可能
#
# [例]
# source ./create_env.sh my_env 3.10
# bash ./create_env.sh my_env 3.10
#

#======================================================================
# 実行モード判定とエラーハンドリング設定
#======================================================================

# スクリプトがsourceされているかどうかを判定
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 直接実行モード
    SCRIPT_SOURCED=false
    EXIT_FUNC="exit"
else
    # sourceモード
    SCRIPT_SOURCED=true
    EXIT_FUNC="return"
fi

# エラーハンドリングの設定
safe_exit() {
    local exit_code=${1:-1}
    if [ "$SCRIPT_SOURCED" = true ]; then
        return $exit_code
    else
        exit $exit_code
    fi
}

#======================================================================
# ユーティリティ関数 
#======================================================================

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
if [ "$SCRIPT_SOURCED" = true ]; then
    export __TIMER_STACK__=()
else
    declare -a __TIMER_STACK__=()
fi

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

        # スタックの最後の要素を削除する
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
    safe_exit $exit_code
}

# エラートラップの設定
set -e
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR

#======================================================================
# メイン処理 
#======================================================================

# --- 実行モードの表示 ---
if [ "$SCRIPT_SOURCED" = true ]; then
    log_info "スクリプトはsourceモードで実行されています"
else
    log_info "スクリプトは直接実行モードで実行されています"
    log_warn "環境をアクティベートするには 'source ./create_env.sh' での実行を推奨します"
fi

# --- 設定変数 ---
ENVNAME=${1}
if [ -z "$ENVNAME" ]; then
    log_error "環境名が指定されていません。"
    log_error "使用方法: source ./create_env.sh <env_name> [python_version]"
    log_error "例: source ./create_env.sh my_env 3.10"
    safe_exit 1
fi

# Pythonバージョンの引数化
PYTHON_VERSION=${2:-"3.12"} # 第2引数がなければ "3.12" をデフォルト値とする
log_info "環境名: ${ENVNAME}, Pythonバージョン: ${PYTHON_VERSION}"

SYSTEM_LD_LIB_PATHS="/usr/lib64:/usr/lib"

# Condaのインストールパスチェック
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
if ! conda info --base &>/dev/null; then
    log_error "Condaが見つかりません。Condaがインストールされ、PATHが通っていることを確認してください。"
    safe_exit 1
fi

# 環境作成パスをホームディレクトリ直下にする
export CONDA_ENV_FULL_PATH="$HOME/envs/$ENVNAME"

print_header "Conda環境 '$ENVNAME' のセットアップ"

if [ ! -d "$CONDA_ENV_FULL_PATH" ]; then
    start_timer "Conda環境 '$ENVNAME' (Python ${PYTHON_VERSION}) を新規作成します..."
    # --prefix オプションで絶対パスを指定して環境を作成
    if ! conda create --prefix "$CONDA_ENV_FULL_PATH" python="$PYTHON_VERSION" -y --quiet; then
        log_error "Conda環境 '$ENVNAME' の作成に失敗しました。"
        safe_exit 1
    fi
    end_timer

    log_info "アクティベート時に読み込む環境変数スクリプトを作成します..."
    ACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/activate.d"
    ACTIVATE_SCRIPT="$ACTIVATE_DIR/env_vars.sh"
    mkdir -p "$ACTIVATE_DIR"

    # ライブラリパスの責務を分離
    # torch/lib のような特定ライブラリへのパスを削除し、Conda環境自身のパスのみを追加
    LD_LIB_PATH_TO_ADD="${SYSTEM_LD_LIB_PATHS}:${CONDA_ENV_FULL_PATH}/lib"

    cat <<EOF > "$ACTIVATE_SCRIPT"
#!/bin/bash
export ORIGINAL_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${LD_LIB_PATH_TO_ADD}:\$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_ENV_FULL_PATH/"
EOF
    chmod +x "$ACTIVATE_SCRIPT"
    log_info "作成完了: $ACTIVATE_SCRIPT"

    # --- deactivate.d スクリプト (CUDA_HOMEのunsetのみに変更) ---
    log_info "ディアクティベート時に読み込む環境変数スクリプトを作成します..."
    DEACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/deactivate.d"
    DEACTIVATE_SCRIPT="$DEACTIVATE_DIR/env_vars_rollback.sh"
    mkdir -p "$DEACTIVATE_DIR"

    cat <<EOF > "$DEACTIVATE_SCRIPT"
#!/bin/bash
export LD_LIBRARY_PATH="\$ORIGINAL_LD_LIBRARY_PATH"
unset CUDA_HOME
unset ORIGINAL_LD_LIBRARY_PATH
EOF
    chmod +x "$DEACTIVATE_SCRIPT"
    log_info "作成完了: $DEACTIVATE_SCRIPT"

else
    log_warn "Conda環境 '$ENVNAME' は既に存在するため、新規作成をスキップします。"
fi

log_success "Conda環境のセットアップスクリプトが完了しました。"

# --- 実行モードに応じた後処理 ---
if [ "$SCRIPT_SOURCED" = true ]; then
    log_info "sourceモードで実行されているため、環境をアクティベートできます。"
    log_info "次のコマンドで環境をアクティベートしてください: conda activate $CONDA_ENV_FULL_PATH"
    
    # sourceモードの場合、オプションで自動アクティベートを提供
    read -p "環境を今すぐアクティベートしますか？ [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "環境をアクティベートしています..."
        conda activate "$CONDA_ENV_FULL_PATH"
        log_success "環境 '$ENVNAME' がアクティベートされました。"
    fi
else
    log_info "直接実行モードで実行されました。"
    log_info "環境をアクティベートするには、以下のコマンドを実行してください:"
    log_info "  conda activate $CONDA_ENV_FULL_PATH"
    log_info "または、sourceモードで再実行してください:"
    log_info "  source ./create_env.sh $ENVNAME $PYTHON_VERSION"
fi