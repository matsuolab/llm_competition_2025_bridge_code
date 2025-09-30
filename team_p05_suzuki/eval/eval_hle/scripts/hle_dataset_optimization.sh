#!/bin/bash
# HLE Dataset Loading Optimization Script (Improved Version)
# 作成日: 2025-08-15
# 更新日: 2025-08-15 (Option B改善版)
# 目的: HLEデータセット読み込み速度の最適化とトラブル予防

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

log_debug() {
    if [ "${DEBUG:-}" = "true" ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# ヘルプ表示
show_help() {
    cat << EOF
HLE Dataset Loading Optimization Script (Improved Version)

使用方法:
    $0 [オプション]

オプション:
    -c, --cleanup       ロックファイルとキャッシュのクリーンアップ
    -o, --optimize      環境変数の最適化設定（インテリジェント）
    -r, --run           最適化後にpredict.pyを実行
    -s, --status        現在の状況確認
    -d, --debug         デバッグモード有効
    -h, --help          このヘルプを表示

例:
    $0 --cleanup --optimize --run    # 全ての最適化を実行してpredict.py開始
    $0 --status                      # 現在の状況のみ確認
    $0 --debug --optimize            # デバッグモードで最適化

改善点 (Option B):
    ✅ 既存キャッシュの自動検出と活用
    ✅ インテリジェントなオンライン/オフライン判定
    ✅ gated dataset対応
    ✅ エラー時の自動フォールバック
    ✅ データセット固有の最適化
EOF
}

# 設定ファイルからデータセット名を取得
get_dataset_name() {
    if [ -f "conf/config.yaml" ]; then
        grep "^dataset:" conf/config.yaml | cut -d' ' -f2 | tr -d '"' || echo "cais/hle"
    else
        echo "cais/hle"
    fi
}

# vLLMサーバーの確認
check_vllm_server() {
    local base_url="${1:-http://localhost:8000}"
    
    if curl -s --connect-timeout 5 "$base_url/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 既存キャッシュの確認
check_existing_cache() {
    local dataset_name="$1"
    
    log_debug "データセット名: $dataset_name"
    
    local cache_dirs=(
        "$HOME/.cache/huggingface/datasets"
        "/tmp/hf_datasets"
        "${HF_DATASETS_CACHE:-}"
    )
    
    for cache_dir in "${cache_dirs[@]}"; do
        if [ -z "$cache_dir" ]; then
            continue
        fi
        
        log_debug "キャッシュディレクトリを確認中: $cache_dir"
        
        if [ -d "$cache_dir" ]; then
            # データセット名に応じた柔軟なパターンマッチング
            local found_dirs=""
            case "$dataset_name" in
                "cais/hle")
                    found_dirs=$(find "$cache_dir" -type d -name "*cais*hle*" 2>/dev/null | head -1)
                    ;;
                "team-suzuki/hle-extract")
                    found_dirs=$(find "$cache_dir" -type d -name "*team-suzuki*hle-extract*" 2>/dev/null | head -1)
                    ;;
                *)
                    # 汎用的なパターンマッチング
                    local safe_name=$(echo "$dataset_name" | sed 's/[\/\-]/_/g')
                    found_dirs=$(find "$cache_dir" -type d -name "*${safe_name}*" 2>/dev/null | head -1)
                    ;;
            esac
            
            if [ -n "$found_dirs" ]; then
                local cache_size=$(du -sh "$found_dirs" 2>/dev/null | cut -f1 || echo "不明")
                log_debug "キャッシュが見つかりました: $found_dirs (サイズ: $cache_size)"
                echo "$cache_dir"
                return 0
            fi
        fi
    done
    
    log_debug "既存キャッシュが見つかりませんでした"
    return 1
}

# オンライン/オフラインモードの自動判定
determine_offline_mode() {
    local dataset_name="$1"
    
    # 既存キャッシュがある場合はオフラインモード
    if check_existing_cache "$dataset_name" > /dev/null; then
        export HF_DATASETS_OFFLINE=1
        log_success "既存キャッシュが見つかりました。オフラインモードを使用します。"
        return 0
    fi
    
    # HF_TOKENがある場合はオンラインモードを試す
    if [ -n "${HF_TOKEN:-}" ]; then
        export HF_DATASETS_OFFLINE=0
        log_info "HF_TOKENが設定されています。オンラインモードを試します。"
        log_warning "gated datasetの場合、アクセス権限が必要です。"
        return 0
    fi
    
    # デフォルトはオフラインモード
    export HF_DATASETS_OFFLINE=1
    log_warning "HF_TOKENが未設定です。オフラインモードを使用します。"
    log_info "オンラインアクセスが必要な場合は、HF_TOKENを設定してください。"
}

# 安全なキャッシュパス選択
optimize_cache_path() {
    local dataset_name="$1"
    local existing_cache=$(check_existing_cache "$dataset_name")
    
    if [ -n "$existing_cache" ]; then
        # 既存キャッシュがある場合はそのまま使用
        log_success "既存キャッシュを使用: $existing_cache"
        local cache_size=$(du -sh "$existing_cache" 2>/dev/null | cut -f1 || echo "不明")
        log_info "キャッシュサイズ: $cache_size"
        
        # 既存キャッシュのパスを明示的に設定（必要に応じて）
        if [ "$existing_cache" != "$HOME/.cache/huggingface/datasets" ]; then
            export HF_DATASETS_CACHE="$existing_cache"
            log_info "HF_DATASETS_CACHEを設定: $existing_cache"
        fi
        return 0
    fi
    
    # 新規の場合のみ高速ストレージを検討
    log_info "既存キャッシュが見つかりません。新規キャッシュ設定を検討中..."
    
    if [ -d "/tmp" ] && [ -w "/tmp" ]; then
        export HF_DATASETS_CACHE="/tmp/hf_datasets"
        export HF_HOME="/tmp/hf_cache"
        mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" 2>/dev/null || true
        log_info "高速ストレージ(/tmp)を新規キャッシュに設定しました"
        log_warning "注意: /tmpは再起動時に削除されます"
    else
        log_info "デフォルトキャッシュパスを使用します"
    fi
}

# 現在の状況確認
check_status() {
    log_info "現在の状況を確認中..."
    
    # データセット名取得
    local dataset_name=$(get_dataset_name)
    log_info "対象データセット: $dataset_name"
    
    # プロセス確認
    if pgrep -f "python predict.py" > /dev/null; then
        local pid=$(pgrep -f "python predict.py")
        log_warning "predict.pyが既に実行中です (PID: $pid)"
        ps -p $pid -o pid,etime,pcpu,pmem,cmd 2>/dev/null || true
    else
        log_info "predict.pyは実行されていません"
    fi
    
    # ロックファイル確認
    local lock_files=$(find ~/.cache/huggingface/datasets/ -name "*.lock" 2>/dev/null | wc -l)
    if [ $lock_files -gt 0 ]; then
        log_warning "ロックファイルが ${lock_files} 個見つかりました:"
        find ~/.cache/huggingface/datasets/ -name "*.lock" 2>/dev/null | head -5
        if [ $lock_files -gt 5 ]; then
            log_info "... (他 $((lock_files - 5)) 個)"
        fi
    else
        log_success "ロックファイルは見つかりませんでした"
    fi
    
    # キャッシュ状況確認
    local existing_cache=$(check_existing_cache "$dataset_name")
    if [ -n "$existing_cache" ]; then
        # 実際のキャッシュディレクトリを取得
        local actual_cache_dir=$(find "$existing_cache" -type d -name "*cais*hle*" 2>/dev/null | head -1)
        local cache_size=$(du -sh "$actual_cache_dir" 2>/dev/null | cut -f1 || echo "不明")
        log_success "データセットキャッシュが見つかりました:"
        log_info "  パス: $actual_cache_dir"
        log_info "  サイズ: $cache_size"
    else
        log_warning "データセットキャッシュが見つかりません"
    fi
    
    # 環境変数確認
    log_info "現在の環境変数:"
    echo "  HF_DATASETS_OFFLINE: ${HF_DATASETS_OFFLINE:-未設定}"
    echo "  HF_HUB_DISABLE_PROGRESS_BARS: ${HF_HUB_DISABLE_PROGRESS_BARS:-未設定}"
    echo "  TOKENIZERS_PARALLELISM: ${TOKENIZERS_PARALLELISM:-未設定}"
    echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-デフォルト}"
    echo "  HF_TOKEN: ${HF_TOKEN:+設定済み}"
    
    # vLLMサーバー確認
    if check_vllm_server; then
        log_success "vLLMサーバーは正常に応答しています"
    else
        log_error "vLLMサーバーが応答しません (http://localhost:8000/health)"
    fi
}

# クリーンアップ実行
cleanup() {
    log_info "クリーンアップを開始します..."
    
    # 実行中のpredict.pyプロセス確認
    if pgrep -f "python predict.py" > /dev/null; then
        log_error "predict.pyが実行中です。先に停止してください。"
        return 1
    fi
    
    # ロックファイル削除
    local lock_files=$(find ~/.cache/huggingface/datasets/ -name "*.lock" 2>/dev/null)
    if [ -n "$lock_files" ]; then
        log_info "ロックファイルを削除中..."
        local count=$(echo "$lock_files" | wc -l)
        find ~/.cache/huggingface/datasets/ -name "*.lock" -delete 2>/dev/null
        log_success "ロックファイル ${count} 個を削除しました"
    else
        log_info "削除するロックファイルはありません"
    fi
    
    # 古いキャッシュファイル削除（7日以上前）
    log_info "古いキャッシュファイルをチェック中..."
    local old_files=$(find ~/.cache/huggingface/ -type f -mtime +7 -name "*.lock" 2>/dev/null | wc -l)
    if [ $old_files -gt 0 ]; then
        find ~/.cache/huggingface/ -type f -mtime +7 -name "*.lock" -delete 2>/dev/null
        log_success "古いロックファイル ${old_files} 個を削除しました"
    fi
    
    # /tmpの古いキャッシュ削除（1日以上前）
    if [ -d "/tmp/hf_datasets" ]; then
        log_info "/tmpの古いキャッシュをチェック中..."
        find /tmp/hf_datasets -type f -mtime +1 -delete 2>/dev/null || true
        find /tmp/hf_cache -type f -mtime +1 -delete 2>/dev/null || true
    fi
}

# インテリジェントな環境変数最適化
optimize_env() {
    log_info "環境変数を最適化中..."
    
    # データセット名取得
    local dataset_name=$(get_dataset_name)
    log_info "対象データセット: $dataset_name"
    
    # オンライン/オフラインモード判定
    determine_offline_mode "$dataset_name"
    
    # キャッシュパス最適化
    optimize_cache_path "$dataset_name"
    
    # 基本最適化設定
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export TOKENIZERS_PARALLELISM=true
    
    # 追加最適化設定
    export HF_HUB_DOWNLOAD_TIMEOUT=300
    export TRANSFORMERS_VERBOSITY=error
    
    log_success "環境変数を最適化しました:"
    echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "  HF_HUB_DISABLE_PROGRESS_BARS=$HF_HUB_DISABLE_PROGRESS_BARS"
    echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
    echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-デフォルト}"
    echo "  HF_HOME=${HF_HOME:-デフォルト}"
}

# 安全なpredict.py実行
safe_predict_run() {
    log_info "predict_improved.pyを安全に実行します..."
    
    # 事前チェック
    if [ ! -f "predict_improved.py" ]; then
        log_error "predict_improved.pyが見つかりません"
        return 1
    fi
    
    if [ ! -f "conf/config.yaml" ]; then
        log_error "conf/config.yamlが見つかりません"
        return 1
    fi
    
    if ! check_vllm_server; then
        log_error "vLLMサーバーが応答しません (http://localhost:8000/health)"
        return 1
    fi
    
    log_success "事前チェック完了"
    
    # 環境変数のバックアップ
    local backup_offline="${HF_DATASETS_OFFLINE:-}"
    local backup_cache="${HF_DATASETS_CACHE:-}"
    
    log_info "predict_improved.pyを開始します..."
    
    # バックグラウンド実行オプション
    if [ "$BACKGROUND" = "true" ]; then
        local log_file="predict_$(date +%Y%m%d_%H%M%S).log"
        nohup python predict_improved.py > "$log_file" 2>&1 &
        local pid=$!
        log_success "predict_improved.pyをバックグラウンドで開始しました (PID: $pid)"
        log_info "ログファイル: $log_file"
        log_info "進捗確認: tail -f $log_file"
        return 0
    fi
    
    # フォアグラウンド実行
    if python predict_improved.py; then
        log_success "predict_improved.pyが正常に完了しました"
        return 0
    else
        local exit_code=$?
        log_error "predict_improved.py実行に失敗しました (終了コード: $exit_code)"
        
        # フォールバック: オフラインモードで再試行
        if [ "$HF_DATASETS_OFFLINE" != "1" ]; then
            log_info "フォールバック: オフラインモードで再試行します..."
            export HF_DATASETS_OFFLINE=1
            unset HF_DATASETS_CACHE
            unset HF_HOME
            
            if python predict_improved.py; then
                log_success "オフラインモードでの再試行が成功しました"
                return 0
            else
                log_error "オフラインモードでの再試行も失敗しました"
                return 1
            fi
        fi
        
        return $exit_code
    fi
}

# メイン処理
main() {
    local cleanup_flag=false
    local optimize_flag=false
    local run_flag=false
    local status_flag=false
    
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cleanup)
                cleanup_flag=true
                shift
                ;;
            -o|--optimize)
                optimize_flag=true
                shift
                ;;
            -r|--run)
                run_flag=true
                shift
                ;;
            -s|--status)
                status_flag=true
                shift
                ;;
            -b|--background)
                BACKGROUND=true
                shift
                ;;
            -d|--debug)
                DEBUG=true
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
    
    # デバッグモード表示
    if [ "${DEBUG:-}" = "true" ]; then
        log_debug "デバッグモードが有効です"
    fi
    
    # オプションが指定されていない場合はヘルプを表示
    if [ "$cleanup_flag" = false ] && [ "$optimize_flag" = false ] && [ "$run_flag" = false ] && [ "$status_flag" = false ]; then
        show_help
        exit 0
    fi
    
    # 処理実行
    if [ "$status_flag" = true ]; then
        check_status
    fi
    
    if [ "$cleanup_flag" = true ]; then
        cleanup
    fi
    
    if [ "$optimize_flag" = true ]; then
        optimize_env
    fi
    
    if [ "$run_flag" = true ]; then
        safe_predict_run
    fi
    
    log_success "処理が完了しました"
}

# スクリプト実行
main "$@"
