#!/bin/bash
#
# Conda環境を生成し、環境変数を設定するスクリプト。
# 親スクリプトから `source` コマンドで実行されることを想定しています。
# 設定値はすべて、事前に読み込まれた環境変数（env_config.sh）から取得します。
#

# --- 事前チェック ---
# 必要な環境変数が存在するか確認
# CONDA_ENV_FULL_PATHもチェック対象に加える
if [ -z "$CONDA_ENV_NAME" ] || [ -z "$PYTHON_VERSION" ] || [ -z "$CONDA_ENV_FULL_PATH" ]; then
    log_error "必要な環境変数 (CONDA_ENV_NAME, PYTHON_VERSION, CONDA_ENV_FULL_PATH) が設定されていません。"
    return 1 # sourceで実行されているため exit ではなく return を使用
fi

print_header "Conda環境 '${CONDA_ENV_NAME}' のセットアップ"

# --- Conda環境の生成と環境変数の初期化 ---
if [ ! -d "$CONDA_ENV_FULL_PATH" ]; then
    start_timer "Conda環境 '${CONDA_ENV_NAME}' (Python ${PYTHON_VERSION}) を新規作成します..."
    # --prefixオプションで指定したパスに環境を作成する
    if ! conda create --prefix "$CONDA_ENV_FULL_PATH" python="$PYTHON_VERSION" -y --quiet; then
        log_error "Conda環境 '${CONDA_ENV_NAME}' の作成に失敗しました。"
        return 1
    fi
    end_timer

    # --- activate.d スクリプトの作成 ---
    log_info "アクティベート時に読み込む環境変数スクリプトを作成します..."
    ACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/activate.d"
    ACTIVATE_SCRIPT="$ACTIVATE_DIR/env_vars.sh"
    mkdir -p "$ACTIVATE_DIR"

    PYTHON_LIB_VERSION="${PYTHON_VERSION%.*}"
    LD_LIB_PATH_TO_ADD="${SYSTEM_LD_LIB_PATHS}:${CONDA_ENV_FULL_PATH}/lib:${CONDA_ENV_FULL_PATH}/lib/python${PYTHON_LIB_VERSION}/site-packages/torch/lib"

    cat <<EOF > "$ACTIVATE_SCRIPT"
#!/bin/bash
export ORIGINAL_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${LD_LIB_PATH_TO_ADD}:\$LD_LIBRARY_PATH"
export CUDNN_PATH="$CONDA_ENV_FULL_PATH/lib"
export CUDA_HOME="$CONDA_ENV_FULL_PATH/"
EOF
    chmod +x "$ACTIVATE_SCRIPT"
    log_info "作成完了: $ACTIVATE_SCRIPT"

    # --- deactivate.d スクリプトの作成 ---
    log_info "ディアクティベート時に読み込む環境変数スクリプトを作成します..."
    DEACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/deactivate.d"
    DEACTIVATE_SCRIPT="$DEACTIVATE_DIR/env_vars_rollback.sh"
    mkdir -p "$DEACTIVATE_DIR"

    cat <<EOF > "$DEACTIVATE_SCRIPT"
#!/bin/bash
export LD_LIBRARY_PATH="\$ORIGINAL_LD_LIBRARY_PATH"
unset CUDNN_PATH
unset CUDA_HOME
unset ORIGINAL_LD_LIBRARY_PATH
EOF
    chmod +x "$DEACTIVATE_SCRIPT"
    log_info "作成完了: $DEACTIVATE_SCRIPT"

else
    log_warn "Conda環境 '${CONDA_ENV_NAME}' は既に存在するため、新規作成をスキップします。"
fi

# --- 環境のアクティベートと確認 ---
print_header "環境 '${CONDA_ENV_NAME}' のアクティベートと状態確認"
# フルパスでアクティベートするのが確実
log_info "Conda環境をフルパスでアクティベートします: ${CONDA_ENV_FULL_PATH}"
conda activate "${CONDA_ENV_FULL_PATH}"
log_success "Conda環境 '${CONDA_ENV_NAME}' がアクティベートされました。"

log_info "--- 現在のConda環境 ---"
conda env list | grep '*'
log_info "--- PythonとPipのパス ---"
log_info "Python: $(which python)"
log_info "Pip:    $(which pip)"
log_info "--- カスタム環境変数 ---"
printenv | grep -E '^(CONDA_PREFIX|CUDA_HOME|CUDNN_PATH|LD_LIBRARY_PATH)'

log_success "Conda環境のセットアップスクリプトが完了しました。"