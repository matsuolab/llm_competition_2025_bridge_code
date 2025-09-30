#!/bin/bash

# 使用方法を表示する関数
usage() {
    echo "使用方法: $0 [オプション] [JOB_ID] [NODE]"
    echo ""
    echo "オプション:"
    echo "  -j, --job-id JOB_ID       接続するジョブIDを指定"
    echo "  -n, --node NODE           アタッチするノードを指定"
    echo "  -l, --list                実行中のジョブ一覧を表示"
    echo "  -h, --help                このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0                        # 実行中のジョブを自動検索して接続"
    echo "  $0 -j 12345               # ジョブID 12345 に接続"
    echo "  $0 -j 12345 -n osk-gpu01  # ジョブID 12345 のosk-gpu01ノードに接続"
    echo "  $0 -l                     # 実行中のジョブ一覧を表示"
    echo "  $0 12345                  # 位置引数でジョブID指定（従来互換）"
    echo "  $0 12345 osk-gpu01        # 位置引数でジョブIDとノード指定"
}

# デフォルト値
JOB_ID=""
NODE=""
SHOW_LIST=false

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--job-id)
            JOB_ID="$2"
            shift 2
            ;;
        -n|--node)
            NODE="$2"
            shift 2
            ;;
        -l|--list)
            SHOW_LIST=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "エラー: 不明なオプション $1"
            usage
            exit 1
            ;;
        *)
            # 位置引数として処理（従来互換性のため）
            if [[ -z "$JOB_ID" ]]; then
                JOB_ID="$1"
            elif [[ -z "$NODE" ]]; then
                NODE="$1"
            fi
            shift
            ;;
    esac
done

# ジョブ一覧表示オプション
if [[ "$SHOW_LIST" == true ]]; then
    echo "P05U019の実行中ジョブ一覧:"
    echo "--------------------------------------------------------------------------------------------------------"
    printf "%-10s %-15s %-10s %-10s %-5s %-10s\n" "JOB_ID" "NODELIST" "STATE" "TIME" "CPUS" "MEMORY"
    echo "--------------------------------------------------------------------------------------------------------"
    squeue -u P05U019 -h -o "%i %N %T %M %C %m" | while read line; do
        printf "%-10s %-15s %-10s %-10s %-5s %-10s\n" $line
    done
    exit 0
fi

# JOB_IDが指定されていない場合、自動検索
if [[ -z "$JOB_ID" ]]; then
    echo "P05U019の実行中ジョブを検索中..."
    SLURM_JOB_IDs=$(squeue -u P05U019 -h -o "%i" -t RUNNING)
    
    if [[ -z "$SLURM_JOB_IDs" ]]; then
        echo "エラー: 実行中のジョブが見つかりません"
        echo "実行中のジョブ一覧を確認するには: $0 -l"
        exit 1
    fi
    
    # ジョブ数をカウント
    JOB_COUNT=$(echo "$SLURM_JOB_IDs" | wc -w)
    
    if [[ $JOB_COUNT -eq 1 ]]; then
        # ジョブが1つだけの場合、それを使用
        JOB_ID="$SLURM_JOB_IDs"
        echo "見つかったジョブID: $JOB_ID"
    else
        # 複数のジョブがある場合は一覧表示して停止
        echo "複数のジョブが見つかりました。ジョブIDを指定してください:"
        echo "--------------------------------------------------------------------------------------------------------"
        printf "%-10s %-15s %-10s %-10s %-5s %-10s\n" "JOB_ID" "NODELIST" "STATE" "TIME" "CPUS" "MEMORY"
        echo "--------------------------------------------------------------------------------------------------------"
        squeue -u P05U019 -h -o "%i %N %T %M %C %m" -t RUNNING | while read line; do
            printf "%-10s %-15s %-10s %-10s %-5s %-10s\n" $line
        done
        echo ""
        echo "使用例: $0 -j <JOB_ID>"
        exit 1
    fi
else
    echo "指定されたジョブID: $JOB_ID"
fi

# ジョブの存在確認
if ! squeue -j "$JOB_ID" -h > /dev/null 2>&1; then
    echo "エラー: ジョブID $JOB_ID が見つかりません"
    echo "実行中のジョブ一覧を確認するには: $0 -l"
    exit 1
fi

# ジョブ情報を表示
echo "ジョブ情報:"
squeue -j "$JOB_ID" -o "%i %N %T %M %u %C %m"

# srun コマンドの構築
SRUN_CMD="srun -v --jobid=\"$JOB_ID\" --overlap --pty"

# ノードが指定されている場合は追加
if [[ -n "$NODE" ]]; then
    echo "指定されたノード: $NODE"
    SRUN_CMD="$SRUN_CMD --nodelist=\"$NODE\""
fi

echo ""
echo "ジョブID $JOB_ID に接続中..."
if [[ -n "$NODE" ]]; then
    echo "ノード: $NODE"
fi

# srun コマンドを実行
eval "$SRUN_CMD -- /bin/bash -l"
