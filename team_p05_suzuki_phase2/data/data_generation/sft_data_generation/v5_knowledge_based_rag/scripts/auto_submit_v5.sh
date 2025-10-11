#!/bin/bash

# ===========================================
# Auto Submit V5 - Enhanced Data Generation Job Orchestrator
# ===========================================
# Automatically manages job submissions across available GPU nodes
# Features:
# - Dynamic node allocation based on availability
# - Graceful handling of job cancellations by other users
# - Automatic retry with intelligent node selection
# - Small chunk processing for better load balancing
# ===========================================

set +e  # Don't exit on error - we handle errors gracefully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"

# Configuration
MONITOR_NODES=("osk-gpu58" "osk-gpu59" "osk-gpu60" "osk-gpu61" "osk-gpu62" "osk-gpu63" "osk-gpu64" "osk-gpu65")
CHECK_INTERVAL=15  # Check every 15 seconds
MAX_JOBS_PER_NODE=1  # Maximum jobs per node

# Parse command line arguments
DATASET="team-suzuki/seed_data_v1"  # Default dataset
LIMIT=0  # Default to 0 (unlimited)
BATCH_SIZE=3  # Default batch size
OUTPUT_DIR=""
NO_KNOWLEDGE_RAG=""
TOTAL_TO_PROCESS=0  # Will be set based on LIMIT
DRY_RUN=false

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            TOTAL_TO_PROCESS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-knowledge-rag)
            NO_KNOWLEDGE_RAG="--no-knowledge-rag"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET      Dataset name (default: team-suzuki/seed_data_v1)"
            echo "  --limit N              Process only N items (default: unlimited)"
            echo "  --batch-size N         Batch size per job (default: 3)"
            echo "  --output-dir DIR       Output directory"
            echo "  --no-knowledge-rag     Disable knowledge RAG"
            echo "  --dry-run             Show what would be done without actually submitting"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Count total items in dataset if LIMIT is 0
if [ "$LIMIT" -eq 0 ]; then
    # Check if dataset is a file path
    if [[ "$DATASET" == /* ]] || [[ "$DATASET" == ./* ]]; then
        # It's a file path
        if [ -f "$DATASET" ]; then
            TOTAL_TO_PROCESS=$(wc -l < "$DATASET")
        else
            echo "ERROR: Dataset file not found: $DATASET"
            exit 1
        fi
    else
        # It's a Hugging Face dataset name (e.g., team-suzuki/seed_data_v1)
        # For HF datasets, we'll pass it directly to the generation script
        # which will handle loading from HuggingFace
        # Estimate a reasonable number for tracking (will be updated by actual processing)
        TOTAL_TO_PROCESS=1000  # Default estimate for HF datasets
        echo "Note: Using Hugging Face dataset: $DATASET"
        echo "      Actual item count will be determined during processing"
    fi
fi

# Job tracking associative arrays
declare -A SUBMITTED_JOBS  # Maps node -> job_id
declare -A JOB_ITEM_COUNTS  # Maps job_id -> number of items
declare -A RETRY_COUNTS  # Maps job_id -> retry count
declare -A JOB_OUTPUT_DIRS  # Track output directories for each job (for retry)
declare -A CANCELLED_ITEMS  # Track items from cancelled jobs for rescheduling
TOTAL_SUBMITTED=0
TOTAL_COMPLETED=0
TOTAL_FAILED=0
TOTAL_CANCELLED=0
MAX_RETRIES=10  # Maximum retry attempts for failed jobs
CANCELLED=false
START_TIME=$(date +%s)
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$  # Unique run ID for this auto_submit session

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_submit_${RUN_ID}.log"
}

# Cancel conflicting jobs from specific users
cancel_conflicting_jobs() {
    local node=$1
    local cancelled=false

    # Cancel P07U010's auto_gen* jobs
    local p07_jobs=$(squeue -w "$node" -u P07U010 --noheader -o "%i %j" 2>/dev/null | grep "auto_gen" | awk '{print $1}')
    if [ -n "$p07_jobs" ]; then
        for job_id in $p07_jobs; do
            log "Cancelling P07U010's auto_gen job $job_id on $node"
            # Use timeout to prevent hanging
            timeout 5 bash /home/Competition2025/P05/shareP05/scripts/scancel.sh "$job_id" &>/dev/null || \
                scancel "$job_id" 2>/dev/null || true
            cancelled=true
        done
    fi

    # Cancel kan.hata's jobs
    local kan_jobs=$(squeue -w "$node" -u kan.hata --noheader -o "%i" 2>/dev/null)
    if [ -n "$kan_jobs" ]; then
        for job_id in $kan_jobs; do
            log "Cancelling kan.hata's job $job_id on $node"
            # Use timeout to prevent hanging
            timeout 5 bash /home/Competition2025/P05/shareP05/scripts/scancel.sh "$job_id" &>/dev/null || \
                scancel "$job_id" 2>/dev/null || true
            cancelled=true
        done
    fi

    # Only wait if we actually cancelled something
    if [ "$cancelled" = true ]; then
        sleep 2
    fi
}

# Check if node is available
is_node_available() {
    local node=$1

    # Check if node is online (idle or mix state)
    local node_state=$(sinfo -n "$node" --noheader -o "%t" 2>/dev/null || echo "DOWN")

    # Skip if node is down or drained
    if [[ "$node_state" == "down"* ]] || [[ "$node_state" == "drain"* ]]; then
        return 1
    fi

    # Check if ANY user has jobs on this node (not just current user)
    local total_jobs=$(squeue -w "$node" --noheader | wc -l)

    if [ "$total_jobs" -gt 0 ]; then
        # Node has jobs, check if we should cancel conflicting ones
        cancel_conflicting_jobs "$node"

        # Re-check total jobs after cancellation
        sleep 1
        total_jobs=$(squeue -w "$node" --noheader | wc -l)

        if [ "$total_jobs" -gt 0 ]; then
            # Still has jobs from other users, node not available
            return 1
        fi
    fi

    # Check current user's jobs on the node
    local my_job_count=$(squeue -w "$node" -u "$USER" --noheader | wc -l)

    if [ "$my_job_count" -lt "$MAX_JOBS_PER_NODE" ]; then
        return 0
    else
        return 1
    fi
}

# Submit job to node
submit_job() {
    local node=$1
    local items_to_process=$2
    local retry_job_id=${3:-}  # Optional: job ID for retry

    # For retries, use the same output directory
    local job_output_dir
    if [ -n "$retry_job_id" ] && [ -n "${JOB_OUTPUT_DIRS[$retry_job_id]}" ]; then
        job_output_dir="${JOB_OUTPUT_DIRS[$retry_job_id]}"
        log "Retrying job $retry_job_id, using existing output dir: $job_output_dir"
    else
        # New job - create stable directory name based on run ID and node
        job_output_dir="${OUTPUT_DIR:-/home/Competition2025/P05/shareP05/data_generation/data_generation_output/v5_doctoral_upgrade/auto_${RUN_ID}_${node}}"
    fi

    # Build sbatch command
    local sbatch_cmd="NODE=\"$node\" sbatch --nodelist=$node"
    sbatch_cmd="$sbatch_cmd --job-name=auto2_gen"
    sbatch_cmd="$sbatch_cmd --output=${LOG_DIR}/job_${node}_%j.out"
    sbatch_cmd="$sbatch_cmd --error=${LOG_DIR}/job_${node}_%j.err"
    sbatch_cmd="$sbatch_cmd ${SCRIPT_DIR}/run_generate_data.sh"
    sbatch_cmd="$sbatch_cmd --dataset \"$DATASET\""
    sbatch_cmd="$sbatch_cmd --limit $items_to_process"
    sbatch_cmd="$sbatch_cmd --batch-size $BATCH_SIZE"
    sbatch_cmd="$sbatch_cmd --output-dir \"$job_output_dir\""

    if [ -n "$NO_KNOWLEDGE_RAG" ]; then
        sbatch_cmd="$sbatch_cmd $NO_KNOWLEDGE_RAG"
    fi

    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would submit to $node: $sbatch_cmd"
        return 0
    fi

    log "Submitting job to $node (processing $items_to_process items)..."
    local job_output=$(eval "$sbatch_cmd" 2>&1)
    local job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')

    if [ -n "$job_id" ]; then
        SUBMITTED_JOBS[$node]=$job_id
        JOB_ITEM_COUNTS[$job_id]=$items_to_process
        JOB_OUTPUT_DIRS[$job_id]=$job_output_dir

        if [ -n "$retry_job_id" ]; then
            # Carry over retry count from original job
            RETRY_COUNTS[$job_id]="${RETRY_COUNTS[$retry_job_id]}"
            unset RETRY_COUNTS[$retry_job_id]
            unset JOB_OUTPUT_DIRS[$retry_job_id]
        else
            RETRY_COUNTS[$job_id]=0
        fi

        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + items_to_process))
        log "✓ Job $job_id submitted to $node (items: $items_to_process, output: $job_output_dir)"
        return 0
    else
        log "✗ Failed to submit job to $node: $job_output"
        return 1
    fi
}

# Check job status
check_job_status() {
    local node=$1
    local job_id="${SUBMITTED_JOBS[$node]}"

    if [ -z "$job_id" ]; then
        return 2  # No job on this node
    fi

    local status=$(squeue -j "$job_id" --noheader -o "%t" 2>/dev/null || echo "")

    case "$status" in
        "R"|"PD"|"CF"|"CG")
            return 0  # Job is running or pending
            ;;
        "")
            # Job completed or disappeared, check sacct
            # Use head -1 to get main job status, not substeps
            local final_status=$(sacct -j "$job_id" --noheader -o "State" | head -1 | tr -d ' ')
            if [[ "$final_status" == "COMPLETED" ]]; then
                log "Job $job_id on $node completed successfully"
                local item_count="${JOB_ITEM_COUNTS[$job_id]:-0}"
                TOTAL_COMPLETED=$((TOTAL_COMPLETED + item_count))
                unset SUBMITTED_JOBS[$node]
                unset JOB_ITEM_COUNTS[$job_id]
                unset JOB_OUTPUT_DIRS[$job_id]
                unset RETRY_COUNTS[$job_id]
                return 1  # Job completed
            elif [[ "$final_status" == "CANCELLED"* ]]; then
                log "Job $job_id on $node was cancelled (likely by another user needing the node)"
                # Cancelled jobs should be rescheduled to ensure all data is processed
                local item_count="${JOB_ITEM_COUNTS[$job_id]:-0}"

                # Don't count cancellation as a retry failure
                log "Will reschedule $item_count items when a node becomes available"

                # Store cancelled items for rescheduling
                CANCELLED_ITEMS[$job_id]=$item_count
                TOTAL_CANCELLED=$((TOTAL_CANCELLED + 1))

                # Reduce TOTAL_SUBMITTED to allow re-submission
                TOTAL_SUBMITTED=$((TOTAL_SUBMITTED - item_count))

                # Clean up job tracking but preserve retry count for rescheduling
                local retry_count="${RETRY_COUNTS[$job_id]:-0}"
                unset SUBMITTED_JOBS[$node]

                # We'll reschedule this in the main loop
                return 3  # Job cancelled - needs rescheduling
            else
                log "Job $job_id on $node failed with status: $final_status"
                # Handle failed job with retry logic
                local item_count="${JOB_ITEM_COUNTS[$job_id]:-0}"
                local retry_count="${RETRY_COUNTS[$job_id]:-0}"

                if [ "$retry_count" -lt "$MAX_RETRIES" ]; then
                    log "Will retry job (attempt $((retry_count + 1))/$MAX_RETRIES) for $item_count items"
                    # Reduce TOTAL_SUBMITTED to allow re-submission
                    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED - item_count))
                    TOTAL_FAILED=$((TOTAL_FAILED + 1))
                    # Increment retry count for tracking
                    RETRY_COUNTS[$job_id]=$((retry_count + 1))

                    # Store failed job ID for retry with same output directory
                    local failed_job_id="$job_id"
                    unset SUBMITTED_JOBS[$node]

                    # Try to find an available node for retry
                    # Prefer a different node if possible (might have better success)
                    local retried=false
                    local shuffled_nodes=($(printf '%s\n' "${MONITOR_NODES[@]}" | shuf))

                    for alt_node in "${shuffled_nodes[@]}"; do
                        if [ "$alt_node" != "$node" ] && is_node_available "$alt_node"; then
                            log "Retrying on alternative node: $alt_node"
                            submit_job "$alt_node" "$item_count" "$failed_job_id"
                            retried=true
                            break
                        fi
                    done

                    # If no alternative node, retry on same node if available
                    if [ "$retried" = false ] && is_node_available "$node"; then
                        log "Retrying on same node: $node"
                        submit_job "$node" "$item_count" "$failed_job_id"
                    elif [ "$retried" = false ]; then
                        log "No nodes available for retry, will try again later"
                        # Store for later retry
                        CANCELLED_ITEMS[$failed_job_id]=$item_count
                    fi
                else
                    log "Job exceeded max retries ($MAX_RETRIES). Skipping $item_count items."
                    # Keep TOTAL_SUBMITTED as is but mark as permanently failed
                    unset SUBMITTED_JOBS[$node]
                    unset JOB_ITEM_COUNTS[$job_id]
                    unset JOB_OUTPUT_DIRS[$job_id]
                    unset RETRY_COUNTS[$job_id]
                fi

                return 4  # Job failed
            fi
            ;;
        *)
            log "Job $job_id on $node has unexpected status: $status"
            return 5
            ;;
    esac
}

# Handle cancellation
handle_cancel() {
    log "Received cancellation signal"
    CANCELLED=true

    # Cancel all submitted jobs
    for node in "${!SUBMITTED_JOBS[@]}"; do
        local job_id="${SUBMITTED_JOBS[$node]}"
        if [ -n "$job_id" ]; then
            log "Cancelling job $job_id on $node..."
            scancel "$job_id" 2>/dev/null || true
        fi
    done

    log "All jobs cancelled. Exiting."
}

# Set up signal handlers
trap handle_cancel SIGINT SIGTERM

# Calculate how many items to process per job
calculate_items_per_job() {
    # First, check if we have cancelled items to reschedule
    for job_id in "${!CANCELLED_ITEMS[@]}"; do
        local items="${CANCELLED_ITEMS[$job_id]}"
        if [ "$items" -gt 0 ]; then
            # Return cancelled items for rescheduling
            unset CANCELLED_ITEMS[$job_id]
            echo "$items"
            return
        fi
    done

    # Calculate new items to process
    if [ "$TOTAL_TO_PROCESS" -gt 0 ]; then
        local remaining=$((TOTAL_TO_PROCESS - TOTAL_SUBMITTED))
        if [ "$remaining" -le 0 ]; then
            echo 0
            return
        fi

        # Count available nodes
        local available_nodes=0
        for node in "${MONITOR_NODES[@]}"; do
            if is_node_available "$node"; then
                available_nodes=$((available_nodes + 1))
            fi
        done

        if [ "$available_nodes" -eq 0 ]; then
            echo 0
            return
        fi

        # Use smaller chunks for better dynamic allocation
        # This allows jobs to be distributed as nodes become available
        local chunk_size=$((BATCH_SIZE * 10))  # Process 10 batches at a time

        if [ "$remaining" -lt "$chunk_size" ]; then
            echo "$remaining"
        else
            echo "$chunk_size"
        fi
    else
        # No limit, use moderate chunk size for flexibility
        echo $((BATCH_SIZE * 10))
    fi
}

# Main monitoring loop
main() {
    # Add error trap to log failures but continue execution
    trap 'log "WARNING: Command failed at line $LINENO (exit code: $?) - continuing..."' ERR

    log "=========================================="
    log "Starting automated v5 data synthesis job submission"
    log "Dataset: $DATASET"
    log "Total items to process: ${TOTAL_TO_PROCESS:-unlimited}"
    log "Batch size: $BATCH_SIZE"
    log "Chunk size: $((BATCH_SIZE * 10)) items per job"
    log "Monitoring nodes: ${MONITOR_NODES[*]}"
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Dry run: $DRY_RUN"
    log "Note: Jobs may be cancelled if other users need nodes"
    log "      Cancelled jobs will be automatically rescheduled"
    log "=========================================="

    local iteration=0

    while true; do
        iteration=$((iteration + 1))

        # Check if we've processed everything
        if [ "$TOTAL_TO_PROCESS" -gt 0 ] && [ "$TOTAL_SUBMITTED" -ge "$TOTAL_TO_PROCESS" ]; then
            # Check if all jobs are complete
            local active_jobs=0
            for node in "${!SUBMITTED_JOBS[@]}"; do
                if [ -n "${SUBMITTED_JOBS[$node]}" ]; then
                    active_jobs=$((active_jobs + 1))
                fi
            done

            # Also check for items pending rescheduling
            local pending_reschedule=0
            for job_id in "${!CANCELLED_ITEMS[@]}"; do
                pending_reschedule=$((pending_reschedule + ${CANCELLED_ITEMS[$job_id]}))
            done

            if [ "$active_jobs" -eq 0 ] && [ "$pending_reschedule" -eq 0 ]; then
                log "✓ All data has been processed. Total submitted: $TOTAL_SUBMITTED"
                break
            fi
        fi

        # Check if cancelled
        if [ "$CANCELLED" = true ]; then
            break
        fi

        # Status update every 10 iterations
        if [ $((iteration % 10)) -eq 1 ]; then
            local pending_reschedule=0
            for job_id in "${!CANCELLED_ITEMS[@]}"; do
                pending_reschedule=$((pending_reschedule + ${CANCELLED_ITEMS[$job_id]}))
            done
            log "Status: Submitted=$TOTAL_SUBMITTED/${TOTAL_TO_PROCESS:-∞}, Completed=$TOTAL_COMPLETED, Failed=$TOTAL_FAILED, Cancelled=$TOTAL_CANCELLED, Active=${#SUBMITTED_JOBS[@]}, Pending_Reschedule=$pending_reschedule"
        fi

        # Check status of existing jobs
        for node in "${!SUBMITTED_JOBS[@]}"; do
            check_job_status "$node"
        done

        # Calculate how many items to process per job
        local items_per_job=$(calculate_items_per_job)

        if [ "$items_per_job" -gt 0 ]; then
            # Check for available nodes and submit jobs
            # Randomize node order for better distribution
            local shuffled_nodes=($(printf '%s\n' "${MONITOR_NODES[@]}" | shuf))
            local submitted_this_round=false

            for node in "${shuffled_nodes[@]}"; do
                if [ "$items_per_job" -le 0 ] || [ "$submitted_this_round" = true ]; then
                    break  # No more items to process or already submitted
                fi

                if is_node_available "$node"; then
                    # Submit job to this node
                    if submit_job "$node" "$items_per_job"; then
                        submitted_this_round=true
                        # Only recalculate if we might submit more
                        # For now, submit only one job per iteration to prevent duplicates
                        break
                    fi
                fi
            done
        fi

        # Sleep before next check
        sleep "$CHECK_INTERVAL"
    done

    # Final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "=========================================="
    log "Automation completed"
    log "Total submitted: $TOTAL_SUBMITTED"
    log "Total completed: $TOTAL_COMPLETED"
    log "Total failed after retries: $((TOTAL_SUBMITTED - TOTAL_COMPLETED))"
    log "Total cancellations: $TOTAL_CANCELLED"
    log "Duration: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
    log "=========================================="
}

# Start main loop
main