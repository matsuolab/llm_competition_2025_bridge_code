#!/bin/bash

# ===========================================
# v5 Job Monitoring Script
# ===========================================
# Monitors running v5 data synthesis jobs and provides real-time status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONITOR_NODES=("osk-gpu58" "osk-gpu59" "osk-gpu60" "osk-gpu61" "osk-gpu62" "osk-gpu63" "osk-gpu64" "osk-gpu65")
REFRESH_INTERVAL=5  # Refresh every 5 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show conflicting jobs
show_conflicting_jobs() {
    echo -e "${YELLOW}Conflicting Jobs (auto-cancelled by auto_submit):${NC}"

    # Check P07U010's auto_gen jobs
    local p07_jobs=$(squeue -u "P07U010" --noheader --format="%i %j %N %t" | grep "auto_gen" || true)
    if [ -n "$p07_jobs" ]; then
        echo -e "${RED}P07U010's auto_gen jobs:${NC}"
        echo "$p07_jobs"
    fi

    # Check kna.hata's jobs
    local kna_jobs=$(squeue -u "kna.hata" --noheader --format="%i %j %N %t" || true)
    if [ -n "$kna_jobs" ]; then
        echo -e "${RED}kna.hata's jobs:${NC}"
        echo "$kna_jobs"
    fi

    if [ -z "$p07_jobs" ] && [ -z "$kna_jobs" ]; then
        echo "No conflicting jobs found"
    fi
    echo ""
}

# Function to get job info
get_job_info() {
    local user="${1:-$USER}"

    echo -e "${BLUE}=== v5 Data Synthesis Job Monitor ===${NC}"
    echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "User: $user"
    echo ""

    # Get all jobs for user
    local jobs=$(squeue -u "$user" --format="%i|%j|%t|%M|%N|%S" --noheader | grep -E "auto2_gen|v5_|run_generate_data" || true)

    if [ -z "$jobs" ]; then
        echo -e "${YELLOW}No v5 jobs currently running${NC}"
        echo ""
        return
    fi

    echo -e "${GREEN}Active Jobs:${NC}"
    printf "%-10s %-20s %-10s %-10s %-15s %-20s\n" "JOB_ID" "NAME" "STATE" "TIME" "NODE" "START_TIME"
    echo "--------------------------------------------------------------------------------"

    while IFS='|' read -r job_id job_name state time node start_time; do
        # Color code by state
        case "$state" in
            "R")
                state_color="${GREEN}RUNNING${NC}"
                ;;
            "PD")
                state_color="${YELLOW}PENDING${NC}"
                ;;
            "CG")
                state_color="${YELLOW}COMPLETING${NC}"
                ;;
            *)
                state_color="${RED}$state${NC}"
                ;;
        esac

        printf "%-10s %-20s %-10b %-10s %-15s %-20s\n" \
            "$job_id" "$job_name" "$state_color" "$time" "$node" "$start_time"
    done <<< "$jobs"

    echo ""
}

# Function to check node availability
check_node_status() {
    echo -e "${BLUE}Node Status:${NC}"
    printf "%-15s %-10s %-15s %-20s %-10s\n" "NODE" "STATE" "JOBS_RUNNING" "GPU_MEMORY_USED" "AVAILABLE"
    echo "--------------------------------------------------------------------------------"

    for node in "${MONITOR_NODES[@]}"; do
        # Get node state
        local node_state=$(sinfo -n "$node" --noheader -o "%t" 2>/dev/null || echo "DOWN")

        # Count jobs on node
        local job_count=$(squeue -w "$node" -u "$USER" --noheader | wc -l)

        # Check if available
        local available="NO"
        local available_color="${RED}"
        if [[ "$node_state" =~ ^(idle|mix)$ ]] && [ "$job_count" -eq 0 ]; then
            available="YES"
            available_color="${GREEN}"
        elif [[ "$node_state" =~ ^(idle|mix)$ ]]; then
            available="BUSY"
            available_color="${YELLOW}"
        fi

        # Try to get GPU memory usage (if accessible)
        local gpu_mem="N/A"
        if command -v ssh &> /dev/null; then
            gpu_mem=$(timeout 2 ssh -o ConnectTimeout=1 "$node" \
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1" 2>/dev/null || echo "N/A")
            if [ "$gpu_mem" != "N/A" ]; then
                gpu_mem=$(echo "$gpu_mem" | sed 's/, / \/ /g')" MB"
            fi
        fi

        printf "%-15s %-10s %-15s %-20s ${available_color}%-10s${NC}\n" \
            "$node" "$node_state" "$job_count" "$gpu_mem" "$available"
    done

    echo ""
}

# Function to show output file statistics
show_output_stats() {
    echo -e "${BLUE}Output Statistics:${NC}"

    # Find recent output directories
    local output_base="/home/Competition2025/P05/shareP05/data_generation/data_generation_output/v5_doctoral_upgrade"

    if [ -d "$output_base" ]; then
        local recent_dirs=$(find "$output_base" -type d -name "auto_*" -mmin -1440 2>/dev/null | sort -r | head -5)

        if [ -n "$recent_dirs" ]; then
            echo "Recent output directories (last 24h):"
            while IFS= read -r dir; do
                if [ -d "$dir" ]; then
                    local count=$(find "$dir" -name "*.jsonl" -type f -exec wc -l {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
                    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                    echo "  $(basename "$dir"): ${count:-0} items, ${size:-0} total"
                fi
            done <<< "$recent_dirs"
        else
            echo "No recent output directories found"
        fi
    fi

    echo ""
}

# Function to show recent logs
show_recent_logs() {
    echo -e "${BLUE}Recent Log Activity:${NC}"

    local log_dir="${PROJECT_DIR}/logs"
    if [ -d "$log_dir" ]; then
        # Find most recent auto_submit log directory
        local recent_log_dir=$(find "$log_dir" -type d -name "auto_submit_*" -mmin -1440 2>/dev/null | sort -r | head -1)

        if [ -n "$recent_log_dir" ] && [ -d "$recent_log_dir" ]; then
            echo "Latest log directory: $(basename "$recent_log_dir")"

            # Show last few lines of main log
            local main_log="$recent_log_dir/auto_submit.log"
            if [ -f "$main_log" ]; then
                echo "Recent activity:"
                tail -5 "$main_log" | sed 's/^/  /'
            fi
        else
            echo "No recent log directories found"
        fi
    fi

    echo ""
}

# Main monitoring loop
main() {
    # Parse arguments
    local continuous=false
    local user="$USER"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --continuous|-c)
                continuous=true
                shift
                ;;
            --user|-u)
                user="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  -c, --continuous    Continuous monitoring mode"
                echo "  -u, --user USER    Monitor jobs for specific user"
                echo "  -h, --help         Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$continuous" = true ]; then
        while true; do
            clear
            get_job_info "$user"
            show_conflicting_jobs
            check_node_status
            show_output_stats
            show_recent_logs
            echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
            sleep "$REFRESH_INTERVAL"
        done
    else
        get_job_info "$user"
        show_conflicting_jobs
        check_node_status
        show_output_stats
        show_recent_logs
    fi
}

# Run main function
main "$@"