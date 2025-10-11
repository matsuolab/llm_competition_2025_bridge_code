#!/bin/bash

# ===========================================
# Stop Auto Submit Script
# ===========================================
# Stops the auto_submit_v5.sh script and optionally cancels all submitted jobs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
CANCEL_JOBS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cancel-jobs|-c)
            CANCEL_JOBS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --cancel-jobs    Also cancel all submitted v5 jobs"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Stopping auto_submit_v5.sh...${NC}"

# Find and kill auto_submit_v5.sh processes
PIDS=$(pgrep -f "auto_submit_v5.sh" || true)

if [ -n "$PIDS" ]; then
    echo -e "Found auto_submit processes: $PIDS"

    # Try graceful shutdown first
    for pid in $PIDS; do
        echo -e "Sending SIGTERM to PID $pid..."
        kill "$pid" 2>/dev/null || true
    done

    # Wait a moment
    sleep 2

    # Check if still running and force kill if necessary
    REMAINING=$(pgrep -f "auto_submit_v5.sh" || true)
    if [ -n "$REMAINING" ]; then
        echo -e "${YELLOW}Some processes still running, forcing shutdown...${NC}"
        for pid in $REMAINING; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi

    echo -e "${GREEN}✓ Auto submit script stopped${NC}"
else
    echo -e "${YELLOW}No auto_submit_v5.sh process found${NC}"
fi

# Optionally cancel all jobs
if [ "$CANCEL_JOBS" = true ]; then
    echo -e "${YELLOW}Cancelling all v5 jobs...${NC}"

    # Find all v5 related jobs
    JOBS=$(squeue -u "$USER" --format="%i %j" --noheader | grep -E "auto2_gen|v5_auto|run_generate_data" | awk '{print $1}' || true)

    if [ -n "$JOBS" ]; then
        JOB_COUNT=$(echo "$JOBS" | wc -l)
        echo -e "Found $JOB_COUNT jobs to cancel"

        for job_id in $JOBS; do
            echo -e "  Cancelling job $job_id..."
            # Try custom cancel script first, fallback to scancel
            if [ -f "/home/Competition2025/P05/shareP05/scripts/scancel.sh" ]; then
                bash /home/Competition2025/P05/shareP05/scripts/scancel.sh "$job_id" 2>/dev/null || \
                    scancel "$job_id" 2>/dev/null || true
            else
                scancel "$job_id" 2>/dev/null || true
            fi
        done

        echo -e "${GREEN}✓ All jobs cancelled${NC}"
    else
        echo -e "${YELLOW}No active v5 jobs found${NC}"
    fi
else
    # Show active jobs but don't cancel
    echo -e "${YELLOW}Checking active v5 jobs...${NC}"
    JOBS=$(squeue -u "$USER" --format="%i %j %t %N" --noheader | grep -E "auto2_gen|v5_auto|run_generate_data" || true)

    if [ -n "$JOBS" ]; then
        echo -e "${YELLOW}Note: The following jobs are still running:${NC}"
        echo "$JOBS"
        echo ""
        echo -e "${YELLOW}To cancel these jobs, run: $0 --cancel-jobs${NC}"
    else
        echo -e "${GREEN}No active v5 jobs${NC}"
    fi
fi

echo -e "${GREEN}Done!${NC}"