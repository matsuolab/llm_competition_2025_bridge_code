#!/bin/bash

# ===========================================
# Submit Auto Runner as Background Process
# ===========================================
# Runs the auto_submit_v5.sh script as a background process with nohup
# This ensures the script continues running even after logout without occupying a compute node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments (pass all to auto_submit script)
ARGS="$@"

# Create log directory
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/auto_runner_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/auto_runner.pid"

# Check if another auto runner is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "✗ Auto runner is already running (PID: $OLD_PID)"
        echo "  Stop it with: kill $OLD_PID"
        exit 1
    else
        echo "Found stale PID file, removing..."
        rm -f "$PID_FILE"
    fi
fi

# Start the auto submit script in background with nohup
echo "Starting auto runner as background process..."
echo "Arguments: $ARGS"
echo "Log file: $LOG_FILE"

cd "$PROJECT_DIR"
nohup ./scripts/auto_submit_v5.sh $ARGS > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"

# Verify process started
sleep 2
if ps -p $PID > /dev/null; then
    echo "✓ Auto runner started (PID: $PID)"
    echo ""
    echo "Commands to manage the auto runner:"
    echo "  Check if running:  ps -p $PID"
    echo "  View logs:         tail -f $LOG_FILE"
    echo "  Stop runner:       kill $PID"
    echo ""
    echo "The auto runner will continue even if you logout."
else
    echo "✗ Failed to start auto runner"
    rm -f "$PID_FILE"
    exit 1
fi