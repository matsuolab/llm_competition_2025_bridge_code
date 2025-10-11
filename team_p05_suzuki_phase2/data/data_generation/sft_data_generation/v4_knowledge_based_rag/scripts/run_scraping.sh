#!/bin/bash
#SBATCH --job-name=arxiv_scraper
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/Competition2025/P05/P05U001/team_suzuki/logs/arxiv_scraper_%j.out
#SBATCH --error=/home/Competition2025/P05/P05U001/team_suzuki/logs/arxiv_scraper_%j.err

# ArXiv Scraping Script
# ArXivスクレイピング実行スクリプト

set -e

# Configuration
PAPERS_PER_BATCH="${1:-200}"
OUTPUT_DIR="${2:-/home/Competition2025/P05/shareP05/data_generation/arxiv_papers}"
MIN_QUALITY="${3:-30}"
BATCH_INTERVAL="${4:-300}"  # 5 minutes default
CHECKPOINT_DIR="${5:-./checkpoints}"

# Setup paths
SCRIPT_DIR="/home/Competition2025/P05/P05U001/team_suzuki/data/data_generation/sft_data_generation/v4_knowledge_based_rag"
ARXIV_DIR="${SCRIPT_DIR}/arxiv"

# Environment setup
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Create directories
mkdir -p "$OUTPUT_DIR" "$(dirname "$CHECKPOINT_DIR")"

# Activate conda environment
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
SHARED_ENV_PATH="/home/Competition2025/P05/shareP05/data_generation/data_generation_env"
if [ -d "${SHARED_ENV_PATH}" ]; then
    conda activate "${SHARED_ENV_PATH}"
else
    conda activate /home/Competition2025/P05/P05U001/envs/data_generation_env
fi

echo "========================================"
echo "🚀 ArXiv Paper Collection"
echo "========================================"
echo "Papers per batch: $PAPERS_PER_BATCH"
echo "Output directory: $OUTPUT_DIR"
echo "Min quality score: $MIN_QUALITY"
echo "Batch interval: $BATCH_INTERVAL seconds"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Start time: $(date)"
echo ""

# Change to arxiv directory
cd "$ARXIV_DIR"

# Run arxiv scraper with comprehensive categories
python arxiv_scraper.py \
  --papers-per-batch "$PAPERS_PER_BATCH" \
  --categories cs.AI cs.LG cs.CL cs.CV cs.CC cs.LO cs.DS cs.IT \
               math.CO math.PR math.NT math.LO math.OC math.DS math.GR \
               stat.ML stat.TH stat.CO stat.ME stat.AP \
               physics.comp-ph eess.SY eess.SP \
  --min-quality "$MIN_QUALITY" \
  --use-citations \
  --output-dir "$OUTPUT_DIR" \
  --batch-interval "$BATCH_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --save-interval 5 \
  --enable-rag-chunking

echo ""
echo "========================================"
echo "✅ ArXiv Paper Collection Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - Safe papers: ${OUTPUT_DIR}/safe_papers_*.jsonl"
echo "  - RAG chunks: ${OUTPUT_DIR}/rag_data/rag_chunks_*.jsonl"
echo "  - Metadata: ${OUTPUT_DIR}/metadata_*.json"
echo "  - Report: ${OUTPUT_DIR}/scraping_report_*.txt"
echo ""
echo "To build knowledge index from collected papers:"
echo "  ./scripts/run_build_index.sh"