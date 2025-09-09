
#!/bin/bash
#SBATCH --job-name=safety_cot_eval
#SBATCH --output=safety_eval_%j.out
#SBATCH --error=safety_eval_%j.err
#SBATCH --partition=P[]
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu[]

# Slurmジョブ内かどうかを判定
set -e


# Create log directory if it doesn't exist


# Activate virtual environment (adjust path as needed)
source haraguchi-env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_TOKEN="your_huggingface_token"  # Replace with your actual token
export HF_HOME="your_repository"

# Install dependencies if needed (can be commented out after first run)
# pip install -r requirements.txt

# Run the main script with 8 GPUs for tensor parallelism
python dna_cot_synthesis.py \
    --model "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5" \
    --batch-size 512 \
    --tensor-parallel 8 \
    --dataset "LLMcompe-Team-Watanabe/translated_answer_carefully_DPO" \
    --output-repo "LLMcompe-Team-Watanabe/safety_CoT_dataset"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed at: $(date)"
    exit 1
fi
