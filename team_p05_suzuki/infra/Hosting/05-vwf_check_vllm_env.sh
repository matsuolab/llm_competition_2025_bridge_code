
#!/usr/bin/env bash
set -eu

# Create logs directory if it doesn't exist
mkdir -p ./logs

OUT="./logs/vllm_env_check_$(date +%s).txt"
echo "==== vLLM / System environment check ====" | tee $OUT

# Initialize counters and arrays for tracking issues
OK_COUNT=0
WARNING_COUNT=0
NG_COUNT=0
WARNING_MESSAGES=()
NG_MESSAGES=()

# Function to print OK/NG/WARNING status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo "✅ $message" | tee -a $OUT
        OK_COUNT=$((OK_COUNT + 1))
    elif [ "$status" = "WARNING" ]; then
        echo "⚠️ $message" | tee -a $OUT
        WARNING_COUNT=$((WARNING_COUNT + 1))
        WARNING_MESSAGES+=("$message")
    else
        echo "❌ $message" | tee -a $OUT
        NG_COUNT=$((NG_COUNT + 1))
        NG_MESSAGES+=("$message")
    fi
}

echo -e "\n🚀 1) Basic host info" | tee -a $OUT
echo "Hostname: $(hostname)" | tee -a $OUT
echo "Date: $(date)" | tee -a $OUT
echo "Uptime: $(uptime -p)" | tee -a $OUT
print_status "OK" "Basic host information retrieved successfully"

echo -e "\n🚀 2) CPU / memory" | tee -a $OUT
echo "nproc (logical CPUs): $(nproc)" | tee -a $OUT
lscpu | tee -a $OUT
echo -e "\n/proc/meminfo (summary):" | tee -a $OUT
grep -E 'MemTotal|MemFree|SwapTotal|SwapFree' /proc/meminfo | tee -a $OUT
free -h | tee -a $OUT

# Check memory status
total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
free_mem_kb=$(grep MemFree /proc/meminfo | awk '{print $2}')
total_mem_gb=$((total_mem_kb / 1024 / 1024))
free_mem_gb=$((free_mem_kb / 1024 / 1024))

if [ $total_mem_gb -ge 16 ]; then
    if [ $free_mem_gb -ge 4 ]; then
        print_status "OK" "Memory: ${total_mem_gb}GB total, ${free_mem_gb}GB free - sufficient for vLLM"
    else
        print_status "NG" "Memory: ${total_mem_gb}GB total, ${free_mem_gb}GB free - low free memory"
    fi
else
    print_status "NG" "Memory: ${total_mem_gb}GB total - insufficient for vLLM (recommend 16GB+)"
fi

echo -e "\n🚀 3) GPU info" | tee -a $OUT
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L | tee -a $OUT
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | tee -a $OUT
  
  # Check GPU status
  gpu_count=$(nvidia-smi -L | wc -l)
  if [ $gpu_count -gt 0 ]; then
    # Check if any GPU has sufficient memory (at least 8GB)
    gpu_mem_ok=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '$1 >= 8192 {count++} END {print count+0}')
    if [ $gpu_mem_ok -gt 0 ]; then
      print_status "OK" "GPU: ${gpu_count} GPU(s) detected with sufficient memory for vLLM"
    else
      print_status "NG" "GPU: ${gpu_count} GPU(s) detected but insufficient memory (need 8GB+)"
    fi
  else
    print_status "NG" "GPU: No GPUs detected"
  fi
else
  echo "nvidia-smi not found" | tee -a $OUT
  print_status "NG" "GPU: nvidia-smi not available - GPU support required for vLLM"
fi

echo -e "\n🚀 4) cgroup / container memory limits (if present)" | tee -a $OUT
cgroup_limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "no cgroup memory limit")
echo "$cgroup_limit" | tee -a $OUT
if [ -f /sys/fs/cgroup/memory.max ]; then
  echo "cgroup v2 memory.max: $(cat /sys/fs/cgroup/memory.max)" | tee -a $OUT
fi

# Check cgroup limits
if [[ "$cgroup_limit" =~ ^[0-9]+$ ]]; then
  # Convert to GB for comparison
  cgroup_limit_gb=$((cgroup_limit / 1024 / 1024 / 1024))
  if [ $cgroup_limit_gb -ge 16 ]; then
    print_status "OK" "cgroup: Memory limit ${cgroup_limit_gb}GB - sufficient"
  else
    print_status "NG" "cgroup: Memory limit ${cgroup_limit_gb}GB - may be insufficient for vLLM"
  fi
else
  print_status "OK" "cgroup: No memory limits detected"
fi

echo -e "\n🚀 5) ulimit (all) and virtual memory limit" | tee -a $OUT
ulimit -a | tee -a $OUT
echo -n "ulimit -v (virtual memory, KB): "; ulimit -v | tee -a $OUT

# Check ulimit values
virtual_mem_limit=$(ulimit -v)
open_files_limit=$(ulimit -n)

if [ "$virtual_mem_limit" = "unlimited" ] || [ $virtual_mem_limit -gt 16777216 ]; then  # 16GB in KB
  vm_status="OK"
  vm_msg="Virtual memory limit: $virtual_mem_limit - sufficient"
else
  vm_status="NG"
  vm_msg="Virtual memory limit: $virtual_mem_limit KB - may be too restrictive"
fi

if [ $open_files_limit -ge 1024 ]; then
  files_status="OK"
  files_msg="Open files limit: $open_files_limit - sufficient"
else
  files_status="NG"
  files_msg="Open files limit: $open_files_limit - may be too low"
fi

print_status "$vm_status" "$vm_msg"
print_status "$files_status" "$files_msg"

echo -e "\n🚀 6) Environment variables of interest" | tee -a $OUT
# list of variables to show (from your chat + common ones)
vars=(
  MALLOC_ARENA_MAX
  NCCL_IB_DISABLE
  NCCL_SOCKET_IFNAME
  PYTORCH_CUDA_ALLOC_CONF
  RANGER_LOAD_DEFAULT_RC
  RAY_ADDRESS
  RAY_HEAD_IP
  RAY_HEAD_NODE
  RAY_WORKER_NODE
  SLURM_JOB_ID
  TORCH_COMPILE_DISABLE
  TRANSFORMERS_NO_TORCHVISION
  VLLM_HOST_IP
  VLLM_USE_RAY_SPMD_WORKER
  VLLM_WORKER_MULTIPROC_METHOD
  CPUS_PER_TASK
  CUDA_CACHE_DISABLE
  CUDA_LAUNCH_BLOCKING
  EXCLUSIVE
  GPUS_PER_NODE
  HEAD_NODE_ID
  JOB_ID
  MAX_MODEL_LEN
  MAX_NUM_SEQS
  MODEL_NAME
  MY_RAY_GPUS
  MY_RAY_HEAD_PORT
  NCCL_DEBUG
  NODE_COMBINATION
  NODELIST
  NUM_GPUS
  NUM_NODES
  OMP_NUM_THREADS
  PARTITION
  PIPELINE_PARALLEL
  RAY_HEAD_NAME
  RAY_HEAD_PORT
  RAY_PORT
  SESSION_FILE
  SESSION_ID
  SLURM_JOBID
  TENSOR_PARALLEL
  TOTAL_GPUS
  VLLM_PORT
)

critical_vars_set=0
total_critical_vars=0

for v in "${vars[@]}"; do
  value="${!v-'<not set>'}"
  printf "%-35s => %s\n" "$v" "$value" | tee -a $OUT
  
  # Check critical variables for vLLM
  case "$v" in
    "CUDA_CACHE_DISABLE"|"TORCH_COMPILE_DISABLE"|"TRANSFORMERS_NO_TORCHVISION")
      total_critical_vars=$((total_critical_vars + 1))
      if [ "$value" != "<not set>" ]; then
        critical_vars_set=$((critical_vars_set + 1))
      fi
      ;;
  esac
done

# Check environment variables status
if [ $critical_vars_set -gt 0 ]; then
  print_status "OK" "Environment: ${critical_vars_set}/${total_critical_vars} critical vLLM variables are set"
else
  print_status "OK" "Environment: No critical issues detected (variables can be set as needed)"
fi

echo -e "\n🚀 7) Processes listening on typical ports (8000/8001) and ray/raylet" | tee -a $OUT
port_output=$(ss -ltnp | egrep '8000|8001' || true)
echo "$port_output" | tee -a $OUT

# Check port status
if [ -n "$port_output" ]; then
  print_status "NG" "Ports: vLLM default ports (8000/8001) are already in use"
else
  print_status "OK" "Ports: vLLM default ports (8000/8001) are available"
fi
echo -e "\n🚀 8) Ray / session sockets (if ray session exists)" | tee -a $OUT

# Check if this is a multi-node setup
is_multinode=false
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    is_multinode=true
    echo "Detected SLURM multi-node setup: ${SLURM_JOB_NUM_NODES} nodes" | tee -a $OUT
elif [ -n "${NUM_NODES:-}" ] && [ "${NUM_NODES}" -gt 1 ]; then
    is_multinode=true
    echo "Detected multi-node setup: ${NUM_NODES} nodes" | tee -a $OUT
elif [ -n "${NODELIST:-}" ] && [[ "${NODELIST}" == *","* ]]; then
    is_multinode=true
    echo "Detected multi-node setup from NODELIST: ${NODELIST}" | tee -a $OUT
else
    echo "Detected single-node setup" | tee -a $OUT
fi

if [ -d /tmp/ray/session_latest ]; then
  echo "Found /tmp/ray/session_latest" | tee -a $OUT
  echo "socket owners (pids):" | tee -a $OUT
  lsof_output=$(lsof -n /tmp/ray/session_latest/sockets/* 2>/dev/null || true)
  echo "$lsof_output" | tee -a $OUT
  
  if [ -n "$lsof_output" ]; then
    print_status "OK" "Ray: Active Ray session detected"
  else
    if [ "$is_multinode" = true ]; then
      print_status "NG" "Ray: Ray session directory exists but no active processes (required for multi-node)"
    else
      print_status "WARNING" "Ray: Ray session directory exists but no active processes (may be stale)"
    fi
  fi
else
  # No Ray session directory found
  if [ "$is_multinode" = true ]; then
    print_status "NG" "Ray: No Ray session found (required for multi-node vLLM setup)"
  else
    print_status "WARNING" "Ray: No Ray session found (single-node vLLM can run without Ray cluster)"
  fi
fi

echo -e "\n🚀 9) Python interpreter / packages (versions)" | tee -a $OUT
python_output=$(python - <<'PY' 2>&1
import sys,importlib,subprocess
print("python:", sys.version.replace('\n',' '))
modules = ['torch','transformers','torchvision','vllm','ray']
results = {}
for m in modules:
    try:
        mod = importlib.import_module(m)
        version = getattr(mod,'__version__', 'unknown')
        print(f"{m}: ok ({version})")
        results[m] = True
    except Exception as e:
        print(f"{m}: import failed: {e}")
        results[m] = False

# Output results for shell parsing
import json
print("PYTHON_CHECK_RESULTS:" + json.dumps(results))
PY
)

echo "$python_output" | tee -a $OUT

# Parse results and check status
if echo "$python_output" | grep -q "PYTHON_CHECK_RESULTS:"; then
  results_line=$(echo "$python_output" | grep "PYTHON_CHECK_RESULTS:" | sed 's/PYTHON_CHECK_RESULTS://')
  
  # Check critical packages
  if echo "$python_output" | grep -q "torch: ok" && echo "$python_output" | grep -q "vllm: ok"; then
    print_status "OK" "Python: Critical packages (torch, vllm) are available"
  elif echo "$python_output" | grep -q "torch: ok"; then
    print_status "NG" "Python: torch available but vllm missing"
  else
    print_status "NG" "Python: Critical packages missing (torch and/or vllm)"
  fi
else
  print_status "NG" "Python: Failed to check package availability"
fi

echo -e "\n🚀 10) Kernel OOM kill log check" | tee -a $OUT
echo "Checking for recent OOM kills in system logs..." | tee -a $OUT
oom_found=false

if command -v dmesg >/dev/null 2>&1; then
  oom_kills=$(dmesg | grep -i "killed process\|out of memory\|oom-kill" | tail -10)
  if [ -n "$oom_kills" ]; then
    echo "Recent OOM kills found:" | tee -a $OUT
    echo "$oom_kills" | tee -a $OUT
    oom_found=true
  else
    echo "No recent OOM kills found in dmesg" | tee -a $OUT
  fi
else
  echo "dmesg command not available" | tee -a $OUT
fi

# Also check journalctl if available
if command -v journalctl >/dev/null 2>&1; then
  journal_oom=$(journalctl --since "24 hours ago" | grep -i "killed process\|out of memory\|oom-kill" | tail -5 2>/dev/null || true)
  if [ -n "$journal_oom" ]; then
    echo "Recent OOM kills in journal:" | tee -a $OUT
    echo "$journal_oom" | tee -a $OUT
    oom_found=true
  fi
fi

# Check OOM status
if [ "$oom_found" = true ]; then
  print_status "NG" "OOM: Recent out-of-memory kills detected - system may be under memory pressure"
else
  print_status "OK" "OOM: No recent out-of-memory kills detected"
fi

echo -e "\n🚀 11) Top memory consuming processes" | tee -a $OUT
echo "Top 20 processes by RSS memory usage:" | tee -a $OUT
ps_output=$(ps aux --sort=-rss | head -n 20)
echo "$ps_output" | tee -a $OUT

# Check memory usage
high_mem_processes=$(echo "$ps_output" | awk 'NR>1 && $4 > 10 {count++} END {print count+0}')
if [ $high_mem_processes -gt 5 ]; then
  print_status "NG" "Memory Usage: ${high_mem_processes} processes using >10% memory - system may be under pressure"
elif [ $high_mem_processes -gt 2 ]; then
  print_status "OK" "Memory Usage: ${high_mem_processes} processes using >10% memory - moderate usage"
else
  print_status "OK" "Memory Usage: Low memory pressure detected"
fi

echo -e "\n🚀 12) GPU topology information" | tee -a $OUT
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU topology matrix:" | tee -a $OUT
  topo_output=$(nvidia-smi topo --matrix 2>&1)
  echo "$topo_output" | tee -a $OUT
  
  # Check topology status
  if echo "$topo_output" | grep -q "GPU0\|GPU1"; then
    print_status "OK" "GPU Topology: Multi-GPU topology information available"
  elif echo "$topo_output" | grep -q "No devices"; then
    print_status "NG" "GPU Topology: No GPU devices found"
  else
    print_status "OK" "GPU Topology: Single GPU or topology info retrieved"
  fi
else
  echo "nvidia-smi not available for topology check" | tee -a $OUT
  print_status "NG" "GPU Topology: nvidia-smi not available"
fi

echo -e "\n🚀 13) Quick import test (will try to import torchvision in a subprocess) - may fail if MemoryError occurs" | tee -a $OUT
import_output=$(python - <<'PY' 2>&1
import subprocess,sys
print("Testing import torchvision in a separate python process (exit code shown).")
cmd = [sys.executable, '-c', 'import time,os,sys\ntry:\n import torchvision\n print(\"torchvision ok\", getattr(torchvision,\"__version__\",\"?\"))\nexcept Exception as e:\n print(\"IMPORT-ERR:\", e);\n sys.exit(1)\n']
rc = subprocess.call(cmd)
print("returncode:", rc)
print("IMPORT_TEST_RC:" + str(rc))
PY
)

echo "$import_output" | tee -a $OUT

# Check import test status
if echo "$import_output" | grep -q "IMPORT_TEST_RC:0"; then
  print_status "OK" "Import Test: torchvision import successful"
elif echo "$import_output" | grep -q "IMPORT_TEST_RC:1"; then
  print_status "NG" "Import Test: torchvision import failed - may indicate memory or dependency issues"
else
  print_status "NG" "Import Test: Unable to determine import test result"
fi

echo -e "\n==== Summary ====" | tee -a $OUT
echo "Check completed. Review the ✅/⚠️/❌ status above for each component." | tee -a $OUT
echo "Items marked as ❌ may need attention before running vLLM successfully." | tee -a $OUT
echo "Items marked as ⚠️ are warnings that may affect performance but won't prevent startup." | tee -a $OUT

echo -e "\n==== Status Summary ====" | tee -a $OUT
echo "✅ OK: ${OK_COUNT}" | tee -a $OUT
echo "⚠️ WARNING: ${WARNING_COUNT}" | tee -a $OUT
echo "❌ NG: ${NG_COUNT}" | tee -a $OUT
echo "Total checks: $((OK_COUNT + WARNING_COUNT + NG_COUNT))" | tee -a $OUT

# Overall assessment
if [ $NG_COUNT -eq 0 ] && [ $WARNING_COUNT -eq 0 ]; then
    echo -e "\n🎉 Overall: All checks passed! System is ready for vLLM." | tee -a $OUT
elif [ $NG_COUNT -eq 0 ]; then
    echo -e "\n👍 Overall: System is ready for vLLM with ${WARNING_COUNT} warning(s)." | tee -a $OUT
else
    echo -e "\n⛔️ Overall: ${NG_COUNT} critical issue(s) found. Please address before running vLLM." | tee -a $OUT
fi

echo -e "\n==== Done ====" | tee -a $OUT

# Display NG and WARNING messages summary if any exist
if [ ${#NG_MESSAGES[@]} -gt 0 ] || [ ${#WARNING_MESSAGES[@]} -gt 0 ]; then
    echo -e "\n🔍 Issues Summary:" | tee -a $OUT
    
    if [ ${#NG_MESSAGES[@]} -gt 0 ]; then
        echo "❌ Critical Issues (${#NG_MESSAGES[@]}):" | tee -a $OUT
        for msg in "${NG_MESSAGES[@]}"; do
            echo "   • $msg" | tee -a $OUT
        done
    fi
    
    if [ ${#WARNING_MESSAGES[@]} -gt 0 ]; then
        echo "⚠️ Warnings (${#WARNING_MESSAGES[@]}):" | tee -a $OUT
        for msg in "${WARNING_MESSAGES[@]}"; do
            echo "   • $msg" | tee -a $OUT
        done
    fi
    echo "" | tee -a $OUT
fi

echo "Saved full output to $OUT"

