#!/bin/bash

# =============================================================================
# 11-vwf_check_vllm_api_endpoints.sh
# vLLM API全エンドポイントをチェックするスクリプト
# 
# 使用方法:
#   ./11-vwf_check_vllm_api_endpoints.sh [OPTIONS] [HOST]
#   
# 例:
#   ./11-vwf_check_vllm_api_endpoints.sh                    # デフォルト: http://0.0.0.0:8000
#   ./11-vwf_check_vllm_api_endpoints.sh http://localhost:8000
#   ./11-vwf_check_vllm_api_endpoints.sh -v http://localhost:8000  # verbose mode
# =============================================================================

# デフォルト設定
VERBOSE=false
VLLM_HOST=""

# ヘルプ表示
show_help() {
    echo "vLLM API Endpoint Checker"
    echo ""
    echo "Usage: $0 [OPTIONS] [HOST]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose Show detailed request/response information"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Arguments:"
    echo "  HOST    vLLM server URL (default: http://0.0.0.0:8000)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Check default host"
    echo "  $0 http://localhost:8000             # Check localhost"
    echo "  $0 -v http://localhost:8000          # Verbose mode"
    echo "  $0 --verbose http://192.168.1.100:8000  # Verbose with remote host"
    exit 0
}

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        http://*)
            VLLM_HOST="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# デフォルトホストの設定
if [ -z "$VLLM_HOST" ]; then
    VLLM_HOST="http://0.0.0.0:8000"
fi

TIMEOUT=30
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/vllm_api_check_${TIMESTAMP}.log"

# ログディレクトリの作成
mkdir -p "$LOG_DIR"

# カウンター初期化
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

echo "🚀 Starting vLLM API Endpoint Check" | tee "$LOG_FILE"
echo "📝 Target: $VLLM_HOST" | tee -a "$LOG_FILE"
echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "📄 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
if [ "$VERBOSE" = true ]; then
    echo "🔍 Verbose mode: ENABLED" | tee -a "$LOG_FILE"
else
    echo "🔍 Verbose mode: DISABLED (use -v to enable)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"
echo "ℹ️  Note: If you see WARNING messages about 'labels' field in vLLM logs," | tee -a "$LOG_FILE"
echo "   this is due to internal vLLM implementation where some endpoints" | tee -a "$LOG_FILE"
echo "   internally call /v1/embeddings and pass through unused fields." | tee -a "$LOG_FILE"
echo "   This is not an error in this test script." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# サーバー接続チェック
echo -e "\n🔍 Pre-check: Server Connectivity" | tee -a "$LOG_FILE"
connectivity_check=$(curl -s --connect-timeout 10 -m 10 "$VLLM_HOST/health" 2>/dev/null)
connectivity_code=$?

if [ $connectivity_code -eq 0 ] && [ -n "$connectivity_check" ]; then
    echo "✅ vLLM server is accessible at $VLLM_HOST" | tee -a "$LOG_FILE"
else
    echo "❌ Cannot connect to vLLM server at $VLLM_HOST" | tee -a "$LOG_FILE"
    echo "🔍 Checking alternative addresses..." | tee -a "$LOG_FILE"
    
    # Try localhost
    if curl -s --connect-timeout 10 -m 10 "http://localhost:8000/health" >/dev/null 2>&1; then
        VLLM_HOST="http://localhost:8000"
        echo "✅ Found vLLM server at $VLLM_HOST" | tee -a "$LOG_FILE"
    elif curl -s --connect-timeout 10 -m 10 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        VLLM_HOST="http://127.0.0.1:8000"
        echo "✅ Found vLLM server at $VLLM_HOST" | tee -a "$LOG_FILE"
    else
        echo "❌ No vLLM server found on common addresses" | tee -a "$LOG_FILE"
        echo "💡 Please ensure vLLM server is running on port 8000" | tee -a "$LOG_FILE"
        echo "💡 You can start it with: python -m vllm.entrypoints.api_server ..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "🛑 Aborting API check due to server unavailability" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# テスト結果を記録する関数
log_test_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    local timestamp=$(date '+%H:%M:%S')
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    case "$status" in
        "PASS")
            echo "[$timestamp] ✅ $test_name: PASSED" | tee -a "$LOG_FILE"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            ;;
        "FAIL")
            echo "[$timestamp] ❌ $test_name: FAILED - $details" | tee -a "$LOG_FILE"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            ;;
        "SKIP")
            echo "[$timestamp] ⏭️  $test_name: SKIPPED - $details" | tee -a "$LOG_FILE"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            ;;
    esac
    
    if [ -n "$details" ] && [ "$status" = "PASS" ]; then
        echo "    Details: $details" | tee -a "$LOG_FILE"
    fi
}

# Verbose出力関数
verbose_log() {
    local message="$1"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: $message" | tee -a "$LOG_FILE"
    else
        echo "🔍 VERBOSE: $message" >> "$LOG_FILE"
    fi
}

# HTTP リクエストを実行する関数（verbose対応）
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local content_type="${4:-application/json}"
    
    verbose_log "=== REQUEST DETAILS ==="
    verbose_log "Method: $method"
    verbose_log "Endpoint: $VLLM_HOST$endpoint"
    verbose_log "Content-Type: $content_type"
    verbose_log "Request Data Length: ${#data} characters"
    verbose_log "Request Data: $data"
    
    # Debug: Log the request data with more details
    echo "DEBUG: $method $endpoint" >> "$LOG_FILE"
    echo "DEBUG: Request data length: ${#data}" >> "$LOG_FILE"
    echo "DEBUG: Request data: $data" >> "$LOG_FILE"
    echo "DEBUG: Content-Type: $content_type" >> "$LOG_FILE"
    echo "---" >> "$LOG_FILE"
    
    if [ "$method" = "GET" ]; then
        verbose_log "Executing GET request..."
        local response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
             -H "Accept: application/json" \
             "$VLLM_HOST$endpoint" 2>/dev/null)
    else
        # Use a temporary file to ensure data integrity
        local temp_file=$(mktemp)
        echo "$data" > "$temp_file"
        
        verbose_log "Created temporary file: $temp_file"
        verbose_log "Temporary file content: $(cat "$temp_file")"
        verbose_log "Executing POST request with --data-binary..."
        
        local response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
             -X "$method" \
             -H "Content-Type: $content_type" \
             -H "Accept: application/json" \
             --data-binary "@$temp_file" \
             "$VLLM_HOST$endpoint" 2>/dev/null)
        
        rm -f "$temp_file"
        verbose_log "Temporary file deleted"
    fi
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)
    
    verbose_log "=== RESPONSE DETAILS ==="
    verbose_log "HTTP Status Code: $http_code"
    verbose_log "Response Body Length: ${#body} characters"
    verbose_log "Response Body: $body"
    verbose_log "========================"
    
    echo "$response"
}

# エラーメッセージを抽出する関数
extract_error_message() {
    local body="$1"
    local default_msg="$2"
    
    # Try different error message formats
    local error_msg=""
    
    # Format 1: "detail": "message"
    error_msg=$(echo "$body" | grep -o '"detail":[^,}]*' | sed 's/"detail"://g' | tr -d '"' | head -1)
    
    # Format 2: "message": "message"
    if [ -z "$error_msg" ]; then
        error_msg=$(echo "$body" | grep -o '"message":[^,}]*' | sed 's/"message"://g' | tr -d '"' | head -1)
    fi
    
    # Format 3: "error": "message"
    if [ -z "$error_msg" ]; then
        error_msg=$(echo "$body" | grep -o '"error":[^,}]*' | sed 's/"error"://g' | tr -d '"' | head -1)
    fi
    
    # Use default if nothing found
    if [ -z "$error_msg" ]; then
        error_msg="$default_msg"
    fi
    
    echo "$error_msg"
}

# 1. 基本的なヘルスチェック系エンドポイント
echo -e "\n🔍 1. Health Check Endpoints" | tee -a "$LOG_FILE"

# /health
response=$(make_request "GET" "/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/health" "PASS" "HTTP $http_code - $body"
else
    log_test_result "/health" "FAIL" "HTTP $http_code"
fi

# /ping (GET)
response=$(make_request "GET" "/ping")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/ping (GET)" "PASS" "HTTP $http_code - $body"
else
    log_test_result "/ping (GET)" "FAIL" "HTTP $http_code"
fi

# /ping (POST)
response=$(make_request "POST" "/ping" "{}")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/ping (POST)" "PASS" "HTTP $http_code - $body"
else
    log_test_result "/ping (POST)" "FAIL" "HTTP $http_code"
fi

# 2. 情報取得系エンドポイント
echo -e "\n🔍 2. Information Endpoints" | tee -a "$LOG_FILE"

# /version
response=$(make_request "GET" "/version")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/version" "PASS" "HTTP $http_code - $body"
else
    log_test_result "/version" "FAIL" "HTTP $http_code"
fi

# /v1/models
response=$(make_request "GET" "/v1/models")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    model_info=$(echo "$body" | grep -o '"id":"[^"]*"' | head -1)
    log_test_result "/v1/models" "PASS" "HTTP $http_code - Found model: $model_info"
else
    log_test_result "/v1/models" "FAIL" "HTTP $http_code"
fi

# /load
response=$(make_request "GET" "/load")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/load" "PASS" "HTTP $http_code - Load info retrieved"
else
    log_test_result "/load" "FAIL" "HTTP $http_code"
fi

# /metrics
response=$(make_request "GET" "/metrics")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    metrics_count=$(echo "$body" | grep -c "^[a-zA-Z]" || echo "0")
    log_test_result "/metrics" "PASS" "HTTP $http_code - $metrics_count metrics found"
else
    log_test_result "/metrics" "FAIL" "HTTP $http_code"
fi

# 3. トークン処理エンドポイント
echo -e "\n🔍 3. Token Processing Endpoints" | tee -a "$LOG_FILE"

# /tokenize - Try multiple request formats
echo "    Trying /tokenize with different request formats..." | tee -a "$LOG_FILE"

# Format 1: prompt field
tokenize_data1='{
    "prompt": "Hello, world!",
    "model": "Qwen/Qwen3-235B-A22B"
}'
response=$(make_request "POST" "/tokenize" "$tokenize_data1")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    # Success with format 1
    if echo "$body" | grep -q '"tokens":\['; then
        token_count=$(echo "$body" | grep -o '"tokens":\[[^]]*\]' | grep -o ',' | wc -l)
        if [ "$token_count" -eq 0 ]; then
            if echo "$body" | grep -o '"tokens":\[[^]]*\]' | grep -q '[0-9]'; then
                token_count=1
            fi
        else
            token_count=$((token_count + 1))
        fi
        tokens=$(echo "$body" | grep -o '"tokens":\[[^]]*\]' | head -c 50)
        log_test_result "/tokenize" "PASS" "HTTP $http_code - Generated $token_count tokens: ${tokens}... (format: prompt)"
    elif echo "$body" | grep -q '"token_ids":\['; then
        token_count=$(echo "$body" | grep -o '"token_ids":\[[^]]*\]' | grep -o ',' | wc -l)
        if [ "$token_count" -eq 0 ]; then
            if echo "$body" | grep -o '"token_ids":\[[^]]*\]' | grep -q '[0-9]'; then
                token_count=1
            fi
        else
            token_count=$((token_count + 1))
        fi
        tokens=$(echo "$body" | grep -o '"token_ids":\[[^]]*\]' | head -c 50)
        log_test_result "/tokenize" "PASS" "HTTP $http_code - Generated $token_count tokens: ${tokens}... (format: prompt)"
    else
        log_test_result "/tokenize" "PASS" "HTTP $http_code - Tokenization completed (format: prompt)"
    fi
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ]; then
    # Try format 2: text field
    echo "    Format 1 failed, trying format 2..." | tee -a "$LOG_FILE"
    tokenize_data2='{
        "text": "Hello, world!",
        "model": "Qwen/Qwen3-235B-A22B"
    }'
    response=$(make_request "POST" "/tokenize" "$tokenize_data2")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        if echo "$body" | grep -q '"tokens":\['; then
            tokens=$(echo "$body" | grep -o '"tokens":\[[^]]*\]' | head -c 50)
            log_test_result "/tokenize" "PASS" "HTTP $http_code - Tokenization successful: ${tokens}... (format: text)"
        else
            log_test_result "/tokenize" "PASS" "HTTP $http_code - Tokenization completed (format: text)"
        fi
    else
        # Try format 3: input field
        echo "    Format 2 failed, trying format 3..." | tee -a "$LOG_FILE"
        tokenize_data3='{
            "input": "Hello, world!",
            "model": "Qwen/Qwen3-235B-A22B"
        }'
        response=$(make_request "POST" "/tokenize" "$tokenize_data3")
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n -1)
        
        if [ "$http_code" = "200" ]; then
            if echo "$body" | grep -q '"tokens":\['; then
                tokens=$(echo "$body" | grep -o '"tokens":\[[^]]*\]' | head -c 50)
                log_test_result "/tokenize" "PASS" "HTTP $http_code - Tokenization successful: ${tokens}... (format: input)"
            else
                log_test_result "/tokenize" "PASS" "HTTP $http_code - Tokenization completed (format: input)"
            fi
        else
            error_msg=$(extract_error_message "$body" "All request formats failed")
            log_test_result "/tokenize" "FAIL" "HTTP $http_code - $error_msg"
        fi
    fi
else
    log_test_result "/tokenize" "FAIL" "HTTP $http_code - $body"
fi

# /detokenize
detokenize_data='{"tokens": [9906, 11, 1917, 0], "model": "Qwen/Qwen3-235B-A22B"}'
response=$(make_request "POST" "/detokenize" "$detokenize_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    text=$(echo "$body" | grep -o '"text":"[^"]*"')
    log_test_result "/detokenize" "PASS" "HTTP $http_code - Decoded: $text"
else
    log_test_result "/detokenize" "FAIL" "HTTP $http_code - $body"
fi

# 4. 主要な生成エンドポイント
echo -e "\n🔍 4. Text Generation Endpoints" | tee -a "$LOG_FILE"

# /v1/completions
completion_data='{
    "model": "Qwen/Qwen3-235B-A22B",
    "prompt": "The capital of Japan is",
    "max_tokens": 10,
    "temperature": 0.1
}'
response=$(make_request "POST" "/v1/completions" "$completion_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    generated_text=$(echo "$body" | grep -o '"text":"[^"]*"' | head -1)
    log_test_result "/v1/completions" "PASS" "HTTP $http_code - Generated: $generated_text"
else
    log_test_result "/v1/completions" "FAIL" "HTTP $http_code - $body"
fi

# /v1/chat/completions
chat_data='{
    "model": "Qwen/Qwen3-235B-A22B",
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 20,
    "temperature": 0.1
}'
response=$(make_request "POST" "/v1/chat/completions" "$chat_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    chat_response=$(echo "$body" | grep -o '"content":"[^"]*"' | head -1)
    log_test_result "/v1/chat/completions" "PASS" "HTTP $http_code - Response: $chat_response"
else
    log_test_result "/v1/chat/completions" "FAIL" "HTTP $http_code - $body"
fi

# 5. 埋め込み・分類系エンドポイント
echo -e "\n🔍 5. Embedding & Classification Endpoints" | tee -a "$LOG_FILE"

# Debug: Check all variables before starting
verbose_log "=== VARIABLE STATE CHECK ==="
verbose_log "All variables containing 'labels': $(set | grep -i labels || echo 'None found')"
verbose_log "All variables containing 'classify': $(set | grep -i classify || echo 'None found')"
verbose_log "All variables containing 'embedding': $(set | grep -i embedding || echo 'None found')"
verbose_log "=========================="

# /v1/embeddings
echo "    Testing /v1/embeddings endpoint..." | tee -a "$LOG_FILE"

verbose_log "=== /v1/embeddings ENDPOINT TEST ==="

# Create a completely isolated subprocess to send the request
EMBEDDINGS_JSON='{"model": "Qwen/Qwen3-235B-A22B", "input": ["Hello world"]}'
verbose_log "Embeddings JSON: $EMBEDDINGS_JSON"

if [ "$VERBOSE" = true ]; then
    echo "🔍 VERBOSE: Sending request to $VLLM_HOST/v1/embeddings"
    echo "🔍 VERBOSE: Using isolated subprocess"
    echo "🔍 VERBOSE: Request time: $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Use a completely isolated approach with printf to avoid any variable contamination
response=$(printf '%s' "$EMBEDDINGS_JSON" | curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Accept: application/json" \
           --data-binary @- \
           "$VLLM_HOST/v1/embeddings" 2>/dev/null)

verbose_log "Raw curl response: $response"

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

verbose_log "Parsed HTTP code: $http_code"
verbose_log "Parsed response body: $body"

if [ "$http_code" = "200" ]; then
    embedding_dim=$(echo "$body" | grep -o '"embedding":\[[^]]*\]' | grep -o ',' | wc -l)
    embedding_dim=$((embedding_dim + 1))
    log_test_result "/v1/embeddings" "PASS" "HTTP $http_code - Embedding dimension: $embedding_dim"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(extract_error_message "$body" "Model may not support embeddings")
    log_test_result "/v1/embeddings" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/v1/embeddings" "FAIL" "HTTP $http_code - $body"
fi

verbose_log "=== /v1/embeddings TEST COMPLETED ==="

# /classify
verbose_log "=== /classify ENDPOINT TEST ==="

# Test without labels field first to see if it reduces warnings
CLASSIFY_TEMP=$(mktemp)
cat > "$CLASSIFY_TEMP" << 'EOF'
{"model": "Qwen/Qwen3-235B-A22B", "input": "This is a positive review"}
EOF

verbose_log "Created temporary file for classify (without labels): $CLASSIFY_TEMP"
verbose_log "Classify temporary file content: $(cat "$CLASSIFY_TEMP")"

echo "    Testing /classify endpoint (without labels)..." | tee -a "$LOG_FILE"
echo "    Request data: $(cat "$CLASSIFY_TEMP")" | tee -a "$LOG_FILE"

if [ "$VERBOSE" = true ]; then
    echo "🔍 VERBOSE: Sending request to $VLLM_HOST/classify"
    echo "🔍 VERBOSE: Using curl with --data-binary from temporary file"
    echo "🔍 VERBOSE: Request time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "🔍 VERBOSE: Testing WITHOUT labels field to avoid internal vLLM warning"
fi

response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Accept: application/json" \
           --data-binary "@$CLASSIFY_TEMP" \
           "$VLLM_HOST/classify" 2>/dev/null)

verbose_log "Raw classify curl response (without labels): $response"

rm -f "$CLASSIFY_TEMP"
verbose_log "Classify temporary file $CLASSIFY_TEMP deleted"

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

verbose_log "Parsed classify HTTP code: $http_code"
verbose_log "Parsed classify response body: $body"

if [ "$http_code" = "200" ]; then
    classification=$(echo "$body" | grep -o '"label":"[^"]*"')
    log_test_result "/classify" "PASS" "HTTP $http_code - Classification: $classification (tested without labels field)"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(extract_error_message "$body" "Model may not support classification")
    log_test_result "/classify" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/classify" "FAIL" "HTTP $http_code - $body"
fi

verbose_log "=== /classify TEST COMPLETED ==="

# 6. スコアリング・ランキング系エンドポイント
echo -e "\n🔍 6. Scoring & Ranking Endpoints" | tee -a "$LOG_FILE"

# /score
score_data='{
    "model": "Qwen/Qwen3-235B-A22B",
    "prompt": "The weather is nice today",
    "text": "weather"
}'
response=$(make_request "POST" "/score" "$score_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    score=$(echo "$body" | grep -o '"score":[0-9.]*')
    log_test_result "/score" "PASS" "HTTP $http_code - Score: $score"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support scoring")
    log_test_result "/score" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/score" "FAIL" "HTTP $http_code - $body"
fi

# /v1/score
response=$(make_request "POST" "/v1/score" "$score_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    score=$(echo "$body" | grep -o '"score":[0-9.]*')
    log_test_result "/v1/score" "PASS" "HTTP $http_code - Score: $score"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support scoring")
    log_test_result "/v1/score" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/v1/score" "FAIL" "HTTP $http_code - $body"
fi

# /rerank
rerank_data='{
    "model": "Qwen/Qwen3-235B-A22B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of AI",
        "The weather is sunny today",
        "Deep learning uses neural networks"
    ]
}'
response=$(make_request "POST" "/rerank" "$rerank_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/rerank" "PASS" "HTTP $http_code - Reranking completed"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support reranking")
    log_test_result "/rerank" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/rerank" "FAIL" "HTTP $http_code - $body"
fi

# /v1/rerank
response=$(make_request "POST" "/v1/rerank" "$rerank_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/v1/rerank" "PASS" "HTTP $http_code - Reranking completed"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support reranking")
    log_test_result "/v1/rerank" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/v1/rerank" "FAIL" "HTTP $http_code - $body"
fi

# /v2/rerank
response=$(make_request "POST" "/v2/rerank" "$rerank_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/v2/rerank" "PASS" "HTTP $http_code - Reranking completed"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support reranking")
    log_test_result "/v2/rerank" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/v2/rerank" "FAIL" "HTTP $http_code - $body"
fi

# 7. 音声処理エンドポイント（スキップ - ファイルアップロードが必要）
echo -e "\n🔍 7. Audio Processing Endpoints" | tee -a "$LOG_FILE"
log_test_result "/v1/audio/transcriptions" "SKIP" "Requires audio file upload"
log_test_result "/v1/audio/translations" "SKIP" "Requires audio file upload"

# 8. その他のエンドポイント
echo -e "\n🔍 8. Other Endpoints" | tee -a "$LOG_FILE"

# /pooling
pooling_data='{
    "model": "Qwen/Qwen3-235B-A22B",
    "input": "Hello world"
}'
response=$(make_request "POST" "/pooling" "$pooling_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/pooling" "PASS" "HTTP $http_code - Pooling completed"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ] || [ "$http_code" = "501" ]; then
    error_msg=$(echo "$body" | grep -o '"detail":"[^"]*"' || echo "Model may not support pooling")
    log_test_result "/pooling" "SKIP" "$error_msg (HTTP $http_code)"
else
    log_test_result "/pooling" "FAIL" "HTTP $http_code - $body"
fi

# /invocations (SageMaker compatible)
invocation_data='{
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.1
}'
response=$(make_request "POST" "/invocations" "$invocation_data")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    generated_text=$(echo "$body" | grep -o '"text":"[^"]*"' | head -1)
    log_test_result "/invocations" "PASS" "HTTP $http_code - SageMaker invocation completed: $generated_text"
elif [ "$http_code" = "400" ] || [ "$http_code" = "422" ]; then
    log_test_result "/invocations" "SKIP" "SageMaker format may not be supported (HTTP $http_code)"
else
    log_test_result "/invocations" "FAIL" "HTTP $http_code - $body"
fi

# 9. ドキュメント系エンドポイント
echo -e "\n🔍 9. Documentation Endpoints" | tee -a "$LOG_FILE"

# /openapi.json
response=$(make_request "GET" "/openapi.json")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    log_test_result "/openapi.json" "PASS" "HTTP $http_code - OpenAPI spec retrieved"
else
    log_test_result "/openapi.json" "FAIL" "HTTP $http_code"
fi

# /docs (HTML response expected)
response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
           -H "Accept: text/html" \
           "$VLLM_HOST/docs" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)

if [ "$http_code" = "200" ]; then
    log_test_result "/docs" "PASS" "HTTP $http_code - Documentation page accessible"
else
    log_test_result "/docs" "FAIL" "HTTP $http_code"
fi

# /redoc (HTML response expected)
response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" -m "$TIMEOUT" \
           -H "Accept: text/html" \
           "$VLLM_HOST/redoc" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)

if [ "$http_code" = "200" ]; then
    log_test_result "/redoc" "PASS" "HTTP $http_code - ReDoc page accessible"
else
    log_test_result "/redoc" "FAIL" "HTTP $http_code"
fi

# 最終結果サマリー
echo -e "\n========================================" | tee -a "$LOG_FILE"
echo "🏁 API Endpoint Check Summary" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "📊 Total Tests: $TOTAL_TESTS" | tee -a "$LOG_FILE"
echo "✅ Passed: $PASSED_TESTS" | tee -a "$LOG_FILE"
echo "❌ Failed: $FAILED_TESTS" | tee -a "$LOG_FILE"
echo "⏭️  Skipped: $SKIPPED_TESTS" | tee -a "$LOG_FILE"

# 成功率計算
if [ $TOTAL_TESTS -gt 0 ]; then
    success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    echo "📈 Success Rate: ${success_rate}%" | tee -a "$LOG_FILE"
fi

# 全体評価
if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 Overall Status: ALL CORE ENDPOINTS WORKING" | tee -a "$LOG_FILE"
elif [ $FAILED_TESTS -le 2 ]; then
    echo "👍 Overall Status: MOSTLY WORKING (minor issues)" | tee -a "$LOG_FILE"
else
    echo "⚠️  Overall Status: MULTIPLE ISSUES DETECTED" | tee -a "$LOG_FILE"
fi

echo -e "\n📄 Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "✅ API check completed at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
