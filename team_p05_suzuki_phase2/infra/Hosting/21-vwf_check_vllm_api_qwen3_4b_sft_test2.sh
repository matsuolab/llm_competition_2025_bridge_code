#!/bin/bash

# =============================================================================
# 12-vwf_check_vllm_api_qwen3_4b_sft_test2.sh
# Qwen3-4B-SFT-TEST2モデル用のvLLM APIエンドポイントチェックスクリプト
# 
# 使用方法:
#   ./12-vwf_check_vllm_api_qwen3_4b_sft_test2.sh [OPTIONS] [HOST]
#   
# 例:
#   ./12-vwf_check_vllm_api_qwen3_4b_sft_test2.sh                    # デフォルト: http://localhost:8000
#   ./12-vwf_check_vllm_api_qwen3_4b_sft_test2.sh -v                # verbose mode
# =============================================================================

# デフォルト設定
VERBOSE=false
VLLM_HOST=""

# ヘルプ表示
show_help() {
    echo "vLLM API Endpoint Checker for Qwen3-4B-SFT-TEST2"
    echo ""
    echo "Usage: $0 [OPTIONS] [HOST]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose Show detailed request/response information"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Arguments:"
    echo "  HOST    vLLM server URL (default: http://localhost:8000)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Check default host (port 8000)"
    echo "  $0 http://localhost:8000             # Check localhost on port 8000"
    echo "  $0 -v http://localhost:8000          # Verbose mode"
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

# デフォルトホストの設定（Qwen3-4B-SFT-TEST2は8000ポートを使用）
if [ -z "$VLLM_HOST" ]; then
    VLLM_HOST="http://localhost:8000"
fi

echo "🚀 Starting vLLM API Endpoint Check for Qwen3-4B-SFT-TEST2"
echo "📝 Target: $VLLM_HOST"
echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$VERBOSE" = true ]; then
    echo "🔍 Verbose mode: ENABLED"
else
    echo "🔍 Verbose mode: DISABLED (use -v to enable)"
fi
echo "=========================================="

# 基本的な接続テスト
echo ""
echo "🔍 Testing basic connectivity..."
if curl -s --connect-timeout 5 "$VLLM_HOST/health" >/dev/null 2>&1; then
    echo "✅ Server is accessible at $VLLM_HOST"
else
    echo "❌ Cannot connect to server at $VLLM_HOST"
    echo "💡 Please ensure the Qwen3-4B-SFT-TEST2 vLLM server is running on port 8000"
    exit 1
fi

# モデル情報の確認
echo ""
echo "🔍 Checking model information..."
model_info=$(curl -s "$VLLM_HOST/v1/models" 2>/dev/null)
if echo "$model_info" | grep -q "Qwen3-4B-SFT-TEST2"; then
    echo "✅ Qwen3-4B-SFT-TEST2 model detected"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Model info: $model_info"
    fi
else
    echo "⚠️  Model name may differ from expected 'Qwen3-4B-SFT-TEST2'"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Available models: $model_info"
    fi
fi

# 簡単なテキスト生成テスト
echo ""
echo "🔍 Testing text generation..."
generation_test='{
    "model": "Qwen3-4B-SFT-TEST2",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
}'

if [ "$VERBOSE" = true ]; then
    echo "🔍 VERBOSE: Sending generation request..."
    echo "🔍 VERBOSE: Request: $generation_test"
fi

response=$(curl -s -X POST "$VLLM_HOST/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$generation_test" 2>/dev/null)

if echo "$response" | grep -q '"text"'; then
    generated_text=$(echo "$response" | grep -o '"text":"[^"]*"' | head -1)
    echo "✅ Text generation successful: $generated_text"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Full response: $response"
    fi
else
    echo "❌ Text generation failed"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Error response: $response"
    fi
fi

# チャット形式のテスト
echo ""
echo "🔍 Testing chat completions..."
chat_test='{
    "model": "Qwen3-4B-SFT-TEST2",
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 30,
    "temperature": 0.1
}'

if [ "$VERBOSE" = true ]; then
    echo "🔍 VERBOSE: Sending chat request..."
    echo "🔍 VERBOSE: Request: $chat_test"
fi

chat_response=$(curl -s -X POST "$VLLM_HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$chat_test" 2>/dev/null)

if echo "$chat_response" | grep -q '"content"'; then
    chat_content=$(echo "$chat_response" | grep -o '"content":"[^"]*"' | head -1)
    echo "✅ Chat completion successful: $chat_content"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Full chat response: $chat_response"
    fi
else
    echo "❌ Chat completion failed"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Chat error response: $chat_response"
    fi
fi

# 推論機能のテスト（Qwen3の<think>タグ機能）
echo ""
echo "🔍 Testing reasoning capabilities (think tags)..."
reasoning_test='{
    "model": "Qwen3-4B-SFT-TEST2",
    "messages": [
        {"role": "user", "content": "Solve this step by step: If a train travels 60 km/h for 2 hours, how far does it go?"}
    ],
    "max_tokens": 200,
    "temperature": 0.1
}'

if [ "$VERBOSE" = true ]; then
    echo "🔍 VERBOSE: Sending reasoning request..."
    echo "🔍 VERBOSE: Request: $reasoning_test"
fi

reasoning_response=$(curl -s -X POST "$VLLM_HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$reasoning_test" 2>/dev/null)

if echo "$reasoning_response" | grep -q '"content"'; then
    reasoning_content=$(echo "$reasoning_response" | grep -o '"content":"[^"]*"' | head -1)
    echo "✅ Reasoning test successful: $reasoning_content"
    if echo "$reasoning_content" | grep -q "think\|Think\|120"; then
        echo "🧠 Model appears to be using reasoning capabilities"
    fi
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Full reasoning response: $reasoning_response"
    fi
else
    echo "❌ Reasoning test failed"
    if [ "$VERBOSE" = true ]; then
        echo "🔍 VERBOSE: Reasoning error response: $reasoning_response"
    fi
fi

echo ""
echo "=========================================="
echo "🏁 Qwen3-4B-SFT-TEST2 API Check Complete"
echo "=========================================="
echo "📊 Summary:"
echo "   🌐 Server: $VLLM_HOST"
echo "   🤖 Model: Qwen3-4B-SFT-TEST2"
echo "   ⏰ Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "💡 For comprehensive API testing, use:"
echo "   ./11-vwf_check_vllm_api_endpoints.sh $VLLM_HOST"
echo ""
echo "✅ Basic functionality check completed!"
