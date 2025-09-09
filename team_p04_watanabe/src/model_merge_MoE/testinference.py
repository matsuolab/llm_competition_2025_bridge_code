from vllm import LLM, SamplingParams

# モデルのロード
model_path = "LLMcompe-Team-Watanabe/Qwen3-32B-moe-test-nonshared"  # ローカルモデルのパス
llm = LLM(model=model_path, tensor_parallel_size=4, max_model_len=4096,
          trust_remote_code=True,enforce_eager=True)

# サンプリングパラメータの設定
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# プロンプト
prompts = ["こんにちは、今日の天気は"]

# 推論の実行
outputs = llm.generate(prompts, sampling_params)

# 結果の表示
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)