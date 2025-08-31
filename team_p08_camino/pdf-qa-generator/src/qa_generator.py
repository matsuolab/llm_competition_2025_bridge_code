# src/qa_generator.py
import json
import traceback
from openai import OpenAI
from typing import List, Dict
from config.settings import Settings

class QAGenerator:
    def __init__(self):
        self.settings = Settings()
        
        # 接続先をローカルで起動したvLLMのAPIサーバーに変更
        self.client = OpenAI(
            base_url="http://127.0.0.1:18888/v1", # 変更点
            api_key="vllm",                     # APIキーは不要なためダミー文字列
        )

        self.qa_generation_prompt = """
You are an expert in creating high-quality, graduate-level benchmark questions for a highly capable AI model. Your task is to generate a question and its corresponding answer based on the provided text.

The QA pair must meet the following strict criteria:
1.  **Source Independence**: The question must be self-contained and answerable without needing external documents. All necessary context, definitions, and data (e.g., specific coefficient values, standard errors, and baseline numbers) must be explicitly included within the question itself.
2.  **No Chain of Thought**: Do not include any step-by-step reasoning or thought processes. The answer should be a direct, final result.
3.  **Correctness**: The final answer must be factually accurate and directly address the question.

**Provided Text:**
{context}

**Your Output should be a single, valid JSON object with the following structure:**
```json
{{
  "qa_pairs": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ]
}}
"""

    def generate_qa_pairs(self, content_list: List[Dict]) -> list:
        try:
            full_text = "\n".join([item["content"] for item in content_list if item["type"] in ["text", "ocr"]])
            if not full_text.strip():
                print("抽出されたテキストが空のため、QA生成をスキップします。")
                return []
            
            messages = [{"role": "user", "content": self.qa_generation_prompt.format(context=full_text)}]
            
            res = self.client.chat.completions.create(
                model=self.settings.PRIMARY_MODEL,
                max_tokens=4096,
                messages=messages,
                response_format={"type": "json_object"},
            )

            json_text = res.choices[0].message.content.strip()
            if not json_text:
                print("モデルからのレスポンスが空です。")
                return []

            try:
                json_start = json_text.find('{')
                json_end = json_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    clean_json_text = json_text[json_start:json_end]
                    qa_pairs_data = json.loads(clean_json_text)
                    return qa_pairs_data.get('qa_pairs', [])
                else:
                    raise json.JSONDecodeError("JSON object not found in response.", json_text, 0)
            except json.JSONDecodeError as json_err:
                print(f"❌ JSONデコードエラー: {json_err}")
                print(f"生のレスポンス (先頭500文字): {json_text[:500]}")
                return []
        except Exception as e:
            print(f"❌ QA生成中に予期せぬエラーが発生しました: {e}")
            print(f"トレースバック:\n{traceback.format_exc()}")
            return []