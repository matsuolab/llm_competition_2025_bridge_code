# src/quality_checker.py
import json
import traceback
from openai import OpenAI
from config.settings import Settings
from typing import List, Dict, Any

class QualityChecker:
    def __init__(self):
        self.settings = Settings()
        
        # 接続先をローカルで起動したvLLMのAPIサーバーに変更
        self.client = OpenAI(
            base_url="http://127.0.0.1:18888/v1", # 変更点
            api_key="vllm", # APIキーは不要なためダミー文字列
        )
        
        self.validation_prompt = """
You are a meticulous and highly skilled quality assurance specialist for an educational content platform. Your task is to rigorously evaluate a single question and answer pair (QA Pair) to determine its suitability for a graduate-level benchmark exam.

The QA Pair are provided below. Your primary goal is to verify the following criteria:

1.  **Source Independence**: Can a well-prepared human examinee answer the question without needing any external documents? The question must be self-contained.
2.  **Internal Consistency**: Are the question and its corresponding answer internally consistent and logically sound?
3.  **No Chain of Thought**: Does the answer refrain from including any step-by-step reasoning, intermediate calculations, or thought processes, presenting only a direct, final result?

**Input QA Pair to Evaluate:**
<qa_pair>
Question: {question}
Answer: {answer}
</qa_pair>

Analyze the QA pair and determine if it meets all three criteria. If it fails, provide specific, constructive feedback on how to improve it.

Return ONLY a single, valid JSON object that starts with '{{' and ends with '}}'. The JSON object must have the following structure:
{{
  "is_valid": boolean,
  "reason": "string"
}}
"""

    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated = []
        if not qa_pairs:
            return validated

        for qa in qa_pairs:
            try:
                prompt_text = self.validation_prompt.format(
                    question=qa['question'],
                    answer=qa['answer']
                )
                
                res = self.client.chat.completions.create(
                    model=self.settings.VERIFICATION_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt_text}],
                    response_format={"type": "json_object"},
                )

                json_text = res.choices[0].message.content.strip()

                json_start = json_text.find('{')
                json_end = json_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    clean_json_text = json_text[json_start:json_end]
                    validation_result = json.loads(clean_json_text)
                else:
                    raise json.JSONDecodeError("JSON object not found in response.", json_text, 0)

                if validation_result.get("is_valid", False):
                    validated.append(qa)
                else:
                    print(f"⚠️ 無効なQAペアが見つかりました: {validation_result.get('reason', '理由なし')}")
            except (json.JSONDecodeError, KeyError) as e:
                error_response_text = 'N/A'
                if 'json_text' in locals():
                    error_response_text = json_text[:100]
                print(f"⚠️ JSONパースエラーまたはキーエラー: {e} - 無効なJSON文字列: '{error_response_text}...'")
            except Exception as e:
                print(f"❌ 検証中に予期せぬエラーが発生しました: {e}")
                print(f"トレースバック:\n{traceback.format_exc()}")
        return validated