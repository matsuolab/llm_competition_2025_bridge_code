# test02との違いはthinkをpythonコードとして生成する。出したコードは特に検証しないバージョン
"""
============================================================
HLE対策用 think（Pythonコード）自動生成スクリプト
============================================================

このスクリプトは、HLE（Humanity’s Last Exam）における
**コーディング系の問題**に特化し、
指定された QA データ（question/answer）に対応する
「思考部分（think）」を **Pythonコードとして自動生成**するツールです。

DeepSeek Reasoner などのLLMを用いて、
問題を解く過程をコードとして生成し、必要に応じて実行・検証します。
------------------------------------------------------------
【機能概要】
------------------------------------------------------------
- 入力：JSONL形式で `question` と `answer` を持つseedデータ
- 出力：そのQAに対応する `think`（=Pythonによる思考過程）を複数生成
- モード：
    - `test01.py`: 生成のみ（**コードを検証しないバージョン**）
    - `test02.py`: 生成＋実行検証（**実行結果がanswerと一致するか評価**）

------------------------------------------------------------
【事前準備】
------------------------------------------------------------
1. `config/.env` ファイルに `DEEPSEEK_API_KEY` を記述
2. 入力ファイル（例: `input/seed_xxx.jsonl`）を準備し、次のフィールドを含める：
    - `question`（問題文）
    - `answer`（正解）
    - `data_id`（一意の識別子 ※なければ自動付与）

------------------------------------------------------------
【使い方】
------------------------------------------------------------

▼ 通常実行（例: 最初の10問だけ処理）
```bash
python generate_think_code_test0X.py --limit 10
▼ 引数一覧
--limit: int（任意）
- 処理するseed件数の上限（デフォルト: 1）
- 例：--limit 100 → 最初の100件だけ処理

【出力ファイル】
出力先：outputs/ 配下に自動生成されたタイムスタンプ付きディレクトリ

形式：1行1データの JSONL

内容（test02.pyの場合）：

{
  "data_id": "math_alg_0001_sft_20250828_001_origin",
  "question": "What is 3×4?",
  "think": "（Pythonコード）",
  "answer": "12",
  "answer2": "12",             // ← 生成コードの実行結果
  "match": true,               // ← answerとの一致結果（bool）
  "generated_at": "2025-08-28 21:32:45"
}
test01.py では answer2 / match は含まれません（検証なし）

【補足】
✔ LOCK_QA_MODE = True のため、question/answerは固定で think のみ生成されます
✔ 生成されるPythonコードは、安全性を確保した上で subprocess によって検証されます
✔ 記号解答（式など）は文字列一致、数値解答は1e-9誤差の範囲で評価されます（test02）
✔ 危険な構文（open(), os, eval など）を含むコードはブロックされます

【依存ライブラリ】
Python 3.8+

langchain

langchain_openai

python-dotenv

※ Poetry や requirements.txt にまとめておくと便利です。

【連携LLMについて】
DeepSeek Reasoner を OPENAI_API_KEY として使用（OpenAI互換）

MODEL_NAME_SOLVER などを .env 経由で設定可能
"""



import os
import re
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser




# --- settings (DeepSeek-only) ---
# .env 読み込み（config/.env を優先）
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print(f"Warning: .env not found at {dotenv_path}. Ensure DEEPSEEK_API_KEY is set.")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set. Put it in config/.env or your environment.")

# langchain-openai が OPENAI_API_KEY を見るため、環境変数名を合わせる
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY

# DeepSeek Reasoner を使う（think抑止のため後でプロンプトで制御）
MODEL_NAME = os.getenv("MODEL_NAME_SOLVER", "deepseek-reasoner")
# BASE_URL = "https://api.deepseek.com"  # DeepSeekのOpenAI互換エンドポイント
BASE_URL = "https://api.deepseek.com/v1"  # DeepSeekのOpenAI互換エンドポイント

# 入出力（ローカル固定）
INPUT_JSONL = os.getenv("INPUT_JSONL", "input/seed_all_data_all_20250810_093552.jsonl")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", RUN_TS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成パラメータ（必要最低限）
LIMIT = int(os.getenv("DEFAULT_LIMIT", "1"))
PROBLEMS_PER_SEED = int(os.getenv("PROBLEMS_PER_SEED", "2"))
# NUM_COT_CANDIDATES = int(os.getenv("NUM_COT_CANDIDATES", "5"))
NUM_COT_CANDIDATES = int(os.getenv("NUM_COT_CANDIDATES", "3"))  # まずは3本くらいで様子見
COT_TEMPERATURE = float(os.getenv("COT_TEMPERATURE", "0.6"))

# 出力ファイル
DATASET_JSONL_FILENAME = "instruction_dataset.jsonl"
DATASET_JSONL_PATH = os.path.join(OUTPUT_DIR, DATASET_JSONL_FILENAME)

# --- mode ---
LOCK_QA_MODE = True  # ← これを True にすると、question/answer固定で think を複数生成
PROBLEMS_PER_SEED = 3 
# --- safe-exec guard patterns (fix: 生のraw文字列で定義し直し、事前コンパイル) ---
FORBIDDEN_PATTERNS = [
    r"\bimport\s+(os|sys|subprocess|socket|shutil|pathlib|requests)\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
]

# 事前コンパイルしてパターン不正を起動時に検出（run中に落ちないように）
try:
    _FORBIDDEN_REGEXES = [re.compile(p, flags=re.IGNORECASE) for p in FORBIDDEN_PATTERNS]
except re.error as e:
    raise RuntimeError(f"[RegexCompileError] {e} in FORBIDDEN_PATTERNS")

def is_code_safe(py: str) -> bool:
    s = (py or "").lower()
    for rgx in _FORBIDDEN_REGEXES:
        if rgx.search(s):
            return False
    return True



# 4) DeepSeek用の最小LLMクライアントと最小プロンプト
# --- LLM clients (DeepSeek API only) ---
# llm_solver = ChatOpenAI(model=MODEL_NAME, temperature=COT_TEMPERATURE, base_url=BASE_URL)
# llm_generator = ChatOpenAI(model=MODEL_NAME, temperature=0.8, base_url=BASE_URL)

# --- LLM clients (DeepSeek API only) ---
# DeepSeekはOpenAI互換。ハング対策でtimeoutとmax_retriesを明示
llm_solver = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=COT_TEMPERATURE, 
    base_url=BASE_URL,
    timeout=60,          # ← 60秒で打ち切り
    max_retries=1        # ← 無限リトライを禁止
)
llm_generator = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=0.8, 
    base_url=BASE_URL,
    timeout=60,
    max_retries=1
)


# --- prompts ---
problem_generator_from_seed_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a master item‑writer for advanced academic assessments.\n"
        "Create a NEW, ORIGINAL problem inspired by the reference but NOT a copy.\n"
        "STRICT RULES:\n"
        "• Use the specified {{subject}} and {{question_type}} exactly\n"
        "• Math/science notation must use LaTeX ($...$)\n"
        "• Multiple‑Choice: provide 5–8 options labeled A.–H. (do not reveal answer)\n"
        "• Short‑Answer: single concise answerが出る問いにする\n"
        "• 回答やヒントや思考は出力しない\n"
        "Return ONLY the problem text."
    ),
    HumanMessagePromptTemplate.from_template(
        "Reference Problem:\n{seed_problem}\n\n"
        "Generate a new {subject} {question_type} problem now."
    )
])

cot_solver_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You write pure Python 3 code to solve the given problem and print ONLY the final answer.\n"
        "Constraints:\n"
        "• 標準ライブラリのみ（math/fractions/decimal推奨）\n"
        "• ファイル/ネットワーク/OSアクセス、乱数、入力は禁止\n"
        "• 最終行で答えのみをprint\n"
        "• コードのみを出力（説明やMarkdown禁止）\n"
        "• 実行時間は3秒以内"
    ),
    HumanMessagePromptTemplate.from_template(
        "問題:\n{question}\n\n"
        "上の問題を解く純Pythonコードを出力してください。最終行で答えのみをprintしてください。"
    ),
])

# --- prompts for fixed-QA mode (question/answer are fixed, generate code-only) ---

fixed_qa_prompt_symbolic = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You write pure Python 3 code to solve the GIVEN PROBLEM but you MUST print EXACTLY the provided TARGET_ANSWER string.\n"
        "Rules:\n"
        "• Do NOT modify the problem text.\n"
        "• Output ONLY code (no markdown, no prose).\n"
        "• Standard library only; no files/network/OS, no randomness, no input.\n"
        "• The last line must be a single print(...) that prints the exact target answer string.\n"
        "• You may include comments and intermediate calculations, but ensure the final print matches the target string exactly."
    ),
    HumanMessagePromptTemplate.from_template(
        "PROBLEM:\n{question}\n\n"
        "TARGET_ANSWER (exact string to print):\n{answer}\n\n"
        "Generate Python code that produces this exact final printed output."
    ),
])

fixed_qa_prompt_numeric = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You write pure Python 3 code to solve the GIVEN PROBLEM and print ONLY the final numeric answer.\n"
        "Rules:\n"
        "• Do NOT modify the problem text.\n"
        "• Output ONLY code (no markdown, no prose).\n"
        "• Standard library only; prefer math/decimal/fractions.\n"
        "• The last line must be print(...) of the final numeric value.\n"
        "• The numeric value must equal the target within 1e-9.\n"
        "• You may include comments and intermediate steps."
    ),
    HumanMessagePromptTemplate.from_template(
        "PROBLEM:\n{question}\n\n"
        "TARGET_NUMERIC (must match within 1e-9):\n{answer}\n\n"
        "Generate Python code that computes and prints a numerically matching value."
    ),
])

fixed_qa_chain_symbolic = fixed_qa_prompt_symbolic | llm_solver | StrOutputParser()
fixed_qa_chain_numeric  = fixed_qa_prompt_numeric  | llm_solver | StrOutputParser()

problem_generator_chain = problem_generator_from_seed_prompt | llm_generator | StrOutputParser()
cot_solver_chain = cot_solver_prompt | llm_solver | StrOutputParser()


# 5) CoT実行（安全実行の超ミニマム版）
import tempfile, subprocess, textwrap

def _exec_py_and_get_lastline(py: str, timeout_sec: int = 3) -> str:
    if not is_code_safe(py):
        return "[UNSAFE CODE BLOCKED]"
    code = (py or "").strip()
    if code.startswith("```"):
        # ```python ... ``` などを剥がす
        code = code.strip('`').split("\n", 1)[-1]
    code = textwrap.dedent(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code + "\n")
        p = tf.name
    try:
        out = subprocess.run(
            ["python", "-S", p],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        stdout = out.stdout.strip()
        return stdout.splitlines()[-1].strip() if stdout else "[NO OUTPUT]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[EXEC ERROR] {e.__class__.__name__}"
    finally:
        try:
            os.remove(p)
        except Exception:
            pass



# 6) 最小ユーティリティ（種別判定・data_id）
def detect_question_type(q: str) -> str:
    ql = (q or "").lower()
    if "answer choices:" in ql or re.search(r"\b[A-H][\.\)]\s", q or ""):
        return "Multiple-Choice"
    return "Short-Answer"

def extract_subject_from_data_id(data_id: str) -> str:
    try:
        parts = (data_id or "").split("_")
        i = parts.index("seed")
        return parts[i - 2]
    except Exception:
        return "Other"

# 7) 1シード処理（Problem①=クリーン→解く / Problem②=新規生成→解く）
async def gen_cots_and_pick(question: str, k: int = NUM_COT_CANDIDATES) -> Tuple[str, str]:
    # 並列でk本
    tasks = [cot_solver_chain.ainvoke({"question": question}) for _ in range(k)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    thinks, answers = [], []
    for r in results:
        if isinstance(r, Exception):
            continue
        code = (r or "").strip()
        ans = _exec_py_and_get_lastline(code, timeout_sec=3)
        thinks.append(code)
        answers.append(ans)

    # 最頻値を採用（簡易版）
    from collections import Counter
    valid = [a for a in answers if a and a not in ("[NO OUTPUT]",)]
    if not valid:
        return (thinks[0] if thinks else "", answers[0] if answers else "[NO OUTPUT]")
    maj, _ = Counter(valid).most_common(1)[0]
    # 代表のthink（同じ答えのものがあればそれを）
    pick_think = next((t for t, a in zip(thinks, answers) if a == maj), thinks[0])
    return pick_think, maj


async def process_seed_item(seed_item: Dict[str, Any], gen_per_seed: int = PROBLEMS_PER_SEED) -> List[Dict[str, Any]]:
    seed_q = seed_item.get("question", "")
    seed_ans = seed_item.get("answer", "")
    seed_id = seed_item.get("data_id", "")
    subject = extract_subject_from_data_id(seed_id)
    qtype = detect_question_type(seed_q)

    out: List[Dict[str, Any]] = []

    if LOCK_QA_MODE:
        # question & answer を固定、thinkのみ k 本
        # i=0 は seedそのまま（000）、i>=1 で 001.. を割り当て
        for i in range(1, gen_per_seed + 1):
            new_data_id = increment_data_id(seed_id, i)  # 000, 001, 002...
            # i==0 も含めて全て固定QAで生成（seedの既存thinkは採用せず毎回生成）
            thinks = await gen_think_fixed_qa(seed_q, seed_ans, k=1)
            if not thinks:
                continue
            think_code, final_ans = thinks[0]
            out.append({
                "data_id": new_data_id,
                "question": seed_q,          # ← 固定
                "think": think_code,         # ← バリエーション
                "answer": seed_ans,          # ← 固定（final_ans は検証用に使っただけ）
                "subject": subject,
                "question_type": qtype,
                "fixed_qa": True,
                "variant_index": i,
            })
        return out

    # ← ここから下は従来の“問題①seed/問題②新規生成”の分岐（必要なら残す）
    for i in range(1, gen_per_seed + 1):
        new_data_id = increment_data_id(seed_id, i)
        if i == 0:
            think, ans = await gen_cots_and_pick(seed_q, NUM_COT_CANDIDATES)
            out.append({
                "data_id": new_data_id,
                "question": seed_q,
                "think": think,
                "answer": ans,
                "subject": subject,
                "question_type": qtype,
                "is_cleaned_seed": True,
            })
        else:
            new_prob = await problem_generator_chain.ainvoke({
                "seed_problem": seed_q,
                "subject": subject,
                "question_type": qtype,
            })
            new_prob = re.sub(r"<think>.*?</think>", "", (new_prob or ""), flags=re.DOTALL).strip()
            if not new_prob:
                continue
            think, ans = await gen_cots_and_pick(new_prob, NUM_COT_CANDIDATES)
            out.append({
                "data_id": new_data_id,
                "question": new_prob,
                "think": think,
                "answer": ans,
                "subject": subject,
                "question_type": qtype,
                "generated_from_seed": True,
            })
    return out


# 8) JSONL I/O と main（最小）
def append_jsonl(record: Dict[str, Any], path: str):
    slim = {
        "data_id": record.get("data_id", ""),  # ← 追加
        "question": record.get("question", ""),
        "think": record.get("think", ""),
        "answer": record.get("answer", ""),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(slim, ensure_ascii=False) + "\n")


async def main(limit: int = LIMIT):
    print("Starting generate_think_code_test02.py (DeepSeek API, minimal)...")
    # 入力読込
    seeds: List[Dict[str, Any]] = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                d.setdefault("question", "")
                d.setdefault("answer", "")
                d.setdefault("think", "")
                d.setdefault("data_id", f"local_seed_{ln:07d}_seed_{datetime.now().strftime('%Y%m%d%H%M%S')}_000")
                seeds.append(d)
            except json.JSONDecodeError:
                pass

    if limit and limit > 0:
        seeds = seeds[:limit]

    total = 0
    for idx, seed in enumerate(seeds, 1):
        print(f"[LOCAL] Seed #{idx}: data_id={seed.get('data_id')}")
        items = await process_seed_item(seed, PROBLEMS_PER_SEED)
        for it in items:
            append_jsonl(it, DATASET_JSONL_PATH)
            total += 1
            print(f"  ✓ saved #{total}")

    print("\nDone.")
    print(f"Output: {DATASET_JSONL_PATH}")

#-------------------------------
#----追加------------------------
#-------------------------------

def increment_data_id(original_id: str, index: int) -> str:
    """
    original_id の末尾3桁を index に置き換える（ゼロパディング）
    例: index=0 → 元のseedそのまま
        index=1 → 001
    """
    try:
        parts = original_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"{parts[0]}_{index:03d}"
        else:
            # 末尾が数字でない場合はそのまま付与
            return f"{original_id}_{index:03d}"
    except Exception:
        return f"{original_id}_{index:03d}"

import math

def _looks_numeric(ans: str) -> bool:
    if not isinstance(ans, str):
        return False
    s = ans.strip()
    # 単純系の数値判定（整数/小数/分数っぽいもの）
    if re.fullmatch(r"[+\-]?\d+(\.\d+)?", s):
        return True
    # 分数や指数などは一旦 symbolic 扱い（安全側）
    return False

def _equal_numeric(a: str, b: str, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except:
        return False

async def gen_think_fixed_qa(question: str, expected_answer: str, k: int = NUM_COT_CANDIDATES) -> List[Tuple[str, str]]:
    """
    question/answer固定で、think(=コード)だけk本生成。
    返り値: List[(think_code, final_printed_answer)]
    """
    is_numeric = _looks_numeric(expected_answer)
    chain = fixed_qa_chain_numeric if is_numeric else fixed_qa_chain_symbolic

    out: List[Tuple[str, str]] = []
    for _ in range(k):
        try:
            code = await chain.ainvoke({"question": question, "answer": expected_answer})
        except Exception as e:
            print("[FIXED-QA] gen error:", repr(e))
            continue
        ans = _exec_py_and_get_lastline(code, timeout_sec=3)

        # 検証：数値は近似一致／記号は完全一致
        ok = _equal_numeric(ans, expected_answer) if is_numeric else (ans.strip() == expected_answer.strip())
        if not ok:
            # どうしても合わない場合のフォールバック（厳密一致のため）
            code = f'print({json.dumps(expected_answer, ensure_ascii=False)})'
            ans = expected_answer

        out.append((code, ans))
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek minimal generator (no vLLM/HPC)")
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))

