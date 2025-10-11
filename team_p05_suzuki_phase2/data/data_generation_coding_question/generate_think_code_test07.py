# ============================================================
# HLE対策用 code生成スクリプト v07 (think+code合体対応)
# ============================================================
# このスクリプトは JSONL データを読み込み、
# answer が数値のものだけを対象に:
#   - question, answer, think を LLMに渡し code を生成
#   - codeをローカルで実行して answer2 を得る
#   - answer と answer2 を数値同値判定で比較
#   - think, code, connector, think_plus_code を出力に保存
#
# v07: 問題+答え+think → code (＋thinkと合体カラムも保存)
# ============================================================

import os, re, json, asyncio, time, subprocess, tempfile, textwrap
from datetime import datetime
from typing import Dict, Any, List
from decimal import Decimal, InvalidOperation
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- 環境設定 ----------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
dotenv_path = os.path.join(
    repo_root,
    "data_management/data_generation/sft_data_generation/config/.env"
)
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"[INFO] .env loaded from {dotenv_path}")
else:
    print(f"[WARN] .env not found at {dotenv_path}")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set.")
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY

MODEL_NAME = os.getenv("MODEL_NAME_SOLVER", "deepseek-chat")
BASE_URL = "https://api.deepseek.com/v1"

# 入出力
INPUT_JSONL = "./team_suzuki/data_management/data_generation/sft_data_generation/seed/split_by_flag/data_flag_3_4_Math.jsonl"
OUTPUT_DIR = os.path.join(repo_root, "data_management/data_generation/sft_data_generation/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "math_gene_test06_output.jsonl")

# ---------- LLMクライアント ----------
llm_solver = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.3,
    base_url=BASE_URL,
    timeout=120,
    max_retries=1
)

# ---------- プロンプト定義 ----------
code_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert mathematician and Python 3 programmer.\n"
        "Return your output STRICTLY in valid JSON format with exactly one field: 'code'.\n"
        "\n"
        "Rules for 'code':\n"
        "- Provide ONLY pure Python3 code (no markdown, no extra prose).\n"
        "- The code must compute the answer programmatically, not hardcode it.\n"
        "- The code must assign the final answer to variable `result` and then print(result).\n"
        "- Use only standard libraries (math, decimal, fractions are allowed).\n"
        "- If the final answer is an integer, print it as an integer (no decimal point).\n"
    ),
    HumanMessagePromptTemplate.from_template(
        "PROBLEM:\n{question}\n\n"
        "THINK:\n{think}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Generate your response in JSON with only 'code'."
    )
])
code_chain = code_prompt | llm_solver | StrOutputParser()

# ---------- 補助関数 ----------
def is_numeric_answer(ans: str) -> bool:
    return bool(re.fullmatch(r"[+\-]?\d+(\.\d+)?", ans.strip()))

def normalize(s: str) -> str:
    return " ".join((s or "").strip().split())

def numeric_equal(a_str: str, b_str: str) -> bool:
    try:
        a = Decimal(a_str.strip())
        b = Decimal(b_str.strip())
        if a == b: return True
        if a == int(a) and b == int(b) and int(a) == int(b): return True
        return False
    except (InvalidOperation, ValueError):
        return normalize(a_str) == normalize(b_str)

def is_code_safe(py: str) -> bool:
    forbidden = [r"\bimport\s+os\b", r"\beval\s*\(", r"\bexec\s*\("]
    return not any(re.search(pat, py) for pat in forbidden)

def exec_code_and_get_lastline(code: str, timeout_sec: int = 5) -> str:
    if not is_code_safe(code):
        return "[UNSAFE]"
    code = textwrap.dedent(code.strip())
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code + "\n")
        path = tf.name
    try:
        out = subprocess.run(
            ["python3", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        stdout = out.stdout.strip()
        if stdout:
            return stdout.splitlines()[-1]
        else:
            return f"[ERROR {out.stderr.splitlines()[0]}]" if out.stderr else "[NO OUTPUT]"
    except Exception as e:
        return f"[ERROR {type(e).__name__}]"
    finally:
        try: os.remove(path)
        except: pass

def append_jsonl(record: Dict[str, Any], path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------- メイン処理 ----------
async def process_one(seed: Dict[str, Any]) -> Dict[str, Any]:
    q, ans, think = seed.get("question",""), seed.get("answer","").strip(), seed.get("think","")
    data_id = seed.get("data_id","")
    print(f"\n[INFO] Processing data_id={data_id}")

    if not is_numeric_answer(ans):
        print(f"[SKIP] Non-numeric answer → {ans}")
        return None

    # --- LLM呼び出し ---
    print(f"[STEP] LLM call... ({datetime.now().strftime('%H:%M:%S')})")
    out = await code_chain.ainvoke({"question": q, "answer": ans, "think": think})
    try:
        obj = json.loads(out)
        code = obj.get("code","").strip()
    except Exception:
        print("[WARN] Invalid JSON output, skipping.")
        return None

    # --- コード実行 ---
    answer2 = exec_code_and_get_lastline(code)
    match = numeric_equal(ans, answer2)
    print(f"[RESULT] ans={ans} | ans2={answer2} | match={match}")

    # connector と think+code を生成
    connector = "Now let's implement this in Python:"
    think_plus_code = f"{think}\n\n{connector}\n{code}"

    rec = {
        "data_id": data_id,
        "question": q,
        "think": think,
        "code": code,
        "connector": connector,
        "think_plus_code": think_plus_code,
        "answer": ans,
        "answer2": answer2,
        "match": match,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return rec

async def main(limit: int = 50):
    print("[START] generate_think_code_test07.py")

    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "data_id" in d: processed_ids.add(d["data_id"])
                except: continue
        print(f"[INFO] {len(processed_ids)} already processed.")

    seeds: List[Dict[str, Any]] = []
    with open(INPUT_JSONL,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: seeds.append(json.loads(line))
            except: pass
            if len(seeds)>=limit: break
    print(f"[INFO] Loaded {len(seeds)} seeds")

    total=0
    for s in seeds:
        if s.get("data_id") in processed_ids:
            print(f"[SKIP] Already done {s.get('data_id')}")
            continue
        rec = await process_one(s)
        if rec:
            append_jsonl(rec, OUTPUT_PATH)
            total+=1
            print(f"[SAVED] {rec['data_id']} (total={total})")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=50)
    args=parser.parse_args()
    asyncio.run(main(limit=args.limit))
