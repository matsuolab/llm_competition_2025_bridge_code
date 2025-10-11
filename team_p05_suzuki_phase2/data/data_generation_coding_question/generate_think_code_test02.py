# test01との違いはthinkをpythonコードとして生成する。出したコードは特に検証するバージョン
""""
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


# セマフォ（実体は main() 内で初期化してもよいが、簡単のためモジュールスコープに）
_sem_llm  = None
_sem_exec = None


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
# RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
# ディレクトリを切りたい場合は上記を利用
RUN_TS = ""
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", RUN_TS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成パラメータ（必要最低限）
LIMIT = int(os.getenv("DEFAULT_LIMIT", "1"))
PROBLEMS_PER_SEED = int(os.getenv("PROBLEMS_PER_SEED", "2"))
# NUM_COT_CANDIDATES = int(os.getenv("NUM_COT_CANDIDATES", "5"))
NUM_COT_CANDIDATES = int(os.getenv("NUM_COT_CANDIDATES", "3"))  # まずは3本くらいで様子見
COT_TEMPERATURE = float(os.getenv("COT_TEMPERATURE", "0.6"))

# 出力ファイル
DATASET_JSONL_FILENAME = f"think_is_pythoncode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
DATASET_JSONL_PATH = os.path.join(OUTPUT_DIR, DATASET_JSONL_FILENAME)

# --- mode ---
LOCK_QA_MODE = True  # ← これを True にすると、question/answer固定で think を複数生成
PROBLEMS_PER_SEED = 3 


# 并列上限（API/環境に合わせて調整）
LLM_CONCURRENCY  = int(os.getenv("LLM_CONCURRENCY", "8"))   # 同時に投げる LLM 呼び出し
EXEC_CONCURRENCY = int(os.getenv("EXEC_CONCURRENCY", "16")) # 同時に回すローカル実行
SEED_CONCURRENCY = int(os.getenv("SEED_CONCURRENCY", "2"))  # 同時に処理する seed 個数（必要なら）

STREAM_WRITE = True             # 生成直後に1行ずつ書く
DURABLE_WRITE = False           # Trueならfsyncでディスクに確定（遅くなる）


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

# 安全ラッパー
async def _ainvoke_bounded(chain, payload):
    # LLM呼び出しの同時数を制限
    async with _sem_llm:
        return await chain.ainvoke(payload)

async def _exec_py_async_bounded(code: str, timeout_sec: int = 3) -> str:
    # ローカル実行の同時数を制限
    async with _sem_exec:
        return await asyncio.to_thread(_exec_py_and_get_lastline, code, timeout_sec)



# 4) DeepSeek用の最小LLMクライアントと最小プロンプト
# --- LLM clients (DeepSeek API only) ---
# DeepSeekはOpenAI互換。ハング対策でtimeoutとmax_retriesを明示
llm_solver = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=COT_TEMPERATURE, 
    base_url=BASE_URL,
    timeout=120,          # ← 60秒で打ち切り
    max_retries=1        # ← 無限リトライを禁止
)
llm_generator = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=0.8, 
    base_url=BASE_URL,
    timeout=120,
    max_retries=1
)


# --- prompts ---
fixed_qa_prompt_symbolic = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        # 用途: 記号処理/一般問題の "thinkをPythonコードで" 生成する。答えは渡さず、推論過程をコメントで残す。
        # 期待する出力: Pythonコードのみ（フェンス/文章なし）。段階的な思考コメント + 最後に print(result)。
        # 注意: ハードコード禁止。標準ライブラリのみ。ファイル/ネット/OS/乱数/入力禁止。

        "You are an expert Python 3 programmer and problem solver.\n"
        "Write **pure Python code** that solves the GIVEN PROBLEM **without being provided the answer**.\n"
        "\n"
        "STRICT RULES:\n"
        "• Output ONLY Python code (no markdown fences, no prose outside code).\n"
        "• Standard library only; allow math/decimal/fractions. No files/network/OS, no randomness, no input().\n"
        "• Show your reasoning as **dense step-by-step COMMENTS** (at least 6 meaningful comment lines):\n"
        "  - A short PLAN block (# PLAN: ...)\n"
        "  - Named steps (# Step 1: ..., # Step 2: ...)\n"
        "  - Key formulas/transformations explained in words\n"
        "  - A SANITY-CHECK block explaining why the result scale/units make sense\n"
        "• Compute the final answer **programmatically** from problem data; never hardcode the final answer.\n"
        "• Store the final answer in a variable named `result`.\n"
        "• On the **last line**, print(result).\n"
        "\n"
        "STYLE REQUIREMENTS:\n"
        "• Use small helper functions if it clarifies reasoning (inside the same file).\n"
        "• Prefer Fraction/Decimal for exactness when appropriate; otherwise use math with clear rounding policy.\n"
        "• Deterministic behavior only.\n"
    ),
    HumanMessagePromptTemplate.from_template(
        # 用途: 問題文のみを渡す。最終答えは result に格納し print(result) する。
        "PROBLEM:\n{question}\n\n"
        "Produce Python code that derives the final answer with commented reasoning, assigns it to `result`, "
        "and prints it on the last line."
    ),
])



fixed_qa_prompt_numeric = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        # 用途: 数値解の "thinkをPythonコードで" 生成。計算根拠をコメントに残し、最後に print(result)。
        # 期待する出力: Pythonコードのみ。数値の丸め/誤差方針もコメントで明示。

        "You are an expert Python 3 programmer and problem solver.\n"
        "Write **pure Python code** that computes a **numeric** final answer for the GIVEN PROBLEM.\n"
        "\n"
        "STRICT RULES:\n"
        "• Output ONLY Python code (no markdown/prose).\n"
        "• Standard library only (math/decimal/fractions OK). No I/O, randomness, filesystem/network/OS.\n"
        "• Provide **clear step-by-step COMMENTS** including:\n"
        "  - # PLAN: high-level approach\n"
        "  - # Derivation: formulas used and why\n"
        "  - # SANITY-CHECK: reasonableness of magnitude/units\n"
        "• Compute the final numeric value from inputs; do not hardcode.\n"
        "• Put the final numeric value in `result`.\n"
        "• If rounding is needed, state the policy in comments and apply it consistently (e.g., 1e-9 tolerance).\n"
        "• Finally, print(result) on the last line.\n"
    ),
    HumanMessagePromptTemplate.from_template(
        # 用途: 数値最終解を result に入れ、最後に print(result)。
        "PROBLEM:\n{question}\n\n"
        "Produce Python code that computes the numeric final answer, assigns it to `result`, and prints it."
    ),
])



# --- Fixed-QA 用 chain 定義 ---
fixed_qa_chain_symbolic = fixed_qa_prompt_symbolic | llm_solver | StrOutputParser()
fixed_qa_chain_numeric  = fixed_qa_prompt_numeric  | llm_solver | StrOutputParser()


# 5) CoT実行（安全実行の超ミニマム版）
import tempfile, subprocess, textwrap

def _exec_py_and_get_lastline(py: str, timeout_sec: int = 3) -> str:
    """
    生成された Python コードをローカルで実行し、標準出力の最終行を返す。
    - 危険なモジュール/関数の使用は is_code_safe() でブロック
    - ```python ... ``` のフェンスは自動で剥がす
    - 出力が空だった場合、末尾に print(result) を自動追記してワンチャン再実行（思考コードは壊さない）
    """
    if not is_code_safe(py):
        return "[UNSAFE CODE BLOCKED]"

    import tempfile, subprocess, textwrap, os

    code = (py or "").strip()
    # ```python ... ``` / ``` ... ``` を剥がす
    if code.startswith("```"):
        # 最初の改行で区切って中身を取り出す
        # 例: ```python\n<code>``` → <code>
        first_nl = code.find("\n")
        if first_nl != -1:
            code = code[first_nl + 1:]
        code = code.strip()
        if code.endswith("```"):
            code = code[:-3].rstrip()

    code = textwrap.dedent(code)

    # 1回目の実行
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
        if stdout:
            return stdout.splitlines()[-1].strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[EXEC ERROR] {e.__class__.__name__}"
    finally:
        try:
            os.remove(p)
        except Exception:
            pass

    # 出力なし → 末尾に print(result) を追記して再実行（思考コードは壊さない）
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf2:
            tf2.write(code + "\ntry:\n    print(result)\nexcept Exception:\n    pass\n")
            p2 = tf2.name
        out2 = subprocess.run(
            ["python", "-S", p2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        stdout2 = out2.stdout.strip()
        if stdout2:
            return stdout2.splitlines()[-1].strip()
        return "[NO OUTPUT]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[EXEC ERROR] {e.__class__.__name__}"
    finally:
        try:
            os.remove(p2)
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


async def process_seed_item(
    seed_item: Dict[str, Any],
    gen_per_seed: int = PROBLEMS_PER_SEED,
    save_path: Optional[str] = None  # 指定時は各行でき次第すぐ追記
) -> List[Dict[str, Any]]:
    """
    固定QAモード専用:
      - seed の question / answer を固定
      - think(=Pythonコード) を gen_per_seed 本生成
      - 各 think を実行して answer2 を得て、answer と比較して match を付与
      - data_id は seed の末尾 000 を維持し、生成は 001.. に割当
      - save_path を渡した場合は、生成したレコードを即 JSONL に追記する
    """
    seed_q  = seed_item.get("question", "")
    seed_ans = seed_item.get("answer", "")
    seed_id  = seed_item.get("data_id", "")
    subject  = extract_subject_from_data_id(seed_id)
    qtype    = detect_question_type(seed_q)

    out: List[Dict[str, Any]] = []

    for i in range(1, gen_per_seed + 1):  # _001 から
        new_data_id = increment_data_id(seed_id, i)

        thinks = await gen_think_fixed_qa(seed_q, seed_ans, k=1)
        if not thinks:
            continue

        think_code, answer2 = thinks[0]
        rec = {
            "data_id": new_data_id,
            "question": seed_q,                  # 固定
            "think": think_code,                 # バリエーション
            "answer": seed_ans,                  # 正解（固定）
            "answer2": answer2,                  # 実行結果
            "match": _judge_match(seed_ans, answer2),
            "subject": subject,
            "question_type": qtype,
            "fixed_qa": True,
            "variant_index": i,
        }

        out.append(rec)

        # 行単位で即書き込み（任意。append_jsonl は answer2/match 対応版であること）
        if save_path:
            append_jsonl(rec, save_path)
            print(f"  ✓ wrote {new_data_id} -> {save_path}")

    return out


# 8) JSONL I/O と main（最小）
# １行ごとに追記していくスタイル
def append_jsonl(record: Dict[str, Any], path: str):
    slim = {
        "data_id": record.get("data_id", ""),
        "question": record.get("question", ""),
        "think": record.get("think", ""),
        "answer": record.get("answer", ""),
        "answer2": record.get("answer2", ""),           # ← 追加
        "match": bool(record.get("match", False)),      # ← 追加
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 1行追記
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(slim, ensure_ascii=False) + "\n")
        if DURABLE_WRITE:
            f.flush()
            os.fsync(f.fileno())



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
    - mismatchでもコードは上書きしない（思考コメント保持）
    - 直書き/薄すぎ検出時はその場で再生成（最大 TRIVIAL_RETRY 回）
    """
    is_numeric = _looks_numeric(expected_answer)
    chain = fixed_qa_chain_numeric if is_numeric else fixed_qa_chain_symbolic

    out: List[Tuple[str, str]] = []
    for _ in range(k):
        code = ""
        # 直書き/薄すぎの検知でリトライ
        for _try in range(TRIVIAL_RETRY):
            try:
                code = await chain.ainvoke({"question": question, "answer": expected_answer})
            except Exception as e:
                print("[FIXED-QA] gen error:", repr(e))
                code = ""
            if not _is_trivial_print(code, expected_answer):
                break  # 合格

        ans = _exec_py_and_get_lastline(code, timeout_sec=3)

        # 検証：数値は近似一致／記号は完全一致（※コードは絶対に上書きしない）
        # ここでは match の真偽は返さず、answer2 だけ返す
        out.append((code, ans))
    return out


def _judge_match(expected: str, observed: str, tol: float = 1e-9) -> bool:
    if _looks_numeric(expected):
        return _equal_numeric(observed, expected, tol=tol)
    norm = lambda x: " ".join((x or "").strip().split())
    return norm(observed) == norm(expected)


async def main(limit: int = LIMIT):
    """
    実行エントリ:
      - 入力JSONLをlimit件だけ読む
      - 各seedを固定QAモードで処理（_001〜を都度保存）
    """
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0
    for idx, seed in enumerate(seeds, 1):
        print(f"[LOCAL] Seed #{idx}: data_id={seed.get('data_id')}")
        # ここで「1本できるごとに即書き込む」運用
        items = await process_seed_item(seed, PROBLEMS_PER_SEED, save_path=DATASET_JSONL_PATH)
        total += len(items)
        print(f"  ✓ saved (stream) +{len(items)} (total={total})")

    print("\nDone.")
    print(f"Output: {DATASET_JSONL_PATH}")


# 2) トリビアル検知（答え直書き/薄すぎ/print一発）
TRIVIAL_RETRY = 3  # 直書き/薄すぎ検出時の再生成回数

def _is_trivial_print(code: str, expected: str) -> bool:
    """
    「学習に使えない薄い think」を弾く簡易フィルタ。
    - 期待答えのリテラルを print するだけ
    - 行数が少なすぎ（コメント含めて3行以下）
    - print 一発系
    """
    s = (code or "")
    if not s.strip():
        return True

    # 期待答えがそのままリテラルで含まれている（直書きっぽい）
    try:
        if expected and json.dumps(expected, ensure_ascii=False) in s:
            return True
    except Exception:
        pass

    # 行数チェック（コメント含めても薄すぎるもの）
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) <= 3:
        return True

    # print一発系（最終行だけでなく全体がprintのみのコードも検知）
    if re.fullmatch(r"\s*print\s*\(.+\)\s*", s.strip()):
        return True

    return False




if __name__ == "__main__":
    import argparse
    # ※ 有界並列を使う場合だけ有効化（下のQ&A参照）
    # _init_semaphores()

    parser = argparse.ArgumentParser(description="DeepSeek minimal generator (fixed-QA)")
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
