import re
import argparse
from typing import Optional, Tuple, List
from datasets import load_dataset, Dataset, DatasetDict, Features, Value

FINAL_PATTERNS: List[re.Pattern] = [
    # 既に付いている形式
    re.compile(r"####\s*answer\s*(?P<ans>.+)", re.IGNORECASE | re.DOTALL),
    # よくある表現
    re.compile(r"(?:final\s*answer|answer)\s*[:：]\s*(?P<ans>.+)", re.IGNORECASE | re.DOTALL),
    # TeX での最終解
    re.compile(r"\\boxed\{(?P<ans>[^}]+)\}"),
    # 文末パターン（「よって/Therefore/Thus」などで締めて最後の行）
    re.compile(r"(?:Therefore|Thus|Hence|よって|以上より).*(?:is|=|は)\s*(?P<ans>[^.\n]+)", re.IGNORECASE),
]

THINK_BLOCK_RE = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL | re.IGNORECASE)

def safe_load(args):
    """
    a-m-team/AM-DeepSeek-R1-Distilled-1.4M の実ファイル名を直指定して読み込む。
    subset(name) -> ファイル名をマッピング。
    """
    # まず普通に試す（成功する環境もある）
    try:
        return load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    except Exception as e:
        print(f"[safe_load] Fallback due to: {e}")

    # subset -> 実ファイル名
    name2file = {
        "am_0.5M": "am_0.5M.jsonl.zst",
        "am_0.9M": "am_0.9M.jsonl.zst",
        "am_0.9M_sample_1k": "am_0.9M_sample_1k.jsonl",
    }
    if args.subset not in name2file:
        raise RuntimeError(f"Unknown subset {args.subset}. Expected one of {list(name2file)}")

    uri = f"hf://datasets/{args.dataset}/{name2file[args.subset]}"
    print(f"[safe_load] Loading via JSON data_files: {uri}")

    # NOTE: zstは拡張子で自動認識される。streaming=True で型揺れを無視して読む
    ds_stream = load_dataset(
        "json",
        data_files=uri,
        split="train",
        streaming=True,
    )

    # ストリームをPythonリストへ（必要ならバッチ書き出しに変えてもOK）
    rows = [ex for ex in ds_stream]
    if not rows:
        raise RuntimeError(f"No rows from {uri}")

    return Dataset.from_list(rows)

def extract_question_answer_from_messages(messages) -> Tuple[str, str]:
    """
    messages: [{"role": "user"/"system"/"assistant", "content": "..."}]
    - question: 最後の user の content（なければ最初の user）
    - assistant: 最初の assistant（なければ空）
    """
    user_msgs = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
    asst_msgs = [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]

    question = user_msgs[-1] if user_msgs else (user_msgs[0] if user_msgs else "")
    assistant_text = asst_msgs[0] if asst_msgs else ""
    return question, assistant_text


def extract_question_answer_from_flat(example) -> Tuple[str, str]:
    """
    フラット構造（prompt/response, instruction/output, input/output 等）を想定
    """
    # 代表的キーにフォールバック
    q = example.get("prompt") or example.get("instruction") or example.get("question") or example.get("input") or ""
    a = example.get("response") or example.get("output") or example.get("answer") or ""
    return str(q), str(a)

def _extract_boxed_all(text: str) -> List[str]:
    """
    \boxed{...} をネスト対応で全件抽出（内容のみを返す）。
    例: \\boxed{\\dfrac{\\sqrt{2}}{\\cos \\theta}} -> "\\dfrac{\\sqrt{2}}{\\cos \\theta}"
    """
    out = []
    i = 0
    needle = r"\boxed{"
    nlen = len(needle)
    L = len(text)
    while True:
        i = text.find(needle, i)
        if i == -1:
            break
        j = i + nlen  # 最初の '{' の直後
        depth = 1
        k = j
        while k < L and depth > 0:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth == 0:
            content = text[j:k-1].strip()
            if content:
                out.append(content)
            i = k  # 続きを探す
        else:
            # 閉じ括弧に到達できなかったら打ち切り
            break
    # 重複除去（順序保持）
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def find_final_answers(text: str) -> List[str]:
    # 1) ネスト対応の \boxed{...} を最優先で全回収
    finals = _extract_boxed_all(text)
    if finals:
        return finals

    # 2) 既存パターンにフォールバック（複数ヒットも許容して、順次追加）
    out = []
    seen = set()
    for pat in FINAL_PATTERNS:
        for m in pat.finditer(text):
            ans = m.group("ans").strip()
            # 1行目を優先
            ans = ans.splitlines()[0].strip()
            ans = ans.strip("：:。.;`*_> ")
            if ans and ans not in seen:
                seen.add(ans)
                out.append(ans)
    return out

def build_canonical_answer(reasoning: str, finals: List[str]) -> Optional[str]:
    """
    Nemotron互換の answer テキストを構築。
    finals が空なら None。
    出力は厳密に "#### answer ..." 形式にする。
    """
    if not finals:
        return None
    # 複数最終解がある場合はカンマ区切り
    final_text = ", ".join(finals)
    reasoning = (reasoning or "").strip()
    return f"<think>\n{reasoning}\n</think>\n#### {final_text}"

# 既存API互換（他で使っていたら壊さないため）
def find_final_answer(text: str) -> Optional[str]:
    finals = find_final_answers(text)
    return finals[0] if finals else None

def split_reasoning_and_final(raw_answer: str) -> Tuple[str, List[str]]:
    """
    raw_answer から reasoning と final(複数) を抽出。
    - <think>...</think> があればそれを優先
    - なければ FINAL_PATTERNS/BOXED を使う
    """
    text = raw_answer.strip()

    m = THINK_BLOCK_RE.search(text)
    if m:
        reasoning = m.group("think").strip()
        after_think = text[m.end():].strip()
        finals = find_final_answers(after_think)
        if not finals:
            finals = find_final_answers(text)
        return reasoning, finals

    finals = find_final_answers(text)
    if finals:
        reasoning = text
        # 最終解の痕跡はざっくり除去（複数OK）
        for pat in FINAL_PATTERNS:
            reasoning = pat.sub("", reasoning)
        reasoning = reasoning.strip()
        return reasoning, finals

    return text, []

def convert_example(example) -> Optional[dict]:
    # まず messages 型を優先的に見る
    messages = example.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        question, raw_answer = extract_question_answer_from_messages(messages)
    else:
        question, raw_answer = extract_question_answer_from_flat(example)

    question = (question or "").strip()
    raw_answer = (raw_answer or "").strip()
    if not question or not raw_answer:
        return None

    reasoning, finals = split_reasoning_and_final(raw_answer)
    canonical_answer = build_canonical_answer(reasoning, finals)
    if not canonical_answer:
        return None

    return {"question": question, "answer": canonical_answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
                        help="Hugging Face dataset path")
    parser.add_argument("--subset", type=str, default=None, help="subset name if any")
    parser.add_argument("--split", type=str, default="train",
                        help="split to load (e.g., train, validation, test)")
    parser.add_argument("--save_format", type=str, default="parquet",
                        choices=["parquet", "jsonl"], help="output file format")
    parser.add_argument("--output", type=str, required=True,
                        help="output file path (e.g., out.parquet / out.jsonl)")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--drop_without_final", action="store_true",
                        help="drop rows where final couldn't be extracted (recommended)")
    args = parser.parse_args()

    ds = safe_load(args)

    # 変換
    def _map_fn(ex):
        out = convert_example(ex)
        return out if out is not None else {"question": None, "answer": None}

    # 代わりに generator で1件ずつ変換（例外は握りつぶしてスキップ）
    def gen():
        for i, ex in enumerate(ds):
            try:
                out = convert_example(ex)   # ←あなたの変換関数
                if out and out.get("question") and out.get("answer"):
                    yield {"question": out["question"], "answer": out["answer"]}
            except Exception:
                # 変な行は飛ばす（必要ならログ）
                # print(f"[warn] skip row {i}: {e}")
                continue

    features = Features({"question": Value("string"), "answer": Value("string")})
    ds2 = Dataset.from_generator(gen, features=features)

    # フィルタ
    if args.drop_without_final:
        ds2 = ds2.filter(lambda ex: ex["question"] is not None and ex["answer"] is not None,
                         num_proc=args.num_proc)
    else:
        # 欠損は空文字にしておく
        ds2 = ds2.map(lambda ex: {
            "question": ex["question"] or "",
            "answer": ex["answer"] or ""
        }, num_proc=args.num_proc)

    # 保存
    if args.save_format == "parquet":
        ds2.to_parquet(args.output)
    else:
        ds2.to_json(args.output, lines=True)


if __name__ == "__main__":
    main()
