#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, time, json, torch
from datetime import datetime
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd

# ----------------- 1. CLI 引数 -----------------
parser = argparse.ArgumentParser(
    description="Phi-4 reasoning inference on Nemotron-CrossThink (vLLM, batched)")

parser.add_argument("--num_samples", type=int, default=5,
                    help="how many samples to process from the top")
parser.add_argument("--batch_size", type=int, default=5,
                    help="how many prompts per vLLM generate call")
parser.add_argument("--dataset", type=str,
                    default="LLM-Compe-2025-Camino/NVIDIA-Nemotron-CrossThink",
                    help="HF dataset path")
parser.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
parser.add_argument("--subset", type=str, default="QA",
                    help="Data file to use (e.g., Math, QA)")
parser.add_argument("--split", type=str, default="train",
                    help="dataset split (usually 'train' when loading a single file)")
parser.add_argument("--out_base", type=str, default="phi4_crossthink",
                    help="base name of output JSONL; timestamp will be appended")
parser.add_argument("--max_tokens", type=int, default=4096,
                    help="maximum new tokens to generate")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"],
                    default="bf16", help="dtype for model weights")
parser.add_argument("--start_index", type=int, default=0,
                    help="Start offset into the dataset (0-based).")
# <<< 追加 (2025-08-02) パイプライン並列 >>>
parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                    help="Number of GPUs for pipeline parallelism")

args = parser.parse_args()


# (2. 出力ファイル名, 3. dtype変換 は変更なし)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"{args.out_base}_{timestamp}.jsonl"
dtype_map_llm = {"fp16": "half", "bf16": "bfloat16", "fp32": "float32"}


# ----------------- 4. HF token & データ取得 --------
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

### <<< 最終修正 >>>

# コマンドライン引数 --subset から正しいファイル名を構築します
# 例: --subset QA -> Nemotron-CrossThink-QA_qa.jsonl
# 例: --subset Math -> Nemotron-CrossThink-Math_qa.jsonl
target_file = f"Nemotron-CrossThink-{args.subset}_qa.jsonl"

print(f"Attempting to load the correct file: {target_file} from repository: {args.dataset}")

try:
    ds_all = load_dataset(
        args.dataset,
        data_files=target_file,
        split=args.split,
        token=token
    )

    # 残り件数を計算して、実際に処理する件数を決める
    remaining = max(0, len(ds_all) - args.start_index)
    num_to_load = min(args.num_samples, remaining)

    # 指定位置から num_to_load 件を選択
    ds = ds_all.select(range(args.start_index, args.start_index + num_to_load))

    print(f"Successfully loaded {len(ds_all)} total rows from {target_file}.")
    print(f"Processing rows [{args.start_index} .. {args.start_index + num_to_load - 1}] "
          f"({len(ds)} samples).")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Failed even with the correct filename. Please check for typos, network issues, or permissions.")
    print(f"(Attempted to load: {target_file})")
    exit(1)


# ----------------- 5. tokenizer -----------------
model_id = "microsoft/Phi-4-reasoning-plus"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ----------------- 6. システムプロンプト -----------------
# (システムプロンプトは変更なし)
SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions "
    "through a systematic thinking process before providing the final precise "
    "and accurate solutions. This requires engaging in a comprehensive cycle "
    "of analysis, summarizing, exploration, reassessment, reflection, "
    "backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and "
    "Solution using the specified format: <think> {Thought section} </think> "
    "{Solution section}. In the Thought section, detail your reasoning "
    "process in steps. Each step should include detailed considerations such "
    "as analysing questions, summarizing relevant findings, brainstorming new "
    "ideas, verifying the accuracy of the current steps, refining any errors, "
    "and revisiting previous steps. In the Solution section, based on various "
    "attempts, explorations, and reflections from the Thought section, "
    "systematically present the final solution that you deem correct. "
    "The Solution section should be logical, accurate, and concise and detail "
    "necessary steps needed to reach the conclusion. "
    "Be thorough, logically consistent, and avoid any mention that you were given the answer."
    "Now, try to solve the following question through the above guidelines:"
)

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

# ----------------- 7. vLLM 準備 -----------------
# <<< 変更 (2025-08-02) パイプライン並列を有効化 >>>
llm = LLM(
    model=model_id,
    dtype=dtype_map_llm[args.dtype],
    tensor_parallel_size=args.tensor_parallel_size,
    pipeline_parallel_size=args.pipeline_parallel_size
)

sampling = SamplingParams(temperature=0.8, top_p=0.95,
                          top_k=50, max_tokens=args.max_tokens)

# ----------------- 8. 出力ファイル open -------------
outfile = open(out_path, "a", encoding="utf-8")
print(f"Outputs will be written to {out_path}")

# ----------------- 9. 推論ループ -----------------
### <<< 変更 >>> バッチ内のインデックス(i)も取得
for step, batch in enumerate(batched(ds, args.batch_size), 1):
    prompt_strs, meta = [], []

    for i, item in enumerate(batch):
        ### <<< 変更 >>> カラム名を 'problem'/'answer' から 'question'/'ground_truth' に変更
        question = item["question"]
        gtruth   = item["ground_truth"]

        # --------------- ユーザープロンプト作成 ---------------
        user_content = (
            f"{question}\n\n"
            f"Known correct final answer: {gtruth}\n\n"
            "Be thorough, logically consistent, and avoid any mention that you were given the answer."
            "Provide the reasoning and final answer as instructed."
            "Put your final answer inside \\boxed{}"
            )

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        prompt_strs.append(prompt)

        ### <<< 変更 >>> メタ情報のキーをデータセットに合わせ、uuidの代わりにidを生成
        meta.append({
            "id":       f"crossthink_{args.subset}_{args.start_index + (step - 1) * args.batch_size + i}",
            "question":   question,
            "ground_truth": gtruth,
        })

    # ----------------- 生成 -----------------
    t0 = time.time()
    outputs = llm.generate(prompt_strs, sampling_params=sampling)
    elapsed = time.time() - t0
    print(f"[Batch {step}] {len(batch)} samples → {elapsed:.2f}s")

    # --------------- JSON Lines へ追記 ----------------
    for m, out in zip(meta, outputs):
        ### <<< 追加 >>> モデル出力をパースして reasoning と answer に分割
        output_text = out.outputs[0].text
        reasoning = ""
        answer_pred = ""
        try:
            # <think> タグを見つけて抽出
            start_think = output_text.find("<think>")
            end_think = output_text.find("</think>")
            if start_think != -1 and end_think != -1:
                reasoning = output_text[start_think + len("<think>"):end_think].strip()
                answer_pred = output_text[end_think + len("</think>"):].strip()
            else:
                # タグが見つからない場合は、全体をanswerに入れる
                answer_pred = output_text.strip()
        except Exception as e:
            print(f"Error parsing output: {e}")
            answer_pred = output_text # パース失敗時は生テキストを保存

        ### <<< 変更 >>> 保存するレコードの形式をSFT用に変更
        record = {
            "id":           m["id"],
            "question":     m["question"],
            "ground_truth": m["ground_truth"],
            "reasoning":    reasoning,
            "answer":       answer_pred,
            "raw_output":   output_text # デバッグ用に元の出力も保存
        }
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    outfile.flush()

outfile.close()
print("All done!")
