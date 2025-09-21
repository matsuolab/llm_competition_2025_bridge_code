#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-32B SFT (Mixed: Math Hard/Mid + Science MC)
- QLoRA: 4bit NF4 + double quant
- LoRA: r=64, alpha=16, dropout=0.05, target: q,k,v,o,gate,up,down
- Patch: resolve model repo-id → cached snapshot (offline) before from_pretrained
- Adds MAX_SAMPLES for smoke, robust DDP bootstrap, and Final Answer normalization.
"""

import os, re, json, math, csv, random, datetime
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# -----------------------------
# グローバル変数 (引数で上書きされる)
# -----------------------------
DATA_ROOT = None
OUT_DIR = None
MAX_LEN = 4096
rank, world_size, local_rank = 0, 1, 0

# -----------------------------
# Snapshot resolver (offline)
# -----------------------------
def resolve_cached_snapshot(repo_id_or_path: str) -> str:
    if os.path.isdir(repo_id_or_path):
        cfg = os.path.join(repo_id_or_path, "config.json")
        if os.path.exists(cfg):
            return repo_id_or_path
        snaps = os.path.join(repo_id_or_path, "snapshots")
        if os.path.isdir(snaps):
            cands = [os.path.join(snaps, d) for d in os.listdir(snaps)]
            cands = [p for p in cands if os.path.exists(os.path.join(p, "config.json"))]
            if cands:
                cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return cands[0]
        raise ValueError(f"config.json not found under {repo_id_or_path}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        if rank == 0:
            print(f"[WARN] huggingface_hub not available ({e}); falling back to repo id.")
        return repo_id_or_path
    return snapshot_download(repo_id=repo_id_or_path, local_files_only=True)

# -----------------------------
# Formatting / extraction utils
# -----------------------------
def _strip_tex(s: str) -> str:
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\$+", "", s)
    return s.strip()

def _extract_final_from_gsm8k(ans: str) -> str:
    m = re.search(r"####\s*([^\n]+)", ans)
    if m:
        return _strip_tex(m.group(1))
    for line in reversed(ans.splitlines()):
        line = line.strip()
        if line:
            return _strip_tex(line)
    return _strip_tex(ans)

def _extract_final_from_math_solution(sol: str) -> str:
    m = re.findall(r"\\boxed\{([^}]+)\}", sol)
    if m: return _strip_tex(m[-1])
    m2 = re.search(r"(?i)answer[:\s]+([^\n]+)", sol)
    if m2: return _strip_tex(m2.group(1))
    for line in reversed(sol.splitlines()):
        line = line.strip()
        if line:
            return _strip_tex(line)
    return _strip_tex(sol)

def _final_line(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^0-9A-Za-z\\-\\+\\*/\\.\\(\\), ]", "", s)

def format_reasoning_prompt(question: str, reasoning: str, final_answer: str, choices_block: str = "") -> str:
    parts = [f"### Question:\n{question}"]
    if choices_block:
        parts.append(f"### Choices:\n{choices_block}")
    parts.append("### Reasoning:\nLet's think step by step.")
    if reasoning:
        parts.append(reasoning.strip())
    parts.append(f"Final Answer: {_final_line(final_answer)}")
    return "\n\n".join(parts)


# -----------------------------
# Dataset loaders
# -----------------------------
def load_gsm8k():
    path = Path(DATA_ROOT) / "gsm8k" / "train.json"
    data = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q = item.get("question", "")
                a = item.get("answer", "")
                fa = _extract_final_from_gsm8k(a)
                data.append(format_reasoning_prompt(q, a, fa))
    return data

def load_metamath():
    root = Path(DATA_ROOT) / "MetaMathQA"
    data = []
    if root.exists():
        for file in root.rglob("*"):
            try:
                if file.suffix in [".jsonl", ".json"]:
                    with file.open("r", encoding="utf-8") as f:
                        if file.suffix == ".jsonl" or "jsonl" in file.name:
                            for line in f:
                                if not line.strip(): continue
                                obj = json.loads(line)
                                q = obj.get("query", obj.get("question", ""))
                                a = obj.get("response", obj.get("answer", ""))
                                fa = _extract_final_from_gsm8k(a)
                                data.append(format_reasoning_prompt(q, a, fa))
                        else:
                            obj = json.load(f)
                            if isinstance(obj, list):
                                for it in obj:
                                    q = it.get("query", it.get("question", ""))
                                    a = it.get("response", it.get("answer", ""))
                                    fa = _extract_final_from_gsm8k(a)
                                    data.append(format_reasoning_prompt(q, a, fa))
            except Exception:
                continue
    return data

def load_math_benchmark(level_min=3):
    root = Path(DATA_ROOT) / "hendrycks_MATH_benchmark" / "train"
    data = []
    if root.exists():
        for subject_dir in root.iterdir():
            if not subject_dir.is_dir(): continue
            for file in subject_dir.glob("*.json"):
                try:
                    obj = json.loads(file.read_text(encoding="utf-8"))
                    if obj.get("level", 0) >= level_min:
                        q = obj.get("problem", "")
                        sol = obj.get("solution", "")
                        fa = _extract_final_from_math_solution(sol)
                        data.append(format_reasoning_prompt(q, sol, fa))
                except Exception:
                    continue
    return data

def load_hardmath():
    root = Path(DATA_ROOT) / "HARDMath"
    data = []
    if root.exists():
        for file in root.rglob("*"):
            try:
                if file.suffix == ".jsonl":
                    for line in file.read_text(encoding="utf-8").splitlines():
                        if not line.strip(): continue
                        obj = json.loads(line)
                        q = obj.get("question") or obj.get("prompt") or ""
                        sol = obj.get("solution") or obj.get("answer_explanation") or ""
                        ans = obj.get("final_answer") or obj.get("answer") or _extract_final_from_math_solution(sol)
                        fa = _extract_final_from_math_solution(ans if isinstance(ans,str) else sol)
                        data.append(format_reasoning_prompt(q, sol, fa))
                elif file.suffix == ".json":
                    obj = json.loads(file.read_text(encoding="utf-8"))
                    if isinstance(obj, list):
                        for it in obj:
                            q = it.get("question") or it.get("prompt") or ""
                            sol = it.get("solution") or it.get("answer_explanation") or ""
                            ans = it.get("final_answer") or it.get("answer") or _extract_final_from_math_solution(sol)
                            fa = _extract_final_from_math_solution(ans if isinstance(ans,str) else sol)
                            data.append(format_reasoning_prompt(q, sol, fa))
            except Exception:
                continue
    return data

def _read_csv_rows(dir_path: Path):
    rows = []
    if not dir_path.exists(): return rows
    for file in dir_path.glob("*.csv"):
        with file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k.lower(): v for k, v in r.items()})
    return rows

def load_gpqa_and_chem():
    gpqa_rows = _read_csv_rows(Path(DATA_ROOT) / "gpqa_csv")
    chem1_rows = _read_csv_rows(Path(DATA_ROOT) / "ChemQA_csv")
    chem2_rows = _read_csv_rows(Path(DATA_ROOT) / "ChemistryQA_csv")

    def build_mc(rows):
        data = []
        for r in rows:
            q = r.get("question") or r.get("prompt") or r.get("stem") or ""
            choices = []
            for key in ["a","b","c","d","e"]:
                if key in r and r[key]:
                    choices.append((key.upper(), r[key]))
            if not choices:
                for key in r.keys():
                    if len(key)==1 and key in "abcde":
                        choices.append((key.upper(), r[key]))
            ans = r.get("answer") or r.get("label") or r.get("gold") or r.get("final_answer") or ""
            ans_letter = str(ans).strip().upper()
            if len(ans_letter) > 1:
                m = re.search(r"\b([A-E])\b", ans_letter)
                if m: ans_letter = m.group(1)
            choice_block = "\n".join([f"{k}. {v}" for k, v in choices if v])
            if q and ans_letter:
                data.append(format_reasoning_prompt(q, "", ans_letter, choices_block=choice_block))
        return data

    return build_mc(gpqa_rows) + build_mc(chem1_rows) + build_mc(chem2_rows)

# -----------------------------
# Mixture builder
# -----------------------------
def build_mixture(args):
    math_hard = load_math_benchmark(level_min=3) + load_hardmath()
    math_mid  = load_gsm8k() + load_metamath()
    science   = load_gpqa_and_chem()

    if rank == 0:
        print(f"[Loaded] MathHard={len(math_hard)}  MathMid={len(math_mid)}  Science={len(science)}")

    total_pool = len(math_hard) + len(math_mid) + len(science)
    if total_pool == 0:
        raise RuntimeError(f"No training data found under {DATA_ROOT}.")

    limit = args.max_samples if args.max_samples > 0 else total_pool

    tgt_hard = int(limit * args.ratio_math_hard)
    tgt_mid  = int(limit * args.ratio_math_mid)
    tgt_sci  = limit - tgt_hard - tgt_mid

    random.shuffle(math_hard); random.shuffle(math_mid); random.shuffle(science)
    mix = math_hard[:tgt_hard] + math_mid[:tgt_mid] + science[:tgt_sci]
    random.shuffle(mix)

    if rank == 0:
        print(f"[Mixture] target={limit} -> Hard={tgt_hard}, Mid={tgt_mid}, Sci={tgt_sci}, Final={len(mix)}")
    return mix

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_dataset(tokenizer, texts, max_length=MAX_LEN):
    ds = Dataset.from_list([{"text": t} for t in texts])

    def tok(batch):
        out = tokenizer(
            batch["text"], truncation=True, padding="max_length",
            max_length=max_length, return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    tokenized = ds.map(tok, batched=True, remove_columns=["text"], desc="Tokenizing")
    tokenized.set_format("torch")
    return tokenized

# -----------------------------
# Model & tokenizer (QLoRA)
# -----------------------------
def setup_model_and_tokenizer(model_id_or_path: str):
    resolved = resolve_cached_snapshot(model_id_or_path)
    if rank == 0:
        print(f"[Model] Resolved path: {resolved}")

    tok = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    try:
        model.resize_token_embeddings(len(tok), pad_to_multiple_of=8)
    except Exception:
        pass
    if rank == 0:
        model.print_trainable_parameters()
    return model, tok

# -----------------------------
# Train loop
# -----------------------------
def train(args):
    global rank, world_size, local_rank, DATA_ROOT, OUT_DIR, MAX_LEN
    
    # DDP bootstrap
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ.get('SLURM_NTASKS', '1'))
        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=7200))

    torch.cuda.set_device(local_rank)
    
    # 引数をグローバル変数に設定
    DATA_ROOT = args.data_root
    OUT_DIR = args.output_dir
    MAX_LEN = args.max_len
    random.seed(args.seed); torch.manual_seed(args.seed)
    
    ACC_STEPS = max(1, args.global_bsz // max(1, (args.bsz_per_gpu * world_size)))

    if rank == 0:
        print("="*50)
        print("DeepSeek-R1-Distill-Qwen-32B :: Mixed SFT (QLoRA, patched)")
        print(f"World size: {world_size} | local_rank: {local_rank}")
        print(f"Per-GPU BS: {args.bsz_per_gpu} | Global BS: {args.global_bsz} | Accum: {ACC_STEPS}")
        print(f"LR: {args.lr} | Epochs: {args.epochs} | MaxLen: {args.max_len} | MAX_SAMPLES: {args.max_samples}")
        print(f"Output Dir: {OUT_DIR}")
        print("="*50)

    model, tok = setup_model_and_tokenizer(args.model_id)
    texts = build_mixture(args)
    dataset = tokenize_dataset(tok, texts, max_length=args.max_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size>1 else None
    loader = DataLoader(dataset, batch_size=args.bsz_per_gpu, sampler=sampler,
                        collate_fn=collator, shuffle=(sampler is None), num_workers=args.num_workers)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / ACC_STEPS) * args.epochs
    sched = CosineAnnealingLR(optim, T_max=total_steps)

    best_loss = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)

    for ep in range(args.epochs):
        if world_size>1 and sampler and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(ep)

        model.train()
        running, step_in_accum = 0.0, 0
        pbar = tqdm(loader, disable=(rank!=0), desc=f"Epoch {ep+1}")

        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
            attn = batch["attention_mask"].to(local_rank, non_blocking=True)
            labels = batch["labels"].to(local_rank, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / ACC_STEPS
            loss.backward()
            step_in_accum += 1
            running += loss.item()

            if step_in_accum == ACC_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step_in_accum = 0

            if rank == 0:
                pbar.set_postfix({"loss": f"{(running*ACC_STEPS)/max(1,(step+1)):.4f}",
                                  "lr": f"{sched.get_last_lr()[0]:.2e}"})

            if step % 50 == 0:
                torch.cuda.empty_cache()
                if world_size>1 and step>0: dist.barrier()

        avg_loss = (running * ACC_STEPS) / max(1, len(loader))
        if rank == 0:
            print(f"Epoch {ep+1} avg loss: {avg_loss:.4f}")
            ep_dir = f"{OUT_DIR}/epoch_{ep+1}"
            os.makedirs(ep_dir, exist_ok=True)
            model.save_pretrained(ep_dir)
            tok.save_pretrained(ep_dir)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_dir = f"{OUT_DIR}/best_model"
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                print(f"✅ Best updated: {best_loss:.4f}")

        if world_size>1: dist.barrier()

    if rank == 0:
        print("\n✅ SFT completed successfully.")
        print(f"Best loss: {best_loss:.4f}")

    if world_size>1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek 32B SFT Script")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", help="Model repository ID")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the unified_datasets directory")
    parser.add_argument("--output_dir", type=str, default="./results/r1_32b_sft_mixed", help="Directory to save the results")
    
    # Mixture Ratios
    parser.add_argument("--ratio_math_hard", type=float, default=0.50)
    parser.add_argument("--ratio_math_mid", type=float, default=0.30)
    parser.add_argument("--ratio_science", type=float, default=0.20)

    # Hparams
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bsz_per_gpu", type=int, default=2)
    parser.add_argument("--global_bsz", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples for training (0 for all)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(args)
