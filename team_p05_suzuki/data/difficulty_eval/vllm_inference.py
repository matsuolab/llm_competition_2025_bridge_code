from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import yaml, json, os
import sys

print('all modules loaded')

# import settings
currdir=os.getcwd()
yml_name = sys.argv[1]
yml_path=os.path.join(currdir,yml_name)
print('yaml path {pth}'.format(pth=yml_path))

with open(yml_path, encoding='utf-8')as f:
    config = yaml.safe_load(f)
print('config file loaded')


# Load dataset from Huggingface hub
HF_TOKEN=config['data']['hf_token']
login(HF_TOKEN)
dataset_name=config['data']['dataset_name']
dataset=load_dataset(dataset_name)
print('dataset loaded')


## load model tokenizer for applying chat template
model_name = config['model']['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_path = os.path.join(config['model']['model_stored_path'],model_name.split("/")[-1])
print('tokenizer loaded')

# prepare chat template
system_prompt = (
    "You are a careful problem-solver.\n"
    "Solve the problem and return exactly one phrase in this format:\n"
    "Answer: {your chosen answer}\n"
    "No other text."
)

def format_chat(question: str, enable_thinking=True) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {question}"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # toggle Qwen “thinking” here
    )

# apply chate template
if isinstance(config['data'].get('sub_sample_inf',False), int):
    cfg_subsample=config['data'].get('sub_sample_inf',False)
    questions: list[str] = dataset["train"]["question"][:cfg_subsample] 
else:
    questions: list[str] = dataset["train"]["question"]

prompts = [format_chat(q) for q in questions]
print('applying chat template: completed')

## subfunctions/config to process batch
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# assign variables
batch_size=config['inference']['batch_size']
llm_kwargs={
        "gpu_memory_utilization": config['model']['gpu_memory_utilization'], # Use 90% of GPU memory
        "tensor_parallel_size": config['model']['tensor_parallel_size'],   # single GPU (RTX 4500 Ada 24GB)
        "max_num_seqs": batch_size,           
        "max_model_len": config['model']['max_model_len'],    
}  

# add quantization option if exists
quant = (config['model'].get("quantization") or "").strip()
if quant:
        llm_kwargs["quantization"] = quant

# start vLLM
llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        **llm_kwargs
    )

# Use task-appropriate sampling
sp = SamplingParams(
    max_tokens = config['inference']['max_tokens'],
    temperature=0.0,  # set >0 only if you want variability
    top_p=1.0,
)

# create output directory
os.makedirs(config['output']['dir'], exist_ok=True)

# output to dataframe and save output
dataname=dataset_name.split("/")[-1]
json_name='results-'+dataset_name.split("/")[-1] + '-' + model_name.split("/")[-1] + '.jsonl'
json_path=os.path.join(currdir,config['output']['dir'],json_name)
log_name='log-'+ dataset_name.split("/")[-1] + '-' + model_name.split("/")[-1] + '.jsonl'
log_path=os.path.join(currdir,config['output']['dir'],log_name)

# functions to save results and log
def save_log(log):
    with open(log_path, 'w') as file:
        file.write(json.dumps(log))

def save_output(subset,keys,output_file):
    with open(output_file, 'a') as f:
        for row_values in zip(*subset.values()):
            row_dict = dict(zip(keys, row_values))
            f.write(json.dumps(row_dict) + '\n')    

# load config file, in case resuming inference
try:
    log = json.load(log_path)
    model_output = json.load(json_path)
except:
    log={"total_number_question": len(questions),
         "questions_processed": 0,
         "model": model_name,
         "dataset": dataset_name,
         }
start_id=log['questions_processed']
processed_id=log['questions_processed']

# batch inference, save results and log for each batch
for batch_id, batch_prompts in enumerate(chunks(prompts, batch_size), start=start_id):
    outs = llm.generate(batch_prompts, sp)
    batch_start_idx=batch_size*batch_id
    batch_end_idx=batch_size*(batch_id+1)
    subset=dataset['train'][batch_start_idx:batch_end_idx]
    model_think=[]
    model_answer=[]
    for out in outs:
        text = out.outputs[0].text
        answer=text.split("</think>")[-1].split("\n\nAnswer:")[-1]
        model_answer.append(answer.strip(" "))        
        model_think.append(text.split("</think>")[0])
    subset['model_think']=model_think
    subset['model_answer']=model_answer
    save_output(subset,subset.keys(),json_path)
    processed_id += len(batch_prompts)
    log['questions_processed']=processed_id
    save_log(log)

print('Inference completed!')