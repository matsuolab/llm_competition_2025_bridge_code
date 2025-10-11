from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import yaml, json, os, glob
import sys

print('all modules loaded')
def main():
    # import settings
    currdir=os.getcwd()
    yml_name = sys.argv[1]
    yml_path=os.path.join(currdir,yml_name)
    print('yaml path {pth}'.format(pth=yml_path))

    with open(yml_path, encoding='utf-8')as f:
        config = yaml.safe_load(f)
    print('config file loaded')

    ## load model tokenizer for applying chat template
    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_path = os.path.join(config['model']['model_stored_path'],model_name.split("/")[-1])
    print('tokenizer loaded')



    # prepare chat template, adapted from MatsuoLab's official judgement script
    # https://github.com/matsuolab/llm_bridge_prod/blob/master/eval_hle/hle_benchmark/run_judge_results.py
    system_prompt = ("Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below."
    "Your judgement must be in the format and criteria specified below:"
    "Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."
    "Answer 'no' if [response] is longer than 40 words"
    )

    def format_chat(question,model_answer,correct_answer) :
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[question]: {question}\n [correct_answer]: {correct_answer}\n [response]: {model_answer}\n" }
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable Qwen “thinking” here
        )

    # function to replace super long answers
    def replace_long_answers(batch): 
        MAX_CHARS = 2000
        answers = batch["model_answer"]
        batch["model_answer"] = [
            "MODEL_COULD_NOT_ANSWER" if (isinstance(a, str) and len(a) > MAX_CHARS) else a
            for a in answers
        ]
        return batch

    # Load inference results files, and conver to judge prompt
    def dataset_to_prompt(file_name):

        dataset=load_dataset('json',data_files=file_name)
        print(file_name)
        print(dataset)
        print('dataset loaded')
        dataset["train"] = dataset["train"].map(replace_long_answers, batched=True)

        questions: list[str] = dataset["train"]["question"]
        model_response: list[str] = dataset["train"]["model_answer"]
        correct_answer: list[str] = dataset["train"]["answer"]
        # applies to all splits (train/validation/test) at once

        response=[]
        for i,item in enumerate(model_response):
            if len(item.split(' '))>100:
                response.append(item[:400])
            else:
                response.append(item)

        # apply chate template
        prompts = [format_chat(q,m,c) for q,m,c in zip(questions,response,correct_answer)]
        print('applying chat template: completed')
        return prompts



    ## subfunctions/config to process batch
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # assign vllm config variables
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
        temperature=0.1,  # set >0 only if you want variability
        top_p=1.0,
    )

    # functions to save results and log
    def save_log(log,log_path):
        with open(log_path, 'w') as file:
            file.write(json.dumps(log))
    def save_output(subset,keys,output_file):
        with open(output_file, 'a') as f:
            for row_values in zip(*subset.values()):
                row_dict = dict(zip(keys, row_values))
                f.write(json.dumps(row_dict) + '\n')    

    # load all results files
    file_list=[os.path.basename(p) for p in glob.glob(os.path.join(config['output']['dir'],'results*.jsonl'))]
    start_id=0

    for i,file in enumerate(file_list):
        file_name=os.path.join(currdir,config['output']['dir'],file)
        prompts=dataset_to_prompt(file_name) # fullpath of jsonl data file

        # batch inference, save results and log for each batch
        judge=[]
        for batch_id, batch_prompts in enumerate(chunks(prompts, batch_size), start=start_id):
            outs = llm.generate(batch_prompts, sp)
            batch_start_idx=batch_size*batch_id
            batch_end_idx=batch_size*(batch_id+1)
            #subset=dataset['train'][batch_start_idx:batch_end_idx]
            for out in outs:
                text = out.outputs[0].text
                judge.append(text)

        dataset_out=load_dataset('json',data_files=file_name)
        dataset_out=dataset_out['train']
        dataset_out=dataset_out.remove_columns(['LLM_judge'])
        dataset_out=dataset_out.add_column('LLM_judge',judge)

        # output to dataframe and save output
        json_name=file.replace('results-','judged-')
        json_path=os.path.join(currdir,config['output']['dir'],json_name)
        export_data=dataset_out[:]
        save_output(export_data,export_data.keys(),json_path)
    
    print('judgement completed')

if __name__ == "__main__":
    main()