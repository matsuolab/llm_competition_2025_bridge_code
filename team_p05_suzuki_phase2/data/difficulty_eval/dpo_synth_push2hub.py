from huggingface_hub import login
from datasets import load_dataset, Dataset
import pandas as pd
import json, os, glob


HF_TOKEN="hf_TOKEN" # read write key
login(HF_TOKEN)

source_datadir='/home/USER/vllm_output'
file_list = glob.glob(os.path.join(source_datadir, 'judged*.jsonl'))
dpo_repo_name="HF_UPLOAD_REPO"


# Load dataset from json
print(file_list)
df1=pd.read_json(file_list[0],lines=True)
df2=pd.read_json(file_list[1],lines=True)
df3=pd.read_json(file_list[2],lines=True)
df4=pd.read_json(file_list[3],lines=True)
print(len(df1))

# create tags for think over flow, answer judge
df1['think_overflow']= (df1['model_think']==df1['model_answer']).astype(int)
df2['think_overflow']= (df2['model_think']==df2['model_answer']).astype(int)
df3['think_overflow']= (df3['model_think']==df3['model_answer']).astype(int)
df4['think_overflow']= (df4['model_think']==df4['model_answer']).astype(int)
think_of=df1['think_overflow']+df2['think_overflow'] + df3['think_overflow']+ df4['think_overflow']

# convert yes/no to 1/0
df1['LLM_judge']=(df1['LLM_judge']=='yes').astype(int)
df2['LLM_judge']=(df2['LLM_judge']=='yes').astype(int)
df3['LLM_judge']=(df3['LLM_judge']=='yes').astype(int)
df4['LLM_judge']=(df4['LLM_judge']=='yes').astype(int)

# assign 1:correct, 0:error, -1: think overflow
df1['final_judge']=df1['LLM_judge']-df1['think_overflow']
df2['final_judge']=df2['LLM_judge']-df2['think_overflow']
df3['final_judge']=df3['LLM_judge']-df3['think_overflow']
df4['final_judge']=df4['LLM_judge']-df4['think_overflow']

# compute questions to include: (correct 1/4 or 2/4) and (think over flow =< 3/4)
correct_num=df1['LLM_judge']+df2['LLM_judge']+df3['LLM_judge']+df4['LLM_judge']
inc_idx=((correct_num>0)&(correct_num<3)&(think_of<4)).to_list()
print('Total questions {:d}'.format(len(df1)))
print('Included questions {:d}'.format(sum(inc_idx)))

# subfunction to format think-answer 
def make_content(df,idx):
    think=df['model_think'][idx].replace("<think>","").replace("</think>","")
    answer=df['model_answer'][idx]
    content = f"<think>{think}</think>{answer}"
    return content

# convert data_id from float to string for Arxiv dataset
df1['data_id']=df1['data_id'].astype('str')

# create DPO dataset
# if include think overflow inference, set df*['final_judge'][idx]<1
# if exclude think overflow inference, set df*['final_judge'][idx]==0
dpo_df=[]
for idx in range(len(df1)):
    if inc_idx[idx]==True:
        chosen_samples=[]
        rejected_samples=[]

        m1_content=make_content(df1,idx)
        if df1['final_judge'][idx]==1:
            chosen_samples.append(m1_content)
        elif df1['final_judge'][idx]<1:
            rejected_samples.append(m1_content)

        m2_content=make_content(df2,idx)
        if df2['final_judge'][idx]==1:
            chosen_samples.append(m2_content)
        elif df2['final_judge'][idx]<1:
            rejected_samples.append(m2_content)

        m3_content=make_content(df3,idx)
        if df3['final_judge'][idx]==1:
            chosen_samples.append(m3_content)
        elif df3['final_judge'][idx]<1:
            rejected_samples.append(m3_content)

        m4_content=make_content(df4,idx)
        if df4['final_judge'][idx]==1:
            chosen_samples.append(m4_content)
        elif df4['final_judge'][idx]<1:
            rejected_samples.append(m4_content)
        

        base_id=df1['data_id'][idx]
        i=1
        question=df1['question'][idx]
        subject=df1['subject'][idx]

        for chosen in chosen_samples:
            for rejected in rejected_samples:

                dpo_entry = {
                    'data_id': f"{base_id}_{i:04d}_dpo",
                    'question': question,  # add original question
                    'prompt': question,  # add prompt column (content is identical to "question")
                    'messages': [
                        {
                            'role': 'user',
                            'content': question
                        },
                        {
                            'role': 'assistant', 
                            'content': chosen
                        }
                    ],
                    'rejected_messages': [
                        {
                            'role': 'user',
                            'content': question
                        },
                        {
                            'role': 'assistant',
                            'content': rejected
                        }
                    ],
                    'subject': subject, 
                }
                dpo_df.append(dpo_entry)


# push DPO dataset to HF hub
ds_dpo = Dataset.from_list(dpo_df)
print(dpo_repo_name)
print(len(ds_dpo))
ds_dpo.push_to_hub(dpo_repo_name, private=True)
print('Upload completed')

