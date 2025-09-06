import argparse
import os
import re
from datasets import load_dataset, concatenate_datasets, Dataset


if __name__ == "__main__":

    data_source = [
            #"llm-compe-2025-kato/HARDMath_to_Bespoke_to_gsm8k_new",
            #"llm-compe-2025-kato/deepmath_data_generated_by_rlt_model",
            #"llm-compe-2025-kato/FORML4-Bespoke-Format_to_gsm8k",
            #"llm-compe-2025-kato/numina-math-to-gsm8k",
            ]
    
   # merged = concatenate_datasets([load_dataset(name)["train"] for name in data_source])
 
    merged = load_dataset("llm-compe-2025-kato/synthetic_data_from_DeepMath_103k_for_SFT", streaming=True)["train"]


    data_list = [x for x in merged.take(900)]

    merged = Dataset.from_list(data_list)

    split = merged.train_test_split(test_size=0.2, seed=42)
    train_dataset, test_dataset = split["train"], split["test"]


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            #question = example.pop("question")
            question = example.pop("Problem")
            #system_prompt = example.pop("system")
            system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
            question_raw = {"system": system_prompt, "user": question }
            #answer_raw = example.pop("answer")
            answer_raw = example.pop("Answer")
            cot = example.pop("CoT")
            solution = ""
            data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question
                            }
                        ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "answer": "<think>" + cot + "</think>"+answer_raw,
                        "question": question_raw,
                        },
                    }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = "/home/Competition2025/P07/shareP07/share_env/data"

    train_dataset.to_parquet(os.path.join(local_dir, "synthetic_DeepMath_900_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "synthetic_DeepMath_900_test.parquet"))
