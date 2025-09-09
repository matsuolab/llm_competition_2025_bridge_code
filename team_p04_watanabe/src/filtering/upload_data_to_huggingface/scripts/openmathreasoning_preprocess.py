import re
from datasets import load_dataset
from huggingface_hub import login

def main():
    # Load the dataset
    print("Loading OpenMathReasoning dataset...")
    dataset = load_dataset("nvidia/OpenMathReasoning")
    
    # Define allowed problem sources
    allowed_sources = {
        "aops_c6_high_school_olympiads",
        "aops_c7_college_math", 
        "MATH_training_set"
    }
    
    # Transform the dataset
    print("Transforming dataset...")
    
    def filter_and_transform(example):
        # Check if problem_type is has_answer_extracted
        if example.get('problem_type') != 'has_answer_extracted':
            return False
            
        # Check if problem_source is in allowed list
        if example.get('problem_source') not in allowed_sources:
            return False
            
        return True
    
    def transform_example(example):
        # Get the expected answer
        expected_answer = example.get('expected_answer', '').strip()
        
        # Create the new example with renamed columns
        new_example = {
            'question': example['problem'],
            'answer': f"{example['generated_solution']}\n\n#### {expected_answer}"
        }
        return new_example
    
    # Check what splits are available
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process each split
    transformed_splits = {}
    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        
        # Filter first
        filtered_split = dataset[split_name].filter(filter_and_transform)
        
        # Then transform
        transformed_split = filtered_split.map(
            transform_example,
            remove_columns=filtered_split.column_names  # Remove all original columns
        )
        
        transformed_splits[split_name] = transformed_split
        
        # Print split info
        print(f"Original {split_name} size: {len(dataset[split_name])}")
        print(f"Filtered {split_name} size: {len(transformed_split)}")
    
    # Create a DatasetDict
    from datasets import DatasetDict
    transformed_dataset = DatasetDict(transformed_splits)
    print(f"Filtered by problem_type='has_answer_extracted' and problem_source in {allowed_sources}")
    
    # Login to Hugging Face
    print("\nPlease enter your Hugging Face token:")
    token = input().strip()
    login(token=token)
    
    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    repo_id = "LLMcompe-Team-Watanabe/OpenMathReasoning-filtered-transformed"
    
    transformed_dataset.push_to_hub(
        repo_id,
        private=False,
        commit_message="Upload filtered and transformed OpenMathReasoning dataset"
    )
    
    print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
