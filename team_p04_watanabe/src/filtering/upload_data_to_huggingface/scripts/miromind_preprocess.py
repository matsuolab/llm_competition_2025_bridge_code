import re
from datasets import load_dataset
from huggingface_hub import login

def extract_answer(solution_str):
    """Extract the final answer from the solution string"""
    # Remove think tags to extract answer from the actual response part
    main_content = solution_str
    if "<think>" in solution_str and "</think>" in solution_str:
        # Extract content after </think> tag
        main_content = solution_str.split("</think>")[-1].strip()
    
    # Look for boxed answers first (common in math problems)
    boxed_pattern = re.search(r"\\boxed\{([^}]+)\}", main_content)
    if boxed_pattern:
        answer = boxed_pattern.group(1).strip()
        # Extract just the number if it's there
        numbers = re.findall(r"[\-\d\.\,]+", answer)
        if numbers:
            return numbers[0].replace(",", "")
        return answer
    
    # Try to find answer after #### pattern (similar to GSM8K)
    solution = re.search(r"####\s*([\-\d\.\,]+)", main_content)
    if solution:
        final_answer = solution.group(1).replace(",", "")
        return final_answer
    
    # Try to find answer in "The answer is X" pattern
    answer_pattern = re.search(r"(?:answer is|答えは)\s*([\-\d\.\,]+)", main_content, re.IGNORECASE)
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "")
    
    # Try to find the last number in the main content
    numbers = re.findall(r"[\-\d\.\,]+", main_content)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""

def transform_response(text):
    """Extract answer and format as #### answer"""
    # First remove the boxed answer patterns that appear at the end
    # Pattern to match \[ \boxed{...} \quad \text{and} \quad \boxed{...} \]
    text = re.sub(r'\\\[\s*\\boxed\{[^}]+\}\s*\\quad\s*\\text\{and\}\s*\\quad\s*\\boxed\{[^}]+\}\s*\\\]', '', text)
    # Also remove simpler boxed patterns at the end
    text = re.sub(r'\\\[\s*\\boxed\{[^}]+\}\s*\\\]\s*$', '', text)
    
    answer = extract_answer(text)
    if answer:
        # Check if there are multiple answers (e.g., "5 and 8")
        if "and" in text.lower() and re.findall(r'\\boxed\{([^}]+)\}', text):
            boxes = re.findall(r'\\boxed\{([^}]+)\}', text)
            if len(boxes) >= 2:
                # Format as "X and Y" instead of just the first answer
                answer = f"{boxes[0]} \\quad \\text{{and}} \\quad {boxes[1]}"
        return f"{text.strip()}\n\n#### {answer}"
    return text

def main():
    # Load the dataset
    print("Loading MiroMind dataset...")
    dataset = load_dataset("miromind-ai/MiroMind-M1-SFT-719K")
    
    # Transform the dataset
    print("Transforming dataset...")
    
    def transform_example(example):
        # Remove 'id' field and rename 'response' to 'answer' with transformation
        new_example = {
            'question': example['question'],
            'answer': transform_response(example['response'])
        }
        return new_example
    
    # Apply transformation to all splits
    transformed_dataset = dataset.map(
        transform_example,
        remove_columns=['id', 'response']  # Remove both 'id' and 'response'
    )
    
    # Login to Hugging Face (you'll need to set your token)
    print("Please enter your Hugging Face token:")
    token = input().strip()
    login(token=token)
    
    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    repo_id = "LLMcompe-Team-Watanabe/MiroMind-M1-SFT-719K-transformed"
    
    transformed_dataset.push_to_hub(
        repo_id,
        private=False,  # Set to True if you want a private dataset
        commit_message="Upload transformed MiroMind dataset"
    )
    
    print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
