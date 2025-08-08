# OpenMathReasoning Prompt Template

This prompt template is extracted from the `convert_openmath_to_prompt_response` function in `train/scripts/data_preprocess/open_math_reasoning_genselect.py`.

## Prompt Structure

The prompt is constructed by combining:
1. **Problem Statement**: Extracted from the problem field
2. **Solution Examples**: Multiple solution approaches with boxed answers

## Template Format

```
{problem_statement}

{solution_1}

{solution_2}

{solution_n}
```

## Components

### Problem Statement
- Extracted using regex pattern: `Problem:\s*\n(.*?)\n\s*Solutions:`
- Line numbers and arrow markers (`\d+→`) are cleaned from the text

### Solutions Dictionary
- Each solution contains:
  - `answer`: Content from `\boxed{...}` expressions
  - `solution`: Full solution text including "Solution X:" header
- Solutions are numbered (0, 1, 2, etc.)
- Only solutions with valid `\boxed{}` content are included

### Response Selection
- Based on judgment/judgement number from generated_solution
- Pattern: `Judgements?:\s*(\d+)` or `Judgment:\s*(\d+)`
- Returns the answer from the selected solution number

## Processing Functions

### extract_problem(text)
Extracts problem statement and cleans formatting

### extract_solutions_dict(text) 
Creates dictionary of numbered solutions with answers and full text

### extract_judgement(text)
Finds the selected solution number from judgment

### convert_openmath_to_prompt_response(example)
Main function that:
1. Extracts problem from example["problem"]
2. Extracts solutions dictionary from example["problem"] 
3. Extracts judgment from example["generated_solution"]
4. Combines problem + all solutions as prompt
5. Returns selected solution's answer as response

## Example Usage

```python
from open_math_reasoning_genselect import convert_openmath_to_prompt_response

# Example dataset sample
example = {
    "problem": "Problem:\nSolve for x...\nSolutions:\nSolution 0:\n...\\boxed{a}...",
    "generated_solution": "Judgment: 0",
    "expected_answer": "a"
}

prompt, response = convert_openmath_to_prompt_response(example)
```

## Data Format Output

For VERL training, the processed data includes:
- `prompt`: List with user role and combined problem+solutions content
- `ability`: "math_reasoning"
- `reward_model`: {"style": "rule", "ground_truth": expected_answer}
- `extra_info`: Original data preservation for SFT training