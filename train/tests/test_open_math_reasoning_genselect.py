import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'data_preprocess'))

from open_math_reasoning_genselect import convert_openmath_to_prompt_response

example_data_dir = os.path.join(os.path.dirname(__file__), '..', 'example_data')
output_dir = os.path.join(os.path.dirname(__file__), "output")


def load_mock_data():
    """Load mock data from example files."""
    # Read problem data
    problem_path = os.path.join(example_data_dir, 'omr_g_problem1.txt')
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_content = f.read()
    
    # Read generated solution data
    solution_path = os.path.join(example_data_dir, 'omr_g_gen_solution1.txt')
    with open(solution_path, 'r', encoding='utf-8') as f:
        solution_content = f.read()
    
    # Create mock example data structure
    mock_example = {
        "problem": problem_content,
        "generated_solution": solution_content,
        "expected_answer": "\\dfrac{2a}{1 + \\sqrt{1 - a^2}}"  # Mock expected answer
    }
    
    return mock_example


def load_mock_data2():
    """Load mock data from example files (dataset 2)."""
    # Read problem data
    problem_path = os.path.join(example_data_dir, 'omr_g_problem2.txt')
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_content = f.read()
    
    # Read generated solution data
    solution_path = os.path.join(example_data_dir, 'omr_g_gen_solution2.txt')
    with open(solution_path, 'r', encoding='utf-8') as f:
        solution_content = f.read()
    
    # Create mock example data structure
    mock_example = {
        "problem": problem_content,
        "generated_solution": solution_content,
        "expected_answer": "\\dfrac{1}{2}"  # Expected answer based on judgment 0
    }
    
    return mock_example


def test_convert_openmath_to_prompt_response():
    """Test convert_openmath_to_prompt_response method with mock dataset sample."""
    
    # Load mock data instead of real dataset
    mock_sample = load_mock_data()
    
    print("Mock sample keys:", list(mock_sample.keys()))
    
    # Call the method under test
    prompt, response = convert_openmath_to_prompt_response(mock_sample)
    
    # Basic validation tests
    assert isinstance(prompt, str), "Prompt should be a string"
    assert len(prompt) > 0, "Prompt should not be empty"
    
    # The response["answer"] should be just the mathematical answer (from \boxed{}), not the full solution text
    assert response == "a", f"Expected answer 'a', got '{response['answer']}'"
    
    # Response should not contain evaluation process text
    assert "Evaluation Process:" not in response, f"Response should not contain 'Evaluation Process:' but found it in: {response}"
    
    # Prompt should contain all solutions (0, 1, and 2)
    assert "Solution 0:" in prompt, "Prompt should contain 'Solution 0:'"
    assert "Solution 1:" in prompt, "Prompt should contain 'Solution 1:'"
    assert "Solution 2:" in prompt, "Prompt should contain 'Solution 2:'"
    
    # Write prompt and response to files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "test_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    
    with open(os.path.join(output_dir, "test_response.txt"), "w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"Test passed!")
    print(f"Prompt length: {len(prompt)}")
    print(f"First 200 chars of prompt: {prompt[:200]}...")
    print(f"First 200 chars of solution: {response[:200]}")
    print(f"Files written to: {output_dir}")
    
    return prompt, response


def test_convert_openmath_to_prompt_response2():
    """Test convert_openmath_to_prompt_response method with mock dataset sample 2."""
    
    # Load mock data instead of real dataset
    mock_sample = load_mock_data2()
    
    print("Mock sample 2 keys:", list(mock_sample.keys()))
    
    # Call the method under test
    prompt, response = convert_openmath_to_prompt_response(mock_sample)
    
    # Basic validation tests
    assert isinstance(prompt, str), "Prompt should be a string"
    assert len(prompt) > 0, "Prompt should not be empty"
    
    # The response should be the mathematical answer from the selected solution (judgment 0)
    assert response == "\\dfrac{1}{2}", f"Expected answer '\\dfrac{{1}}{{2}}', got '{response}'"
    
    # Response should not contain evaluation process text
    assert "Evaluation Process:" not in response, f"Response should not contain 'Evaluation Process:' but found it in: {response}"
    
    # Prompt should contain all solutions (0, 1, 2, and 3)
    assert "Solution 0:" in prompt, "Prompt should contain 'Solution 0:'"
    assert "Solution 1:" in prompt, "Prompt should contain 'Solution 1:'"
    assert "Solution 2:" in prompt, "Prompt should contain 'Solution 2:'"
    assert "Solution 3:" in prompt, "Prompt should contain 'Solution 3:'"
    
    # Write prompt and response to files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "test_prompt2.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    
    with open(os.path.join(output_dir, "test_response2.txt"), "w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"Test passed!")
    print(f"Prompt length: {len(prompt)}")
    print(f"First 200 chars of prompt: {prompt[:200]}...")
    print(f"First 200 chars of solution: {response[:200]}")
    print(f"Files written to: {output_dir}")
    
    return prompt, response


def test_extraction_functions():
    """Test individual extraction functions with mock data."""
    
    mock_sample = load_mock_data()
    
    # Import individual extraction functions
    from open_math_reasoning_genselect import extract_problem, extract_solutions_dict, extract_judgement
    
    # Test problem extraction
    problem = extract_problem(mock_sample["problem"])
    assert isinstance(problem, str), "Problem should be a string"
    assert len(problem) > 0, "Problem should not be empty"
    
    # Test solutions extraction
    solutions = extract_solutions_dict(mock_sample["problem"])
    assert isinstance(solutions, dict), "Solutions should be a dictionary"
    assert len(solutions) > 0, "Solutions should not be empty"
    for key, solution in solutions.items():
        assert isinstance(key, int), "Solution key should be integer"
        assert isinstance(solution, dict), "Solution should be dictionary"
        assert "answer" in solution, "Solution should have 'answer' key"
        assert "solution" in solution, "Solution should have 'solution' key"
    
    # Test judgment extraction
    judgement = extract_judgement(mock_sample["generated_solution"])
    assert isinstance(judgement, int), "Judgement should be an integer"
    assert judgement in solutions, f"Judgement {judgement} should exist in solutions {list(solutions.keys())}"
    
    print("Individual extraction functions test passed!")
    print(f"Extracted problem length: {len(problem)}")
    print(f"Solutions found: {list(solutions.keys())}")
    print(f"Selected judgement: {judgement}")
    
    return problem, solutions, judgement


def test_mock_data_structure():
    """Test that mock data has expected structure."""
    mock_sample = load_mock_data()
    
    # Check required keys
    required_keys = ["problem", "generated_solution", "expected_answer"]
    for key in required_keys:
        assert key in mock_sample, f"Required key '{key}' missing from mock sample"
        assert mock_sample[key] is not None, f"Key '{key}' should not be None"
        assert isinstance(mock_sample[key], str), f"Key '{key}' should be string"
        assert len(mock_sample[key]) > 0, f"Key '{key}' should not be empty"
    
    print("Mock data structure validation passed!")
    print("Problem snippet:", mock_sample["problem"][:100] + "...")
    print("Solution snippet:", mock_sample["generated_solution"][:100] + "...")


if __name__ == "__main__":
    print("=== Running Mock Data Structure Test ===")
    test_mock_data_structure()
    print("\n=== Running Individual Extraction Functions Test ===")
    test_extraction_functions()
    print("\n=== Running Main Conversion Test ===")
    test_convert_openmath_to_prompt_response()
    print("\n=== Running Main Conversion Test 2 ===")
    test_convert_openmath_to_prompt_response2()
    print("\n=== All tests passed! ===")