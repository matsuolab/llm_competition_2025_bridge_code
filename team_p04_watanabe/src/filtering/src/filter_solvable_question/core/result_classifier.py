import json
from typing import List, Dict, Any, Optional


class ResultClassifier:
    def __init__(self):
        pass
    
    def classify_results(
        self,
        questions: List[str],
        correct_answers: List[str],
        evaluations: List[bool],
        llm_responses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if llm_responses is None:
            llm_responses = [""] * len(questions)
        
        if not (len(questions) == len(correct_answers) == len(evaluations) == len(llm_responses)):
            raise ValueError("All input lists must have the same length")
        
        solved = []
        unsolved = []
        
        for i, (question, answer, is_correct, llm_resp) in enumerate(
            zip(questions, correct_answers, evaluations, llm_responses)
        ):
            item = {
                'question': question,
                'answer': answer,
                'llm_response': llm_resp,
                'correct': is_correct,
                'index': i
            }
            
            if is_correct:
                solved.append(item)
            else:
                unsolved.append(item)
        
        statistics = self.get_statistics(evaluations)
        
        return {
            'solved': solved,
            'unsolved': unsolved,
            'statistics': statistics
        }
    
    def get_statistics(self, evaluations: List[bool]) -> Dict[str, float]:
        total = len(evaluations)
        if total == 0:
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        correct = sum(evaluations)
        incorrect = total - correct
        accuracy = correct / total
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy
        }
    
    def save_results_to_file(self, results: Dict[str, Any], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def load_results_from_file(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)