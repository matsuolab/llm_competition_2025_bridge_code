import pytest
from unittest.mock import patch, MagicMock
from filter_solvable_question.core.llm_solver import LLMSolver
from filter_solvable_question.utils.config import VLLMConfig

class TestLLMSolver:
    def test_init_with_model_name(self):
        solver = LLMSolver("Qwen/Qwen3-0.6B")
        assert solver.model_name == "Qwen/Qwen3-0.6B"
    
    @patch('filter_solvable_question.core.llm_solver.LLM')
    def test_initialize_model_creates_vllm_instance(self, mock_llm_class):
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        vllm_config = VLLMConfig(
            tensor_parallel_size=1,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            dtype="float16",
        )
        
        solver = LLMSolver("Qwen/Qwen3-0.6B", vllm_config)
        solver.initialize_model()
        
        model_args = {"model": "Qwen/Qwen3-0.6B"}
        model_args.update(vllm_config.dict())
        model_args.update({"rope_scaling": {"type": "yarn", "factor": 4.0, "max_position_embeddings": 32768}})
        mock_llm_class.assert_called_once_with(**model_args)
        assert solver.llm == mock_llm_instance
    
    @patch('filter_solvable_question.core.llm_solver.LLM')
    def test_solve_question_returns_generated_answer(self, mock_llm_class):
        mock_llm_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "The answer is 4"
        mock_llm_instance.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm_instance
        
        solver = LLMSolver("Qwen/Qwen3-0.6B")
        solver.initialize_model()
        
        question = "What is 2+2?"
        result = solver.solve_question(question)
        
        assert result == "The answer is 4"
        mock_llm_instance.generate.assert_called_once()
    
    def test_format_prompt_creates_proper_prompt(self):
        solver = LLMSolver("Qwen/Qwen3-0.6B")
        question = "What is 2+2?"
        
        prompt = solver.format_prompt(question)
        
        assert question in prompt
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "####" in prompt