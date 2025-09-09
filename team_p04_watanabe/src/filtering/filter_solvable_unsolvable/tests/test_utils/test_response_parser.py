import pytest
from filter_solvable_question.utils.response_parser import ResponseParser


class TestResponseParser:
    def test_parse_complete_response(self):
        """Test parsing a complete response with reasoning and answer."""
        raw_response = """Let me solve this step by step.

<think>
This is a math problem asking for 2 + 2.
I need to add 2 and 2 together.
2 + 2 = 4
</think>

The answer is:
#### 4
"""
        
        parser = ResponseParser()
        result = parser.parse_response(raw_response)
        
        assert result['raw_output'] == raw_response
        assert "This is a math problem asking for 2 + 2" in result['reasoning']
        assert "2 + 2 = 4" in result['reasoning']
        assert result['answer'] == "4"
    
    def test_parse_no_reasoning(self):
        """Test parsing response without reasoning section."""
        raw_response = """The answer is simple.
#### 42
"""
        
        parser = ResponseParser()
        result = parser.parse_response(raw_response)
        
        assert result['raw_output'] == raw_response
        assert result['reasoning'] is None
        assert result['answer'] == "42"
    
    def test_parse_no_answer(self):
        """Test parsing response without answer section."""
        raw_response = """<think>
Let me think about this problem.
This is complex.
</think>

I'm not sure about the answer."""
        
        parser = ResponseParser()
        result = parser.parse_response(raw_response)
        
        assert result['raw_output'] == raw_response
        assert "Let me think about this problem" in result['reasoning']
        assert result['answer'] is None
    
    def test_parse_multiline_answer(self):
        """Test parsing answer that continues on next line."""
        raw_response = """<think>
Calculate the area of a rectangle.
Length = 5, Width = 3
Area = 5 * 3 = 15
</think>

#### The area is 15 square units
Additional explanation here."""
        
        parser = ResponseParser()
        result = parser.parse_response(raw_response)
        
        assert result['reasoning'] is not None
        assert "Area = 5 * 3 = 15" in result['reasoning']
        assert result['answer'] == "The area is 15 square units"
    
    def test_parse_empty_sections(self):
        """Test parsing response with empty reasoning/answer sections."""
        raw_response = """<think>
</think>

#### 
"""
        
        parser = ResponseParser()
        result = parser.parse_response(raw_response)
        
        assert result['raw_output'] == raw_response
        assert result['reasoning'] is None  # Empty reasoning should be None
        assert result['answer'] is None     # Empty answer should be None
    
    def test_parse_batch(self):
        """Test parsing multiple responses at once."""
        responses = [
            "<think>Simple math</think>\n#### 5",
            "Just the answer\n#### 10",
            "<think>Complex reasoning here</think>\nNo final answer"
        ]
        
        parser = ResponseParser()
        results = parser.parse_batch(responses)
        
        assert len(results) == 3
        assert results[0]['reasoning'] == "Simple math"
        assert results[0]['answer'] == "5"
        assert results[1]['reasoning'] is None
        assert results[1]['answer'] == "10"
        assert results[2]['reasoning'] == "Complex reasoning here"
        assert results[2]['answer'] is None