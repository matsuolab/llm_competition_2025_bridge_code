"""
Chat Template Adapter for various LLM models.

This module provides a unified interface for converting datasets to different chat formats.
All available templates are registered here, making it easy to add new ones.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json


class ChatTemplateAdapter(ABC):
    """Abstract base class for chat template adapters"""
    
    @abstractmethod
    def format_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a dataset item to the model's chat format"""
        pass
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Return the name of this template"""
        pass


class DeepSeekR1Adapter(ChatTemplateAdapter):
    """Chat template adapter for DeepSeek-R1 models"""
    
    def __init__(self):
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
    
    def get_template_name(self) -> str:
        return "deepseek-r1"
    
    def format_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dataset item to DeepSeek-R1 chat format.
        
        Args:
            item: Dict with 'question', 'think', and 'answer' fields
            
        Returns:
            Dict with 'messages' field containing the chat conversation
        """
        messages = []
        
        # User message (question)
        messages.append({
            "role": "user",
            "content": item.get("question", "")
        })
        
        # Assistant message (thinking + answer)
        assistant_content = ""
        
        # Add thinking process if available
        if item.get("think"):
            assistant_content += f"<｜thinking｜>\n{item['think']}\n<｜/thinking｜>\n\n"
        
        # Add answer
        assistant_content += item.get("answer", "")
        
        messages.append({
            "role": "assistant", 
            "content": assistant_content
        })
        
        # Format with special tokens
        formatted_text = self.bos_token
        
        for message in messages:
            if message["role"] == "user":
                formatted_text += f"<｜User｜>{message['content']}"
            else:
                formatted_text += f"<｜Assistant｜>{message['content']}<｜end▁of▁sentence｜>"
        
        return {
            "text": formatted_text,
            "messages": messages
        }


class Qwen3Adapter(ChatTemplateAdapter):
    """Chat template adapter for Qwen3 models"""
    
    def get_template_name(self) -> str:
        return "qwen3"
    
    def format_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dataset item to Qwen3 chat format.
        
        Args:
            item: Dict with 'question', 'think', and 'answer' fields
            
        Returns:
            Dict with 'messages' field containing the chat conversation
        """
        messages = []
        
        # System message (optional)
        messages.append({
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        })
        
        # User message
        messages.append({
            "role": "user",
            "content": item.get("question", "")
        })
        
        # Assistant message (combine thinking and answer)
        assistant_content = ""
        if item.get("think"):
            assistant_content += f"{item['think']}\n\n"
        assistant_content += item.get("answer", "")
        
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        return {
            "messages": messages
        }


class TemplateAdapterFactory:
    """Factory class to create appropriate template adapters"""
    
    # Registry of all available adapters
    _adapters = {
        'deepseek-r1': DeepSeekR1Adapter,
        'qwen3': Qwen3Adapter,
    }
    
    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get list of all available template names"""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_adapter(cls, model_type: str) -> ChatTemplateAdapter:
        """Get appropriate adapter for the given model type"""
        adapter_class = cls._adapters.get(model_type.lower())
        if not adapter_class:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {cls.get_available_templates()}")
        return adapter_class()
    
    @classmethod
    def register_adapter(cls, model_type: str, adapter_class: type):
        """Register a new adapter class"""
        cls._adapters[model_type.lower()] = adapter_class
    
    @classmethod
    def format_all_templates(cls, item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Apply all available templates to a dataset item.
        
        Args:
            item: Dataset item with question/think/answer fields
            
        Returns:
            Dict mapping template names to formatted items
        """
        results = {}
        for template_name in cls.get_available_templates():
            try:
                adapter = cls.get_adapter(template_name)
                results[template_name] = adapter.format_dataset_item(item)
            except Exception as e:
                print(f"Warning: Failed to apply {template_name} template: {e}")
        return results


def main():
    """Test the template adapters"""
    # Example dataset item
    test_item = {
        "question": "Pythonでリストを逆順にする方法を教えてください。",
        "think": "リストを逆順にする方法はいくつかあります。最も簡単な方法を説明します。",
        "answer": "Pythonでリストを逆順にする主な方法は以下の3つです：\n\n1. `reverse()`メソッド（元のリストを変更）\n2. スライシング `[::-1]`（新しいリストを作成）\n3. `reversed()`関数（イテレータを返す）"
    }
    
    factory = TemplateAdapterFactory()
    
    print("Available templates:", factory.get_available_templates())
    print("\nTesting all templates:")
    
    results = factory.format_all_templates(test_item)
    for template_name, formatted in results.items():
        print(f"\n--- {template_name} ---")
        print(json.dumps(formatted, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()